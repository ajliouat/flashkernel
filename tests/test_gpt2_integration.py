"""
FlashKernel — GPT-2 Integration Tests (v1.0.7)

Verifies that monkey-patching GPT-2 with FlashKernel custom kernels:
  1. Produces identical greedy-decoded tokens as HF default
  2. Correctly patches/unpatches all attention and MLP layers
  3. Handles different sequence lengths and batch sizes
  4. Maintains numerical stability in fp16

Requires: transformers, torch (CUDA)
"""

import pytest
import torch


# ─── Skip if no CUDA ─────────────────────────────────────────────────────────

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS, reason="transformers not installed"
)

try:
    import flashkernel
    HAS_FLASHKERNEL = True
except ImportError:
    HAS_FLASHKERNEL = False

requires_flashkernel = pytest.mark.skipif(
    not HAS_FLASHKERNEL, reason="flashkernel C extension not built"
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2-124M once for all tests in this module."""
    if not HAS_TRANSFORMERS:
        pytest.skip("transformers not installed")
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().half()
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load GPT-2 tokenizer once for all tests."""
    if not HAS_TRANSFORMERS:
        pytest.skip("transformers not installed")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


# ─── Patch / Unpatch Tests ──────────────────────────────────────────────────

@requires_cuda
@requires_transformers
@requires_flashkernel
class TestPatchingMechanics:
    """Test that patching and unpatching works correctly."""

    def test_patch_attention_only(self, gpt2_model):
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_attention,
            unpatch_gpt2_model,
            count_patched_modules,
        )

        patch_gpt2_attention(gpt2_model)
        counts = count_patched_modules(gpt2_model)
        assert counts["attention"] == 12, "Should patch all 12 attention layers"
        assert counts["mlp"] == 0, "MLP should be unpatched"

        unpatch_gpt2_model(gpt2_model)
        counts = count_patched_modules(gpt2_model)
        assert counts["attention"] == 0
        assert counts["mlp"] == 0

    def test_patch_mlp_only(self, gpt2_model):
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_mlp,
            unpatch_gpt2_model,
            count_patched_modules,
        )

        patch_gpt2_mlp(gpt2_model)
        counts = count_patched_modules(gpt2_model)
        assert counts["attention"] == 0
        assert counts["mlp"] == 12, "Should patch all 12 MLP layers"

        unpatch_gpt2_model(gpt2_model)
        counts = count_patched_modules(gpt2_model)
        assert counts["mlp"] == 0

    def test_patch_full_model(self, gpt2_model):
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_model,
            unpatch_gpt2_model,
            count_patched_modules,
        )

        patch_gpt2_model(gpt2_model)
        counts = count_patched_modules(gpt2_model)
        assert counts["attention"] == 12
        assert counts["mlp"] == 12

        unpatch_gpt2_model(gpt2_model)
        counts = count_patched_modules(gpt2_model)
        assert counts["attention"] == 0
        assert counts["mlp"] == 0

    def test_double_patch_idempotent(self, gpt2_model):
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_model,
            unpatch_gpt2_model,
            count_patched_modules,
        )

        patch_gpt2_model(gpt2_model)
        patch_gpt2_model(gpt2_model)  # Should not break
        counts = count_patched_modules(gpt2_model)
        assert counts["attention"] == 12
        assert counts["mlp"] == 12

        unpatch_gpt2_model(gpt2_model)
        counts = count_patched_modules(gpt2_model)
        assert counts["attention"] == 0
        assert counts["mlp"] == 0

    def test_get_config_info(self, gpt2_model):
        from src.integration.gpt2_custom_kernels import get_gpt2_config_info

        info = get_gpt2_config_info(gpt2_model)
        assert info["n_layer"] == 12
        assert info["n_head"] == 12
        assert info["n_embd"] == 768
        assert info["head_dim"] == 64
        assert info["n_params_m"] > 100  # ~124M params


# ─── Greedy Decoding Identity ───────────────────────────────────────────────

@requires_cuda
@requires_transformers
@requires_flashkernel
class TestGreedyIdentity:
    """Verify FlashKernel produces identical greedy output as HF default."""

    def _generate(self, model, input_ids, max_new_tokens=32):
        """Helper: greedy generation."""
        with torch.no_grad():
            return model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False
            )

    def test_greedy_short_prompt(self, gpt2_model, tokenizer):
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_model,
            unpatch_gpt2_model,
        )

        prompt = "The quick brown fox"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

        # Default output
        hf_output = self._generate(gpt2_model, input_ids, max_new_tokens=32)

        # FlashKernel output
        patch_gpt2_model(gpt2_model)
        fk_output = self._generate(gpt2_model, input_ids, max_new_tokens=32)
        unpatch_gpt2_model(gpt2_model)

        assert torch.equal(hf_output, fk_output), (
            f"Greedy tokens differ!\n"
            f"HF:  {hf_output[0].tolist()}\n"
            f"FK:  {fk_output[0].tolist()}"
        )

    def test_greedy_medium_prompt(self, gpt2_model, tokenizer):
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_model,
            unpatch_gpt2_model,
        )

        prompt = "In the field of artificial intelligence, " * 16  # ~128 tokens
        input_ids = tokenizer.encode(prompt, return_tensors="pt")[:, :128].cuda()

        hf_output = self._generate(gpt2_model, input_ids, max_new_tokens=64)

        patch_gpt2_model(gpt2_model)
        fk_output = self._generate(gpt2_model, input_ids, max_new_tokens=64)
        unpatch_gpt2_model(gpt2_model)

        assert torch.equal(hf_output, fk_output), "Greedy tokens differ for medium prompt"

    def test_attention_only_preserves_output(self, gpt2_model, tokenizer):
        """Patching only attention should still give identical output."""
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_attention,
            unpatch_gpt2_model,
        )

        prompt = "Hello world"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

        hf_output = self._generate(gpt2_model, input_ids, max_new_tokens=16)

        patch_gpt2_attention(gpt2_model)
        fk_output = self._generate(gpt2_model, input_ids, max_new_tokens=16)
        unpatch_gpt2_model(gpt2_model)

        assert torch.equal(hf_output, fk_output), "Attention-only patch changed output"

    def test_mlp_only_preserves_output(self, gpt2_model, tokenizer):
        """Patching only MLP should still give identical output."""
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_mlp,
            unpatch_gpt2_model,
        )

        prompt = "Hello world"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

        hf_output = self._generate(gpt2_model, input_ids, max_new_tokens=16)

        patch_gpt2_mlp(gpt2_model)
        fk_output = self._generate(gpt2_model, input_ids, max_new_tokens=16)
        unpatch_gpt2_model(gpt2_model)

        assert torch.equal(hf_output, fk_output), "MLP-only patch changed output"


# ─── Forward Pass Smoke Tests ───────────────────────────────────────────────

@requires_cuda
@requires_transformers
@requires_flashkernel
class TestForwardPass:
    """Verify patched model runs without errors on various inputs."""

    def test_single_token_input(self, gpt2_model, tokenizer):
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_model,
            unpatch_gpt2_model,
        )

        input_ids = torch.tensor([[50256]]).cuda()  # <|endoftext|>

        patch_gpt2_model(gpt2_model)
        with torch.no_grad():
            output = gpt2_model(input_ids)
        unpatch_gpt2_model(gpt2_model)

        assert output.logits.shape == (1, 1, 50257)
        assert not torch.isnan(output.logits).any()
        assert not torch.isinf(output.logits).any()

    def test_batch_input(self, gpt2_model, tokenizer):
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_model,
            unpatch_gpt2_model,
        )

        # Batch of 4 sequences, length 16
        input_ids = torch.randint(0, 50257, (4, 16)).cuda()

        patch_gpt2_model(gpt2_model)
        with torch.no_grad():
            output = gpt2_model(input_ids)
        unpatch_gpt2_model(gpt2_model)

        assert output.logits.shape == (4, 16, 50257)
        assert not torch.isnan(output.logits).any()

    def test_long_sequence(self, gpt2_model, tokenizer):
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_model,
            unpatch_gpt2_model,
        )

        # 512 tokens
        input_ids = torch.randint(0, 50257, (1, 512)).cuda()

        patch_gpt2_model(gpt2_model)
        with torch.no_grad():
            output = gpt2_model(input_ids)
        unpatch_gpt2_model(gpt2_model)

        assert output.logits.shape == (1, 512, 50257)
        assert not torch.isnan(output.logits).any()


# ─── Generation Quality Tests ───────────────────────────────────────────────

@requires_cuda
@requires_transformers
@requires_flashkernel
class TestGenerationQuality:
    """Verify generated text is coherent (basic sanity checks)."""

    def test_generates_coherent_text(self, gpt2_model, tokenizer):
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_model,
            unpatch_gpt2_model,
        )

        prompt = "Once upon a time in a land far away,"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

        patch_gpt2_model(gpt2_model)
        with torch.no_grad():
            output = gpt2_model.generate(
                input_ids, max_new_tokens=50, do_sample=False
            )
        unpatch_gpt2_model(gpt2_model)

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Basic sanity: output should be longer than input
        assert len(text) > len(prompt)
        # Should contain mostly ASCII / printable
        assert sum(c.isalpha() or c.isspace() for c in text) / len(text) > 0.7

    def test_deterministic_generation(self, gpt2_model, tokenizer):
        """Two greedy runs should produce identical output."""
        from src.integration.gpt2_custom_kernels import (
            patch_gpt2_model,
            unpatch_gpt2_model,
        )

        prompt = "The meaning of life is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

        patch_gpt2_model(gpt2_model)
        with torch.no_grad():
            out1 = gpt2_model.generate(input_ids, max_new_tokens=32, do_sample=False)
            out2 = gpt2_model.generate(input_ids, max_new_tokens=32, do_sample=False)
        unpatch_gpt2_model(gpt2_model)

        assert torch.equal(out1, out2), "Greedy decoding is not deterministic"
