"""
FlashKernel — FlashAttention Forward Tests (v1.0.2)

Validates correctness of the tiled FlashAttention CUDA kernel against
PyTorch's F.scaled_dot_product_attention (SDPA).

Test matrix from ROADMAP.md:
  | Config                        | Check                          |
  |-------------------------------|--------------------------------|
  | (B=1, H=8, N=512, d=64) fp16 | Correctness vs PyTorch SDPA    |
  | (B=4, H=8, N=1024, d=64)     | Correctness + no OOM           |
  | (B=1, H=8, N=4096, d=64)     | No OOM (key benefit)           |
  | (B=8, H=12, N=2048, d=128)   | Correctness with fallback tiles|
  | Causal mask variant           | Upper triangle masked          |

Additional tests:
  - Non-divisible sequence lengths (N not multiple of tile size)
  - Determinism (same input → same output)
  - Output shape validation
  - Log-sum-exp output validation
"""

import math
import pytest
import torch
import torch.nn.functional as F

# Skip entire module if no CUDA GPU available
pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def fk():
    """Import flashkernel."""
    import flashkernel
    return flashkernel


# ─── Helpers ─────────────────────────────────────────────────────────────────

def reference_attention(Q, K, V, scale=None, is_causal=False):
    """
    Reference attention using PyTorch SDPA.
    Input/output: [B, H, N, D] fp16
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])

    # PyTorch SDPA expects the same layout
    with torch.no_grad():
        out = F.scaled_dot_product_attention(
            Q.float(), K.float(), V.float(),
            attn_mask=None,
            is_causal=is_causal,
            scale=scale,
        )
    return out.half()


def max_abs_error(a, b):
    """Compute max absolute error between two tensors."""
    return (a.float() - b.float()).abs().max().item()


def mean_abs_error(a, b):
    """Compute mean absolute error."""
    return (a.float() - b.float()).abs().mean().item()


# ═════════════════════════════════════════════════════════════════════════════
# CORRECTNESS — head_dim=64, non-causal
# ═════════════════════════════════════════════════════════════════════════════

class TestFlashAttentionD64:
    """FlashAttention with head_dim=64 — primary tile config (64×64)."""

    def test_basic_b1_h8_n512(self, fk):
        """ROADMAP: (B=1, H=8, N=512, d=64) — basic correctness."""
        B, H, N, D = 1, 8, 512, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        assert O.shape == (B, H, N, D)
        assert O.dtype == torch.float16
        err = max_abs_error(O, O_ref)
        assert err < 1e-2, f"Max abs error = {err} (expected < 1e-2)"

    def test_b4_h8_n1024(self, fk):
        """ROADMAP: (B=4, H=8, N=1024, d=64) — batched, correctness + no OOM."""
        B, H, N, D = 4, 8, 1024, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        assert O.shape == (B, H, N, D)
        err = max_abs_error(O, O_ref)
        assert err < 1e-2, f"Max abs error = {err}"

    def test_b1_h8_n4096_no_oom(self, fk):
        """ROADMAP: (B=1, H=8, N=4096, d=64) — long sequence, no OOM."""
        B, H, N, D = 1, 8, 4096, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        # Should not OOM — this is the key benefit of FlashAttention
        O, L = fk.flash_attention_forward(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        assert O.shape == (B, H, N, D)
        err = max_abs_error(O, O_ref)
        # Longer sequences accumulate more error
        assert err < 5e-2, f"Max abs error = {err} (expected < 5e-2)"

    @pytest.mark.parametrize("N", [128, 256, 512, 1024, 2048])
    def test_various_seq_lengths(self, fk, N):
        """Sweep sequence lengths for head_dim=64."""
        B, H, D = 2, 4, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        err = max_abs_error(O, O_ref)
        # Tolerance scales slightly with N
        tol = max(1e-2, N * 1e-5)
        assert err < tol, f"N={N}: max abs error = {err}"


# ═════════════════════════════════════════════════════════════════════════════
# CORRECTNESS — head_dim=128, fallback tile config (32×64)
# ═════════════════════════════════════════════════════════════════════════════

class TestFlashAttentionD128:
    """FlashAttention with head_dim=128 — fallback tile config (32×64)."""

    def test_b8_h12_n2048(self, fk):
        """ROADMAP: (B=8, H=12, N=2048, d=128) — correctness with fallback tiles."""
        B, H, N, D = 8, 12, 2048, 128
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        assert O.shape == (B, H, N, D)
        err = max_abs_error(O, O_ref)
        assert err < 5e-2, f"Max abs error = {err}"

    def test_b1_h8_n512(self, fk):
        """Small config with head_dim=128."""
        B, H, N, D = 1, 8, 512, 128
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        err = max_abs_error(O, O_ref)
        assert err < 1e-2, f"Max abs error = {err}"

    @pytest.mark.parametrize("N", [128, 256, 512, 1024])
    def test_various_seq_lengths(self, fk, N):
        """Sweep sequence lengths for head_dim=128."""
        B, H, D = 1, 4, 128
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        err = max_abs_error(O, O_ref)
        tol = max(2e-2, N * 2e-5)
        assert err < tol, f"N={N}: max abs error = {err}"


# ═════════════════════════════════════════════════════════════════════════════
# CAUSAL MASKING
# ═════════════════════════════════════════════════════════════════════════════

class TestFlashAttentionCausal:
    """Causal (autoregressive) attention — upper triangle masked to -inf."""

    def test_causal_d64(self, fk):
        """Causal attention with head_dim=64."""
        B, H, N, D = 2, 8, 512, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V, is_causal=True)
        O_ref = reference_attention(Q, K, V, is_causal=True)

        err = max_abs_error(O, O_ref)
        assert err < 1e-2, f"Causal d=64: max abs error = {err}"

    def test_causal_d128(self, fk):
        """Causal attention with head_dim=128."""
        B, H, N, D = 1, 4, 512, 128
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V, is_causal=True)
        O_ref = reference_attention(Q, K, V, is_causal=True)

        err = max_abs_error(O, O_ref)
        assert err < 2e-2, f"Causal d=128: max abs error = {err}"

    def test_causal_long_seq(self, fk):
        """Causal with N=2048 — verify early termination in KV loop."""
        B, H, N, D = 1, 4, 2048, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V, is_causal=True)
        O_ref = reference_attention(Q, K, V, is_causal=True)

        err = max_abs_error(O, O_ref)
        assert err < 5e-2, f"Causal N=2048: max abs error = {err}"

    @pytest.mark.parametrize("N", [64, 128, 256, 512, 1024])
    def test_causal_sweep(self, fk, N):
        """Sweep causal attention across seq lengths."""
        B, H, D = 1, 4, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, _ = fk.flash_attention_forward(Q, K, V, is_causal=True)
        O_ref = reference_attention(Q, K, V, is_causal=True)

        err = max_abs_error(O, O_ref)
        tol = max(1e-2, N * 1e-5)
        assert err < tol, f"Causal N={N}: max abs error = {err}"


# ═════════════════════════════════════════════════════════════════════════════
# NON-DIVISIBLE SEQUENCE LENGTHS
# ═════════════════════════════════════════════════════════════════════════════

class TestFlashAttentionBoundary:
    """Sequence lengths that are not multiples of tile size."""

    @pytest.mark.parametrize("N", [65, 100, 127, 129, 200, 513, 1000])
    def test_non_divisible_d64(self, fk, N):
        """N not divisible by Br=64 or Bc=64."""
        B, H, D = 1, 4, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        err = max_abs_error(O, O_ref)
        tol = max(2e-2, N * 2e-5)
        assert err < tol, f"N={N} (non-divisible): max abs error = {err}"

    @pytest.mark.parametrize("N", [33, 63, 97, 255])
    def test_non_divisible_d128(self, fk, N):
        """N not divisible by Br=32 or Bc=64, head_dim=128."""
        B, H, D = 1, 2, 128
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        err = max_abs_error(O, O_ref)
        tol = max(2e-2, N * 2e-5)
        assert err < tol, f"N={N} d=128 (non-divisible): max abs error = {err}"


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT STRUCTURE & LOG-SUM-EXP
# ═════════════════════════════════════════════════════════════════════════════

class TestFlashAttentionOutputs:
    """Validate output shapes, dtypes, and log-sum-exp values."""

    def test_output_shapes(self, fk):
        B, H, N, D = 2, 4, 256, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, L = fk.flash_attention_forward(Q, K, V)

        assert O.shape == (B, H, N, D), f"O shape: {O.shape}"
        assert L.shape == (B, H, N), f"L shape: {L.shape}"
        assert O.dtype == torch.float16
        assert L.dtype == torch.float32
        assert O.device.type == "cuda"
        assert L.device.type == "cuda"

    def test_lse_finite(self, fk):
        """Log-sum-exp values should be finite (not NaN/Inf)."""
        B, H, N, D = 1, 4, 256, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        _, L = fk.flash_attention_forward(Q, K, V)
        assert torch.isfinite(L).all(), "LSE contains NaN or Inf"

    def test_output_finite(self, fk):
        """Output should not contain NaN or Inf."""
        B, H, N, D = 2, 8, 512, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O, _ = fk.flash_attention_forward(Q, K, V)
        assert torch.isfinite(O.float()).all(), "Output contains NaN or Inf"

    def test_custom_scale(self, fk):
        """Custom scale parameter should be respected."""
        B, H, N, D = 1, 4, 128, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        custom_scale = 0.05
        O, _ = fk.flash_attention_forward(Q, K, V, scale=custom_scale)
        O_ref = reference_attention(Q, K, V, scale=custom_scale)

        err = max_abs_error(O, O_ref)
        assert err < 1e-2, f"Custom scale: max abs error = {err}"


# ═════════════════════════════════════════════════════════════════════════════
# DETERMINISM
# ═════════════════════════════════════════════════════════════════════════════

class TestFlashAttentionDeterminism:
    """Same input → same output (no non-deterministic ops)."""

    def test_deterministic_d64(self, fk):
        B, H, N, D = 1, 4, 256, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O1, L1 = fk.flash_attention_forward(Q, K, V)
        O2, L2 = fk.flash_attention_forward(Q, K, V)

        assert torch.equal(O1, O2), "Non-deterministic output O"
        assert torch.equal(L1, L2), "Non-deterministic output L"

    def test_deterministic_causal(self, fk):
        B, H, N, D = 1, 4, 256, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        O1, _ = fk.flash_attention_forward(Q, K, V, is_causal=True)
        O2, _ = fk.flash_attention_forward(Q, K, V, is_causal=True)

        assert torch.equal(O1, O2), "Non-deterministic causal output"


# ═════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═════════════════════════════════════════════════════════════════════════════

class TestFlashAttentionErrors:
    """Input validation and error cases."""

    def test_cpu_raises(self, fk):
        B, H, N, D = 1, 4, 128, 64
        Q = torch.randn(B, H, N, D, dtype=torch.float16)  # CPU
        K = torch.randn(B, H, N, D, dtype=torch.float16)
        V = torch.randn(B, H, N, D, dtype=torch.float16)
        with pytest.raises(RuntimeError):
            fk.flash_attention_forward(Q, K, V)

    def test_fp32_raises(self, fk):
        B, H, N, D = 1, 4, 128, 64
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError):
            fk.flash_attention_forward(Q, K, V)

    def test_wrong_head_dim_raises(self, fk):
        B, H, N, D = 1, 4, 128, 96  # unsupported head_dim
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError):
            fk.flash_attention_forward(Q, K, V)

    def test_shape_mismatch_raises(self, fk):
        Q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
        K = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.float16)  # different N
        V = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError):
            fk.flash_attention_forward(Q, K, V)

    def test_3d_raises(self, fk):
        """Must be 4-D [B, H, N, D]."""
        Q = torch.randn(4, 128, 64, device="cuda", dtype=torch.float16)
        K = torch.randn(4, 128, 64, device="cuda", dtype=torch.float16)
        V = torch.randn(4, 128, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError):
            fk.flash_attention_forward(Q, K, V)
