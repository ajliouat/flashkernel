"""
FlashKernel — Triton FlashAttention Tests (v1.0.3)

Validates correctness of the Triton FlashAttention kernel against
PyTorch's F.scaled_dot_product_attention (SDPA) and cross-validates
against the CUDA C++ kernel from v1.0.2.

Test matrix mirrors test_flash_attention.py (CUDA) plus:
  - Cross-validation: Triton output matches CUDA output
  - Autotuned configs behave correctly across all shapes

40+ test cases across 8 test classes.
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
def triton_attn():
    """Import Triton FlashAttention."""
    try:
        from src.triton.flash_attention import triton_flash_attention_forward
    except ImportError:
        from flashkernel._triton.flash_attention import triton_flash_attention_forward
    return triton_flash_attention_forward


@pytest.fixture(scope="module")
def cuda_attn():
    """Import CUDA FlashAttention (for cross-validation)."""
    try:
        import flashkernel
        return flashkernel.flash_attention_forward
    except (ImportError, RuntimeError):
        pytest.skip("CUDA FlashKernel extension not compiled")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def reference_attention(Q, K, V, scale=None, is_causal=False):
    """Reference attention using PyTorch SDPA."""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    with torch.no_grad():
        out = F.scaled_dot_product_attention(
            Q.float(), K.float(), V.float(),
            attn_mask=None,
            is_causal=is_causal,
            scale=scale,
        )
    return out.half()


def max_abs_error(a, b):
    return (a.float() - b.float()).abs().max().item()


def mean_abs_error(a, b):
    return (a.float() - b.float()).abs().mean().item()


def make_qkv(B, H, N, D, device="cuda", dtype=torch.float16):
    """Create random Q, K, V tensors."""
    Q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    K = torch.randn(B, H, N, D, device=device, dtype=dtype)
    V = torch.randn(B, H, N, D, device=device, dtype=dtype)
    return Q, K, V


# ═════════════════════════════════════════════════════════════════════════════
# CORRECTNESS — head_dim=64, non-causal
# ═════════════════════════════════════════════════════════════════════════════

class TestTritonFlashAttentionD64:
    """Triton FlashAttention with head_dim=64."""

    def test_basic_b1_h8_n512(self, triton_attn):
        """Basic correctness: B=1, H=8, N=512, d=64."""
        B, H, N, D = 1, 8, 512, 64
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        assert O.shape == (B, H, N, D)
        assert O.dtype == torch.float16
        err = max_abs_error(O, O_ref)
        assert err < 1e-2, f"Max abs error = {err} (expected < 1e-2)"

    def test_b4_h8_n1024(self, triton_attn):
        """Batched: B=4, H=8, N=1024, d=64."""
        B, H, N, D = 4, 8, 1024, 64
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        assert O.shape == (B, H, N, D)
        err = max_abs_error(O, O_ref)
        assert err < 1e-2, f"Max abs error = {err}"

    def test_b1_h8_n4096_no_oom(self, triton_attn):
        """Long sequence: B=1, H=8, N=4096, d=64 — no OOM."""
        B, H, N, D = 1, 8, 4096, 64
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        assert O.shape == (B, H, N, D)
        err = max_abs_error(O, O_ref)
        assert err < 5e-2, f"Max abs error = {err} (expected < 5e-2)"

    @pytest.mark.parametrize("N", [128, 256, 512, 1024, 2048])
    def test_various_seq_lengths(self, triton_attn, N):
        """Sweep sequence lengths for head_dim=64."""
        B, H, D = 2, 4, 64
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        err = max_abs_error(O, O_ref)
        tol = max(1e-2, N * 1e-5)
        assert err < tol, f"N={N}: max abs error = {err}"


# ═════════════════════════════════════════════════════════════════════════════
# CORRECTNESS — head_dim=128
# ═════════════════════════════════════════════════════════════════════════════

class TestTritonFlashAttentionD128:
    """Triton FlashAttention with head_dim=128."""

    def test_b8_h12_n2048(self, triton_attn):
        """B=8, H=12, N=2048, d=128 — large config."""
        B, H, N, D = 8, 12, 2048, 128
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        assert O.shape == (B, H, N, D)
        err = max_abs_error(O, O_ref)
        assert err < 5e-2, f"Max abs error = {err}"

    def test_b1_h8_n512(self, triton_attn):
        """Small config with head_dim=128."""
        B, H, N, D = 1, 8, 512, 128
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        err = max_abs_error(O, O_ref)
        assert err < 1e-2, f"Max abs error = {err}"

    @pytest.mark.parametrize("N", [128, 256, 512, 1024])
    def test_various_seq_lengths(self, triton_attn, N):
        """Sweep sequence lengths for head_dim=128."""
        B, H, D = 1, 4, 128
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        err = max_abs_error(O, O_ref)
        tol = max(2e-2, N * 2e-5)
        assert err < tol, f"N={N}: max abs error = {err}"


# ═════════════════════════════════════════════════════════════════════════════
# CAUSAL MASKING
# ═════════════════════════════════════════════════════════════════════════════

class TestTritonFlashAttentionCausal:
    """Causal (autoregressive) attention — upper triangle masked."""

    def test_causal_d64(self, triton_attn):
        B, H, N, D = 2, 8, 512, 64
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V, is_causal=True)
        O_ref = reference_attention(Q, K, V, is_causal=True)

        err = max_abs_error(O, O_ref)
        assert err < 1e-2, f"Causal d=64: max abs error = {err}"

    def test_causal_d128(self, triton_attn):
        B, H, N, D = 1, 4, 512, 128
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V, is_causal=True)
        O_ref = reference_attention(Q, K, V, is_causal=True)

        err = max_abs_error(O, O_ref)
        assert err < 2e-2, f"Causal d=128: max abs error = {err}"

    def test_causal_long_seq(self, triton_attn):
        """Causal N=2048 — verify early termination works."""
        B, H, N, D = 1, 4, 2048, 64
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V, is_causal=True)
        O_ref = reference_attention(Q, K, V, is_causal=True)

        err = max_abs_error(O, O_ref)
        assert err < 5e-2, f"Causal N=2048: max abs error = {err}"

    @pytest.mark.parametrize("N", [64, 128, 256, 512, 1024])
    def test_causal_sweep(self, triton_attn, N):
        B, H, D = 1, 4, 64
        Q, K, V = make_qkv(B, H, N, D)

        O, _ = triton_attn(Q, K, V, is_causal=True)
        O_ref = reference_attention(Q, K, V, is_causal=True)

        err = max_abs_error(O, O_ref)
        tol = max(1e-2, N * 1e-5)
        assert err < tol, f"Causal N={N}: max abs error = {err}"


# ═════════════════════════════════════════════════════════════════════════════
# NON-DIVISIBLE SEQUENCE LENGTHS
# ═════════════════════════════════════════════════════════════════════════════

class TestTritonFlashAttentionBoundary:
    """Sequence lengths that are not multiples of tile size."""

    @pytest.mark.parametrize("N", [65, 100, 127, 129, 200, 513, 1000])
    def test_non_divisible_d64(self, triton_attn, N):
        B, H, D = 1, 4, 64
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        err = max_abs_error(O, O_ref)
        tol = max(2e-2, N * 2e-5)
        assert err < tol, f"N={N} (non-divisible): max abs error = {err}"

    @pytest.mark.parametrize("N", [33, 63, 97, 255])
    def test_non_divisible_d128(self, triton_attn, N):
        B, H, D = 1, 2, 128
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)
        O_ref = reference_attention(Q, K, V)

        err = max_abs_error(O, O_ref)
        tol = max(2e-2, N * 2e-5)
        assert err < tol, f"N={N} d=128 (non-divisible): max abs error = {err}"


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT STRUCTURE & LOG-SUM-EXP
# ═════════════════════════════════════════════════════════════════════════════

class TestTritonFlashAttentionOutputs:
    """Validate output shapes, dtypes, and log-sum-exp values."""

    def test_output_shapes(self, triton_attn):
        B, H, N, D = 2, 4, 256, 64
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)

        assert O.shape == (B, H, N, D), f"O shape: {O.shape}"
        assert L.shape == (B, H, N), f"L shape: {L.shape}"
        assert O.dtype == torch.float16
        assert L.dtype == torch.float32
        assert O.device.type == "cuda"
        assert L.device.type == "cuda"

    def test_lse_finite(self, triton_attn):
        """Log-sum-exp values should be finite."""
        B, H, N, D = 1, 4, 256, 64
        Q, K, V = make_qkv(B, H, N, D)

        _, L = triton_attn(Q, K, V)
        assert torch.isfinite(L).all(), "LSE contains NaN or Inf"

    def test_output_finite(self, triton_attn):
        """Output should not contain NaN or Inf."""
        B, H, N, D = 2, 8, 512, 64
        Q, K, V = make_qkv(B, H, N, D)

        O, _ = triton_attn(Q, K, V)
        assert torch.isfinite(O.float()).all(), "Output contains NaN or Inf"

    def test_custom_scale(self, triton_attn):
        """Custom scale parameter."""
        B, H, N, D = 1, 4, 128, 64
        Q, K, V = make_qkv(B, H, N, D)

        custom_scale = 0.05
        O, _ = triton_attn(Q, K, V, scale=custom_scale)
        O_ref = reference_attention(Q, K, V, scale=custom_scale)

        err = max_abs_error(O, O_ref)
        assert err < 1e-2, f"Custom scale: max abs error = {err}"

    def test_output_shapes_d128(self, triton_attn):
        """Output shapes for head_dim=128."""
        B, H, N, D = 1, 4, 128, 128
        Q, K, V = make_qkv(B, H, N, D)

        O, L = triton_attn(Q, K, V)

        assert O.shape == (B, H, N, D)
        assert L.shape == (B, H, N)
        assert O.dtype == torch.float16
        assert L.dtype == torch.float32


# ═════════════════════════════════════════════════════════════════════════════
# DETERMINISM
# ═════════════════════════════════════════════════════════════════════════════

class TestTritonFlashAttentionDeterminism:
    """Same input → same output (no non-deterministic ops)."""

    def test_deterministic_d64(self, triton_attn):
        B, H, N, D = 1, 4, 256, 64
        Q, K, V = make_qkv(B, H, N, D)

        O1, L1 = triton_attn(Q, K, V)
        O2, L2 = triton_attn(Q, K, V)

        assert torch.equal(O1, O2), "Non-deterministic output O"
        assert torch.equal(L1, L2), "Non-deterministic output L"

    def test_deterministic_causal(self, triton_attn):
        B, H, N, D = 1, 4, 256, 64
        Q, K, V = make_qkv(B, H, N, D)

        O1, _ = triton_attn(Q, K, V, is_causal=True)
        O2, _ = triton_attn(Q, K, V, is_causal=True)

        assert torch.equal(O1, O2), "Non-deterministic causal output"

    def test_deterministic_d128(self, triton_attn):
        B, H, N, D = 1, 4, 256, 128
        Q, K, V = make_qkv(B, H, N, D)

        O1, L1 = triton_attn(Q, K, V)
        O2, L2 = triton_attn(Q, K, V)

        assert torch.equal(O1, O2), "Non-deterministic d=128 output O"
        assert torch.equal(L1, L2), "Non-deterministic d=128 output L"


# ═════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═════════════════════════════════════════════════════════════════════════════

class TestTritonFlashAttentionErrors:
    """Input validation and error cases."""

    def test_cpu_raises(self, triton_attn):
        Q, K, V = make_qkv(1, 4, 128, 64, device="cpu")
        with pytest.raises(RuntimeError):
            triton_attn(Q, K, V)

    def test_fp32_raises(self, triton_attn):
        Q, K, V = make_qkv(1, 4, 128, 64, dtype=torch.float32)
        with pytest.raises(RuntimeError):
            triton_attn(Q, K, V)

    def test_wrong_head_dim_raises(self, triton_attn):
        Q, K, V = make_qkv(1, 4, 128, 96)
        with pytest.raises(RuntimeError):
            triton_attn(Q, K, V)

    def test_shape_mismatch_raises(self, triton_attn):
        Q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
        K = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.float16)
        V = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError):
            triton_attn(Q, K, V)

    def test_3d_raises(self, triton_attn):
        Q = torch.randn(4, 128, 64, device="cuda", dtype=torch.float16)
        K = torch.randn(4, 128, 64, device="cuda", dtype=torch.float16)
        V = torch.randn(4, 128, 64, device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError):
            triton_attn(Q, K, V)


# ═════════════════════════════════════════════════════════════════════════════
# CUDA vs TRITON CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

class TestCUDAvsTritonCrossValidation:
    """CUDA and Triton FlashAttention produce the same output."""

    def test_cross_d64_noncausal(self, triton_attn, cuda_attn):
        """Both kernels agree on d=64 non-causal."""
        B, H, N, D = 2, 8, 512, 64
        Q, K, V = make_qkv(B, H, N, D)

        O_triton, L_triton = triton_attn(Q, K, V)
        O_cuda, L_cuda = cuda_attn(Q, K, V)

        err_o = max_abs_error(O_triton, O_cuda)
        err_l = (L_triton - L_cuda).abs().max().item()
        assert err_o < 1e-2, f"CUDA vs Triton O: max abs error = {err_o}"
        assert err_l < 1e-1, f"CUDA vs Triton LSE: max abs error = {err_l}"

    def test_cross_d128_noncausal(self, triton_attn, cuda_attn):
        """Both kernels agree on d=128 non-causal."""
        B, H, N, D = 1, 4, 512, 128
        Q, K, V = make_qkv(B, H, N, D)

        O_triton, L_triton = triton_attn(Q, K, V)
        O_cuda, L_cuda = cuda_attn(Q, K, V)

        err_o = max_abs_error(O_triton, O_cuda)
        assert err_o < 2e-2, f"CUDA vs Triton d=128 O: max abs error = {err_o}"

    def test_cross_d64_causal(self, triton_attn, cuda_attn):
        """Both kernels agree on d=64 causal."""
        B, H, N, D = 2, 8, 512, 64
        Q, K, V = make_qkv(B, H, N, D)

        O_triton, _ = triton_attn(Q, K, V, is_causal=True)
        O_cuda, _ = cuda_attn(Q, K, V, is_causal=True)

        err_o = max_abs_error(O_triton, O_cuda)
        assert err_o < 1e-2, f"CUDA vs Triton causal O: max abs error = {err_o}"

    def test_cross_d128_causal(self, triton_attn, cuda_attn):
        """Both kernels agree on d=128 causal."""
        B, H, N, D = 1, 4, 512, 128
        Q, K, V = make_qkv(B, H, N, D)

        O_triton, _ = triton_attn(Q, K, V, is_causal=True)
        O_cuda, _ = cuda_attn(Q, K, V, is_causal=True)

        err_o = max_abs_error(O_triton, O_cuda)
        assert err_o < 2e-2, f"CUDA vs Triton causal d=128: error = {err_o}"

    @pytest.mark.parametrize("N", [128, 256, 513, 1024])
    def test_cross_sweep(self, triton_attn, cuda_attn, N):
        """CUDA vs Triton agree across multiple seq lengths."""
        B, H, D = 1, 4, 64
        Q, K, V = make_qkv(B, H, N, D)

        O_triton, _ = triton_attn(Q, K, V)
        O_cuda, _ = cuda_attn(Q, K, V)

        err = max_abs_error(O_triton, O_cuda)
        tol = max(1e-2, N * 1e-5)
        assert err < tol, f"N={N} CUDA vs Triton: max abs error = {err}"

    def test_cross_large_batch(self, triton_attn, cuda_attn):
        """CUDA vs Triton agree on large batched input."""
        B, H, N, D = 4, 8, 1024, 64
        Q, K, V = make_qkv(B, H, N, D)

        O_triton, _ = triton_attn(Q, K, V)
        O_cuda, _ = cuda_attn(Q, K, V)

        err = max_abs_error(O_triton, O_cuda)
        assert err < 2e-2, f"Large batch CUDA vs Triton: error = {err}"
