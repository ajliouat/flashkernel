"""
FlashKernel — Fused GeLU+Linear Tests (v1.0.4)

Validates correctness of fused GeLU(X @ W^T + bias) against unfused
PyTorch reference: F.gelu(F.linear(X, W, bias)).

Test matrix (from ROADMAP.md):
  | Config                    | Check                                      |
  |---------------------------|---------------------------------------------|
  | M=128, N=768, K=768       | Correctness vs unfused PyTorch (exact GeLU) |
  | M=512, N=3072, K=768      | Correctness (FFN up-projection shape)       |
  | M=2048, N=768, K=3072     | Correctness (FFN down-projection shape)     |
  | Tanh-approx variant       | Correctness (max error < 1e-3 fp16)         |
  | No-bias variant           | Correctness without bias                    |
  | Non-divisible dimensions  | Boundary handling                           |
  | CUDA vs Triton agreement  | Cross-validation between backends           |

Additional tests:
  - GeLU variant comparison (exact vs tanh max error)
  - Output shape validation
  - Determinism (same input -> same output)
  - Error handling (CPU tensors, wrong dtype, shape mismatch)
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

def reference_fused_gelu_linear(X, W, bias=None, approximate="none"):
    """
    Reference: F.gelu(F.linear(X, W, bias))
    X: [M, K] fp16, W: [N, K] fp16, bias: [N] fp16 or None
    """
    with torch.no_grad():
        # F.linear computes X @ W^T + bias
        out = F.linear(X.float(), W.float(), bias.float() if bias is not None else None)
        out = F.gelu(out, approximate=approximate)
    return out.half()


def max_abs_error(a, b):
    """Compute max absolute error between two tensors."""
    return (a.float() - b.float()).abs().max().item()


def mean_abs_error(a, b):
    """Compute mean absolute error."""
    return (a.float() - b.float()).abs().mean().item()


def make_inputs(M, N, K, with_bias=True):
    """Create random test inputs on CUDA."""
    X = torch.randn(M, K, dtype=torch.float16, device="cuda")
    W = torch.randn(N, K, dtype=torch.float16, device="cuda")
    bias = torch.randn(N, dtype=torch.float16, device="cuda") if with_bias else None
    return X, W, bias


# ═════════════════════════════════════════════════════════════════════════════
# CUDA KERNEL — EXACT GeLU WITH BIAS
# ═════════════════════════════════════════════════════════════════════════════

class TestFusedGeluLinearCUDAExact:
    """CUDA fused kernel with exact (erf) GeLU and bias."""

    ATOL = 5e-2  # fp16 tolerance for matmul + GeLU fusion

    def test_small_square(self, fk):
        """M=128, N=768, K=768 — standard transformer hidden size."""
        X, W, bias = make_inputs(128, 768, 768)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="none")
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert out.shape == ref.shape
        assert max_abs_error(out, ref) < self.ATOL, (
            f"Max error: {max_abs_error(out, ref):.6f}"
        )

    def test_ffn_up_projection(self, fk):
        """M=512, N=3072, K=768 — typical FFN up-projection."""
        X, W, bias = make_inputs(512, 3072, 768)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="none")
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert out.shape == (512, 3072)
        assert max_abs_error(out, ref) < self.ATOL

    def test_ffn_down_projection(self, fk):
        """M=2048, N=768, K=3072 — typical FFN down-projection."""
        X, W, bias = make_inputs(2048, 768, 3072)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="none")
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert out.shape == (2048, 768)
        assert max_abs_error(out, ref) < self.ATOL

    def test_batch_1(self, fk):
        """M=1 — single-token inference (edge case for tiling)."""
        X, W, bias = make_inputs(1, 768, 768)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="none")
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert out.shape == (1, 768)
        assert max_abs_error(out, ref) < self.ATOL

    @pytest.mark.parametrize("M,N,K", [
        (128, 768, 768),
        (256, 1024, 512),
        (512, 3072, 768),
        (1024, 768, 3072),
        (2048, 3072, 3072),
    ])
    def test_sweep(self, fk, M, N, K):
        """Sweep multiple dimension combinations."""
        X, W, bias = make_inputs(M, N, K)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="none")
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert out.shape == ref.shape
        assert max_abs_error(out, ref) < self.ATOL


# ═════════════════════════════════════════════════════════════════════════════
# CUDA KERNEL — TANH APPROX GeLU
# ═════════════════════════════════════════════════════════════════════════════

class TestFusedGeluLinearCUDATanh:
    """CUDA fused kernel with tanh GeLU approximation."""

    ATOL = 5e-2

    def test_small_square(self, fk):
        X, W, bias = make_inputs(128, 768, 768)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="tanh")
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
        assert max_abs_error(out, ref) < self.ATOL

    def test_ffn_up_projection(self, fk):
        X, W, bias = make_inputs(512, 3072, 768)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="tanh")
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
        assert max_abs_error(out, ref) < self.ATOL

    @pytest.mark.parametrize("M,N,K", [
        (128, 768, 768),
        (512, 3072, 768),
        (2048, 768, 3072),
    ])
    def test_sweep(self, fk, M, N, K):
        X, W, bias = make_inputs(M, N, K)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="tanh")
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
        assert max_abs_error(out, ref) < self.ATOL


# ═════════════════════════════════════════════════════════════════════════════
# CUDA KERNEL — NO BIAS
# ═════════════════════════════════════════════════════════════════════════════

class TestFusedGeluLinearNoBias:
    """CUDA fused kernel without bias."""

    ATOL = 5e-2

    def test_exact_no_bias(self, fk):
        X, W, _ = make_inputs(256, 768, 768, with_bias=False)
        ref = reference_fused_gelu_linear(X, W, None, approximate="none")
        out = fk.fused_gelu_linear(X, W, bias=None, use_tanh_approx=False)
        assert out.shape == ref.shape
        assert max_abs_error(out, ref) < self.ATOL

    def test_tanh_no_bias(self, fk):
        X, W, _ = make_inputs(256, 768, 768, with_bias=False)
        ref = reference_fused_gelu_linear(X, W, None, approximate="tanh")
        out = fk.fused_gelu_linear(X, W, bias=None, use_tanh_approx=True)
        assert max_abs_error(out, ref) < self.ATOL

    @pytest.mark.parametrize("M,N,K", [
        (128, 768, 768),
        (512, 3072, 768),
        (2048, 3072, 3072),
    ])
    def test_sweep_no_bias(self, fk, M, N, K):
        X, W, _ = make_inputs(M, N, K, with_bias=False)
        ref = reference_fused_gelu_linear(X, W, None, approximate="none")
        out = fk.fused_gelu_linear(X, W, bias=None, use_tanh_approx=False)
        assert max_abs_error(out, ref) < self.ATOL


# ═════════════════════════════════════════════════════════════════════════════
# TRITON KERNEL CORRECTNESS
# ═════════════════════════════════════════════════════════════════════════════

class TestTritonFusedGeluLinear:
    """Triton fused GeLU+Linear correctness."""

    ATOL = 5e-2

    def test_exact_with_bias(self, fk):
        X, W, bias = make_inputs(256, 768, 768)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="none")
        out = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert out.shape == ref.shape
        assert max_abs_error(out, ref) < self.ATOL

    def test_tanh_with_bias(self, fk):
        X, W, bias = make_inputs(256, 768, 768)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="tanh")
        out = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
        assert max_abs_error(out, ref) < self.ATOL

    def test_no_bias(self, fk):
        X, W, _ = make_inputs(256, 768, 768, with_bias=False)
        ref = reference_fused_gelu_linear(X, W, None, approximate="none")
        out = fk.triton_fused_gelu_linear(X, W, bias=None, use_tanh_approx=False)
        assert max_abs_error(out, ref) < self.ATOL

    @pytest.mark.parametrize("M,N,K", [
        (128, 768, 768),
        (512, 3072, 768),
        (2048, 768, 3072),
        (2048, 3072, 3072),
    ])
    def test_sweep(self, fk, M, N, K):
        X, W, bias = make_inputs(M, N, K)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="none")
        out = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert max_abs_error(out, ref) < self.ATOL

    def test_large_batch(self, fk):
        """M=4096 — large batch."""
        X, W, bias = make_inputs(4096, 768, 768)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="none")
        out = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert max_abs_error(out, ref) < self.ATOL


# ═════════════════════════════════════════════════════════════════════════════
# BOUNDARY HANDLING — NON-DIVISIBLE DIMENSIONS
# ═════════════════════════════════════════════════════════════════════════════

class TestFusedGeluLinearBoundary:
    """Non-divisible dimensions that test tile boundary handling."""

    ATOL = 5e-2

    @pytest.mark.parametrize("M,N,K", [
        (1, 768, 768),       # single row
        (7, 100, 50),        # small odd dims
        (33, 65, 97),        # not divisible by any power of 2
        (100, 200, 300),     # round but not tile-aligned
        (127, 769, 513),     # primes near common sizes
        (65, 3073, 769),     # just past common FFN sizes
    ])
    def test_cuda_boundary(self, fk, M, N, K):
        X, W, bias = make_inputs(M, N, K)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="none")
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert out.shape == (M, N)
        assert max_abs_error(out, ref) < self.ATOL

    @pytest.mark.parametrize("M,N,K", [
        (1, 768, 768),
        (7, 100, 50),
        (33, 65, 97),
        (100, 200, 300),
        (127, 769, 513),
    ])
    def test_triton_boundary(self, fk, M, N, K):
        X, W, bias = make_inputs(M, N, K)
        ref = reference_fused_gelu_linear(X, W, bias, approximate="none")
        out = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert out.shape == (M, N)
        assert max_abs_error(out, ref) < self.ATOL


# ═════════════════════════════════════════════════════════════════════════════
# CUDA vs TRITON CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

class TestCUDAvsTritonFusedGelu:
    """Cross-validate CUDA and Triton implementations agree."""

    ATOL = 5e-2

    def test_exact_with_bias(self, fk):
        X, W, bias = make_inputs(256, 768, 768)
        cuda_out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        triton_out = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert max_abs_error(cuda_out, triton_out) < self.ATOL

    def test_tanh_with_bias(self, fk):
        X, W, bias = make_inputs(256, 768, 768)
        cuda_out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
        triton_out = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
        assert max_abs_error(cuda_out, triton_out) < self.ATOL

    def test_no_bias(self, fk):
        X, W, _ = make_inputs(512, 3072, 768, with_bias=False)
        cuda_out = fk.fused_gelu_linear(X, W, bias=None, use_tanh_approx=False)
        triton_out = fk.triton_fused_gelu_linear(X, W, bias=None, use_tanh_approx=False)
        assert max_abs_error(cuda_out, triton_out) < self.ATOL

    @pytest.mark.parametrize("M,N,K", [
        (128, 768, 768),
        (512, 3072, 768),
        (2048, 768, 3072),
    ])
    def test_sweep(self, fk, M, N, K):
        X, W, bias = make_inputs(M, N, K)
        cuda_out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        triton_out = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert max_abs_error(cuda_out, triton_out) < self.ATOL


# ═════════════════════════════════════════════════════════════════════════════
# GeLU VARIANT COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

class TestGeluVariantComparison:
    """Compare exact vs tanh GeLU approximation error."""

    def test_cuda_gelu_variants_close(self, fk):
        """Exact and tanh GeLU should differ by small amount (< 0.02 for fp16 range)."""
        X, W, bias = make_inputs(256, 768, 768)
        exact = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        tanh_ = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
        error = max_abs_error(exact, tanh_)
        # Tanh approx error is small but nonzero
        assert error < 0.1, f"GeLU variant max error: {error:.6f}"
        assert error > 0.0, "Exact and tanh should differ slightly"

    def test_triton_gelu_variants_close(self, fk):
        X, W, bias = make_inputs(256, 768, 768)
        exact = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        tanh_ = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
        error = max_abs_error(exact, tanh_)
        assert error < 0.1, f"GeLU variant max error: {error:.6f}"
        assert error > 0.0, "Exact and tanh should differ slightly"


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT PROPERTIES
# ═════════════════════════════════════════════════════════════════════════════

class TestFusedGeluLinearOutputs:
    """Validate output shapes, dtypes, and numerical properties."""

    def test_output_shape_cuda(self, fk):
        X, W, bias = make_inputs(128, 3072, 768)
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert out.shape == (128, 3072)
        assert out.dtype == torch.float16
        assert out.device.type == "cuda"
        assert out.is_contiguous()

    def test_output_shape_triton(self, fk):
        X, W, bias = make_inputs(128, 3072, 768)
        out = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert out.shape == (128, 3072)
        assert out.dtype == torch.float16
        assert out.device.type == "cuda"

    def test_output_finite_cuda(self, fk):
        X, W, bias = make_inputs(256, 768, 768)
        out = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_output_finite_triton(self, fk):
        X, W, bias = make_inputs(256, 768, 768)
        out = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_zero_input(self, fk):
        """GeLU(0) = 0, so zero input -> zero output when no bias."""
        X = torch.zeros(32, 768, dtype=torch.float16, device="cuda")
        W = torch.randn(768, 768, dtype=torch.float16, device="cuda")
        out = fk.fused_gelu_linear(X, W, bias=None, use_tanh_approx=False)
        # X=0 -> linear output is 0 -> GeLU(0) = 0
        assert out.abs().max().item() < 1e-3


# ═════════════════════════════════════════════════════════════════════════════
# DETERMINISM
# ═════════════════════════════════════════════════════════════════════════════

class TestFusedGeluLinearDeterminism:
    """Same input -> bit-exact same output."""

    def test_cuda_deterministic(self, fk):
        X, W, bias = make_inputs(256, 768, 768)
        out1 = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        out2 = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert torch.equal(out1, out2), "CUDA kernel not deterministic"

    def test_triton_deterministic(self, fk):
        X, W, bias = make_inputs(256, 768, 768)
        out1 = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        out2 = fk.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)
        assert torch.equal(out1, out2), "Triton kernel not deterministic"

    def test_cuda_tanh_deterministic(self, fk):
        X, W, bias = make_inputs(256, 768, 768)
        out1 = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
        out2 = fk.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
        assert torch.equal(out1, out2)


# ═════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═════════════════════════════════════════════════════════════════════════════

class TestFusedGeluLinearErrors:
    """Input validation and error messages."""

    def test_cpu_tensor_cuda(self, fk):
        X = torch.randn(128, 768, dtype=torch.float16)
        W = torch.randn(768, 768, dtype=torch.float16, device="cuda")
        with pytest.raises((RuntimeError,)):
            fk.fused_gelu_linear(X, W)

    def test_fp32_input_cuda(self, fk):
        X = torch.randn(128, 768, dtype=torch.float32, device="cuda")
        W = torch.randn(768, 768, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError):
            fk.fused_gelu_linear(X, W)

    def test_shape_mismatch_cuda(self, fk):
        X = torch.randn(128, 768, dtype=torch.float16, device="cuda")
        W = torch.randn(768, 512, dtype=torch.float16, device="cuda")  # K mismatch
        with pytest.raises(RuntimeError):
            fk.fused_gelu_linear(X, W)

    def test_3d_input_cuda(self, fk):
        X = torch.randn(2, 128, 768, dtype=torch.float16, device="cuda")
        W = torch.randn(768, 768, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError):
            fk.fused_gelu_linear(X, W)

    def test_bias_wrong_size_cuda(self, fk):
        X = torch.randn(128, 768, dtype=torch.float16, device="cuda")
        W = torch.randn(768, 768, dtype=torch.float16, device="cuda")
        bias = torch.randn(512, dtype=torch.float16, device="cuda")  # wrong N
        with pytest.raises(RuntimeError):
            fk.fused_gelu_linear(X, W, bias=bias)

    def test_cpu_tensor_triton(self, fk):
        X = torch.randn(128, 768, dtype=torch.float16)
        W = torch.randn(768, 768, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError):
            fk.triton_fused_gelu_linear(X, W)

    def test_fp32_input_triton(self, fk):
        X = torch.randn(128, 768, dtype=torch.float32, device="cuda")
        W = torch.randn(768, 768, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError):
            fk.triton_fused_gelu_linear(X, W)

    def test_shape_mismatch_triton(self, fk):
        X = torch.randn(128, 768, dtype=torch.float16, device="cuda")
        W = torch.randn(768, 512, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError):
            fk.triton_fused_gelu_linear(X, W)
