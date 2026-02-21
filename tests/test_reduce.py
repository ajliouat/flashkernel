"""
FlashKernel — Parallel reduction tests (v1.0.1)

Tests:
  1. reduce_sum correctness (fp32) — full reduction, match torch.sum
  2. reduce_sum correctness (fp16) — with tolerance
  3. reduce_max correctness (fp32/fp16)
  4. reduce_sum_rows — per-row reduction, multiple shapes
  5. reduce_sum with dim argument — arbitrary dim reduction
  6. Triton variants — same tests against Triton implementations
  7. Edge cases — single element, large tensors

Test matrix from ROADMAP.md:
  | Shape              | dtype | Expected                    |
  |--------------------|-------|-----------------------------|
  | (1024,)            | fp32  | Match torch.sum             |
  | (4096,)            | fp32  | Match torch.sum             |
  | (1, 128, 4096)     | fp16  | Match torch.sum (atol=1e-2) |
  | (8, 64, 2048)      | fp16  | Match torch.sum             |
"""

import pytest
import torch
import sys
import os

# Skip entire module if no CUDA GPU available
pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ─── Import ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fk():
    """Import flashkernel C extension."""
    import flashkernel
    return flashkernel


@pytest.fixture(scope="module")
def triton_reduce():
    """Import Triton reduction kernels."""
    # Add project root to path so src.triton.reduce import works
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.triton.reduce import (
        triton_reduce_sum,
        triton_reduce_max,
        triton_reduce_sum_rows,
    )
    return {
        "sum": triton_reduce_sum,
        "max": triton_reduce_max,
        "sum_rows": triton_reduce_sum_rows,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CUDA REDUCE SUM — Full Reduction
# ═════════════════════════════════════════════════════════════════════════════

class TestReduceSumF32:
    """fp32 full sum reduction — should match torch.sum closely."""

    @pytest.mark.parametrize("n", [1, 32, 1024, 4096, 65536, 1_000_000])
    def test_shapes(self, fk, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x)
        expected = x.sum()
        # fp32 reduction can have small accumulation errors for large n
        atol = 1e-3 if n > 100_000 else 1e-5
        assert torch.allclose(result, expected, atol=atol, rtol=1e-4), \
            f"n={n}: got {result.item()}, expected {expected.item()}, " \
            f"diff={abs(result.item() - expected.item())}"

    def test_zeros(self, fk):
        x = torch.zeros(4096, device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x)
        assert result.item() == 0.0

    def test_ones(self, fk):
        n = 1024
        x = torch.ones(n, device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x)
        assert torch.allclose(result, torch.tensor(float(n), device="cuda"))

    def test_negative(self, fk):
        x = -torch.ones(1024, device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x)
        assert torch.allclose(result, torch.tensor(-1024.0, device="cuda"))


class TestReduceSumF16:
    """fp16 full sum reduction — wider tolerance for half precision."""

    @pytest.mark.parametrize("n", [1, 32, 1024, 4096, 65536])
    def test_shapes(self, fk, n):
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        result = fk.reduce_sum(x)
        expected = x.float().sum().half()  # accumulate in fp32 then back to fp16
        # fp16 tolerance: relative can be loose for large sums
        atol = max(1e-1, abs(expected.item()) * 1e-2)
        assert torch.allclose(result.float(), expected.float(), atol=atol), \
            f"n={n}: got {result.item()}, expected {expected.item()}"


class TestReduceSumRoadmapShapes:
    """Exact shapes from the ROADMAP.md test matrix."""

    def test_1024_fp32(self, fk):
        x = torch.randn(1024, device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x)
        expected = x.sum()
        assert torch.allclose(result, expected, atol=1e-5)

    def test_4096_fp32(self, fk):
        x = torch.randn(4096, device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x)
        expected = x.sum()
        assert torch.allclose(result, expected, atol=1e-4)

    def test_1_128_4096_fp16(self, fk):
        x = torch.randn(1, 128, 4096, device="cuda", dtype=torch.float16)
        result = fk.reduce_sum(x)
        expected = x.float().sum().half()
        atol = max(1.0, abs(expected.item()) * 0.01)
        assert torch.allclose(result.float(), expected.float(), atol=atol)

    def test_8_64_2048_fp16(self, fk):
        x = torch.randn(8, 64, 2048, device="cuda", dtype=torch.float16)
        result = fk.reduce_sum(x)
        expected = x.float().sum().half()
        atol = max(1.0, abs(expected.item()) * 0.01)
        assert torch.allclose(result.float(), expected.float(), atol=atol)


# ═════════════════════════════════════════════════════════════════════════════
# CUDA REDUCE MAX
# ═════════════════════════════════════════════════════════════════════════════

class TestReduceMaxF32:
    """fp32 max reduction."""

    @pytest.mark.parametrize("n", [1, 32, 1024, 4096, 65536, 1_000_000])
    def test_shapes(self, fk, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        result = fk.reduce_max(x)
        expected = x.max()
        assert torch.allclose(result, expected), \
            f"n={n}: got {result.item()}, expected {expected.item()}"

    def test_all_negative(self, fk):
        x = -torch.abs(torch.randn(1024, device="cuda", dtype=torch.float32)) - 1.0
        result = fk.reduce_max(x)
        expected = x.max()
        assert torch.allclose(result, expected)

    def test_known_max(self, fk):
        x = torch.zeros(1024, device="cuda", dtype=torch.float32)
        x[512] = 42.0
        result = fk.reduce_max(x)
        assert result.item() == 42.0


class TestReduceMaxF16:
    """fp16 max reduction."""

    @pytest.mark.parametrize("n", [1, 32, 1024, 4096, 65536])
    def test_shapes(self, fk, n):
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        result = fk.reduce_max(x)
        expected = x.max()
        assert torch.allclose(result.float(), expected.float(), atol=1e-3), \
            f"n={n}: got {result.item()}, expected {expected.item()}"


# ═════════════════════════════════════════════════════════════════════════════
# CUDA ROW-WISE REDUCTION (dim argument)
# ═════════════════════════════════════════════════════════════════════════════

class TestReduceSumRows:
    """Per-row sum reduction using the dim argument."""

    def test_2d_last_dim(self, fk):
        x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x, dim=1)
        expected = x.sum(dim=1)
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-4)

    def test_2d_first_dim(self, fk):
        x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x, dim=0)
        expected = x.sum(dim=0)
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-4)

    def test_3d_last_dim(self, fk):
        x = torch.randn(8, 64, 256, device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x, dim=2)
        expected = x.sum(dim=2)
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-3)

    def test_3d_last_dim_fp16(self, fk):
        x = torch.randn(8, 64, 2048, device="cuda", dtype=torch.float16)
        result = fk.reduce_sum(x, dim=2)
        expected = x.float().sum(dim=2).half()
        assert result.shape == expected.shape
        atol = 0.5  # fp16 row sums over 2048 elements
        assert torch.allclose(result.float(), expected.float(), atol=atol)

    def test_single_row(self, fk):
        x = torch.randn(1, 1024, device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x, dim=1)
        expected = x.sum(dim=1)
        assert torch.allclose(result, expected, atol=1e-5)


# ═════════════════════════════════════════════════════════════════════════════
# CUDA REDUCE EDGE CASES
# ═════════════════════════════════════════════════════════════════════════════

class TestReduceEdgeCases:
    """Error handling and edge cases."""

    def test_cpu_tensor_raises(self, fk):
        x = torch.randn(100)  # CPU
        with pytest.raises(RuntimeError):
            fk.reduce_sum(x)

    def test_non_contiguous_raises(self, fk):
        x = torch.randn(100, 2, device="cuda")[:, 0]
        assert not x.is_contiguous()
        with pytest.raises(RuntimeError):
            fk.reduce_sum(x)

    def test_single_element_sum(self, fk):
        x = torch.tensor([3.14], device="cuda", dtype=torch.float32)
        result = fk.reduce_sum(x)
        assert torch.allclose(result, x.squeeze())

    def test_single_element_max(self, fk):
        x = torch.tensor([2.71], device="cuda", dtype=torch.float32)
        result = fk.reduce_max(x)
        assert torch.allclose(result, x.squeeze())


# ═════════════════════════════════════════════════════════════════════════════
# TRITON EQUIVALENTS
# ═════════════════════════════════════════════════════════════════════════════

class TestTritonReduceSum:
    """Triton sum reduction — verify it matches torch.sum."""

    @pytest.mark.parametrize("n", [32, 1024, 4096, 65536, 1_000_000])
    def test_full_reduce_f32(self, triton_reduce, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        result = triton_reduce["sum"](x)
        expected = x.sum()
        atol = 1e-3 if n > 100_000 else 1e-4
        assert torch.allclose(result, expected, atol=atol, rtol=1e-3), \
            f"n={n}: got {result.item()}, expected {expected.item()}"

    @pytest.mark.parametrize("n", [32, 1024, 4096])
    def test_full_reduce_f16(self, triton_reduce, n):
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        result = triton_reduce["sum"](x)
        expected = x.float().sum().half()
        atol = max(0.5, abs(expected.item()) * 0.01)
        assert torch.allclose(result.float(), expected.float(), atol=atol)


class TestTritonReduceMax:
    """Triton max reduction."""

    @pytest.mark.parametrize("n", [32, 1024, 4096, 65536])
    def test_full_reduce_f32(self, triton_reduce, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        result = triton_reduce["max"](x)
        expected = x.max()
        assert torch.allclose(result, expected)


class TestTritonReduceSumRows:
    """Triton row-wise reduction."""

    def test_2d(self, triton_reduce):
        x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        result = triton_reduce["sum_rows"](x, dim=-1)
        expected = x.sum(dim=-1)
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-4)

    def test_3d(self, triton_reduce):
        x = torch.randn(8, 64, 256, device="cuda", dtype=torch.float32)
        result = triton_reduce["sum_rows"](x, dim=-1)
        expected = x.sum(dim=-1)
        assert result.shape == expected.shape
        assert torch.allclose(result, expected, atol=1e-3)


# ═════════════════════════════════════════════════════════════════════════════
# CUDA vs TRITON CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

class TestCrossValidation:
    """Ensure CUDA and Triton implementations agree."""

    @pytest.mark.parametrize("n", [1024, 65536])
    def test_sum_cuda_vs_triton(self, fk, triton_reduce, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        cuda_result = fk.reduce_sum(x)
        triton_result = triton_reduce["sum"](x)
        assert torch.allclose(cuda_result, triton_result, atol=1e-4), \
            f"CUDA={cuda_result.item()}, Triton={triton_result.item()}"

    @pytest.mark.parametrize("n", [1024, 65536])
    def test_max_cuda_vs_triton(self, fk, triton_reduce, n):
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        cuda_result = fk.reduce_max(x)
        triton_result = triton_reduce["max"](x)
        assert torch.allclose(cuda_result, triton_result), \
            f"CUDA={cuda_result.item()}, Triton={triton_result.item()}"
