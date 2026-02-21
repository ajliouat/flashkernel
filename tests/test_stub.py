"""
FlashKernel — Stub kernel tests (v1.0.0)

Tests:
  1. vector_add correctness (fp32) — exact match with torch.add
  2. vector_add correctness (fp16) — close match (half precision)
  3. device_info returns expected keys
  4. Edge cases: empty tensor, large tensor, non-contiguous input
"""

import pytest
import torch

# Skip entire module if no CUDA GPU available
pytestmark = pytest.mark.cuda

def have_cuda():
    return torch.cuda.is_available()

@pytest.fixture(autouse=True)
def skip_without_cuda():
    if not have_cuda():
        pytest.skip("CUDA not available")


# ─── Import ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fk():
    """Import flashkernel C extension."""
    import flashkernel
    return flashkernel


# ─── Correctness Tests ──────────────────────────────────────────────────────

class TestVectorAddF32:
    """fp32 vector addition — should match PyTorch exactly (no rounding)."""

    @pytest.mark.parametrize("n", [1, 32, 1024, 65536, 1_000_000])
    def test_shapes(self, fk, n):
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        c = fk.vector_add(a, b)
        expected = a + b
        assert torch.allclose(c, expected, atol=0, rtol=0), \
            f"Max error: {(c - expected).abs().max().item()}"

    def test_zeros(self, fk):
        a = torch.zeros(1024, device="cuda", dtype=torch.float32)
        b = torch.zeros(1024, device="cuda", dtype=torch.float32)
        c = fk.vector_add(a, b)
        assert (c == 0).all()

    def test_negative(self, fk):
        a = torch.ones(1024, device="cuda", dtype=torch.float32)
        b = -torch.ones(1024, device="cuda", dtype=torch.float32)
        c = fk.vector_add(a, b)
        assert torch.allclose(c, torch.zeros_like(c))

    def test_large_values(self, fk):
        a = torch.full((1024,), 1e30, device="cuda", dtype=torch.float32)
        b = torch.full((1024,), 1e30, device="cuda", dtype=torch.float32)
        c = fk.vector_add(a, b)
        expected = a + b
        assert torch.allclose(c, expected)


class TestVectorAddF16:
    """fp16 vector addition — allow tolerance for half-precision rounding."""

    @pytest.mark.parametrize("n", [1, 32, 1024, 65536, 1_000_000])
    def test_shapes(self, fk, n):
        a = torch.randn(n, device="cuda", dtype=torch.float16)
        b = torch.randn(n, device="cuda", dtype=torch.float16)
        c = fk.vector_add(a, b)
        expected = a + b
        # fp16 has ~3 decimal digits of precision
        assert torch.allclose(c, expected, atol=1e-3, rtol=1e-3), \
            f"Max error: {(c - expected).abs().max().item()}"

    def test_small_values(self, fk):
        """fp16 subnormals — values near zero."""
        a = torch.full((1024,), 1e-4, device="cuda", dtype=torch.float16)
        b = torch.full((1024,), 1e-4, device="cuda", dtype=torch.float16)
        c = fk.vector_add(a, b)
        expected = a + b
        assert torch.allclose(c, expected, atol=1e-3)


class TestVectorAddMultiDim:
    """Multi-dimensional tensors — internally contiguous, so should work."""

    def test_2d(self, fk):
        a = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        b = torch.randn(64, 128, device="cuda", dtype=torch.float32)
        c = fk.vector_add(a, b)
        expected = a + b
        assert torch.allclose(c, expected)

    def test_3d(self, fk):
        a = torch.randn(8, 64, 256, device="cuda", dtype=torch.float32)
        b = torch.randn(8, 64, 256, device="cuda", dtype=torch.float32)
        c = fk.vector_add(a, b)
        expected = a + b
        assert torch.allclose(c, expected)


class TestVectorAddEdgeCases:
    """Error handling and edge cases."""

    def test_mismatched_shapes_raises(self, fk):
        a = torch.randn(100, device="cuda")
        b = torch.randn(200, device="cuda")
        with pytest.raises(RuntimeError):
            fk.vector_add(a, b)

    def test_cpu_tensor_raises(self, fk):
        a = torch.randn(100)  # CPU
        b = torch.randn(100)  # CPU
        with pytest.raises(RuntimeError):
            fk.vector_add(a, b)

    def test_non_contiguous_raises(self, fk):
        a = torch.randn(100, 2, device="cuda")[:, 0]  # Non-contiguous
        b = torch.randn(100, device="cuda")
        assert not a.is_contiguous()
        with pytest.raises(RuntimeError):
            fk.vector_add(a, b)


# ─── Device Info ─────────────────────────────────────────────────────────────

class TestDeviceInfo:
    """Verify device_info returns the right structure."""

    def test_returns_dict(self, fk):
        info = fk.device_info()
        assert isinstance(info, dict)

    def test_has_expected_keys(self, fk):
        info = fk.device_info()
        expected_keys = [
            "name", "compute_capability", "sm_count",
            "global_mem_mb", "shared_mem_per_block_kb",
            "max_threads_per_block", "warp_size",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_reasonable_values(self, fk):
        info = fk.device_info()
        assert info["warp_size"] == 32
        assert info["max_threads_per_block"] >= 256
        assert info["global_mem_mb"] > 0
        assert info["sm_count"] > 0

    def test_name_not_empty(self, fk):
        info = fk.device_info()
        assert len(info["name"]) > 0
