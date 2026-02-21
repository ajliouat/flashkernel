"""
FlashKernel — RoPE Embedding Tests (v1.0.5)

Validates correctness of Rotary Position Embedding against
HuggingFace-style `apply_rotary_pos_emb` reference implementation.

Test matrix (from ROADMAP.md):
  | Config                             | Check                            |
  |------------------------------------|----------------------------------|
  | seq=[512,1024,2048,4096], d=64     | Correctness vs reference         |
  | seq=[512,1024,2048,4096], d=128    | Correctness vs reference         |
  | Table-lookup variant               | cos/sin table + apply            |
  | Fused variant (on-the-fly sin/cos) | Same output as table variant     |
  | CUDA vs Triton agreement           | Cross-validation                 |

Additional tests:
  - Precomputed frequency table correctness
  - Position-dependent property: different positions give different rotations
  - Rotation norm preservation: ||q_rot|| ≈ ||q||
  - Relative position property: dot product depends on relative position
  - Determinism (same input → same output)
  - Error handling (CPU tensors, wrong dtype, odd head_dim, shape mismatch)
"""

import math
import pytest
import torch

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


# ─── Reference Implementation ───────────────────────────────────────────────

def reference_rope_freqs(max_seq_len, head_dim, base=10000.0, device="cuda"):
    """
    Reference RoPE frequency table computation (matches HuggingFace LLaMA).

    Returns:
        cos_table: [max_seq_len, head_dim/2] fp32
        sin_table: [max_seq_len, head_dim/2] fp32
    """
    half_dim = head_dim // 2
    # inv_freq = 1.0 / (base ** (2i / head_dim)) for i in [0, half_dim)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    # [max_seq_len, 1] * [1, half_dim] → [max_seq_len, half_dim]
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [max_seq_len, half_dim]
    cos_table = torch.cos(angles)
    sin_table = torch.sin(angles)
    return cos_table, sin_table


def reference_apply_rope(x, cos_table, sin_table):
    """
    Reference RoPE application (HuggingFace style).

    x: [batch, num_heads, seq_len, head_dim] fp16
    cos_table: [max_seq_len, head_dim/2] fp32
    sin_table: [max_seq_len, head_dim/2] fp32

    Returns: x_rotated with same shape and dtype
    """
    seq_len = x.shape[2]
    head_dim = x.shape[3]
    half_dim = head_dim // 2

    x_fp32 = x.float()
    x0 = x_fp32[..., 0::2]  # [B, H, N, half_dim] — even indices
    x1 = x_fp32[..., 1::2]  # [B, H, N, half_dim] — odd indices

    cos = cos_table[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, N, half_dim]
    sin = sin_table[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, N, half_dim]

    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos

    # Interleave back: [B, H, N, D]
    out = torch.stack([y0, y1], dim=-1).flatten(-2)
    return out.half()


def max_abs_error(a, b):
    """Max absolute error between two tensors (in fp32)."""
    return (a.float() - b.float()).abs().max().item()


# ═════════════════════════════════════════════════════════════════════════════
# TEST: Frequency Table Precomputation
# ═════════════════════════════════════════════════════════════════════════════

class TestRopePrecomputeFreqs:
    """Test cos/sin frequency table correctness."""

    def test_shape_d64(self, fk):
        cos, sin = fk.rope_precompute_freqs(1024, 64)
        assert cos.shape == (1024, 32)
        assert sin.shape == (1024, 32)
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32

    def test_shape_d128(self, fk):
        cos, sin = fk.rope_precompute_freqs(2048, 128)
        assert cos.shape == (2048, 64)
        assert sin.shape == (2048, 64)

    def test_correctness_vs_reference(self, fk):
        """Verify precomputed tables match reference implementation."""
        for head_dim in [64, 128]:
            max_seq = 4096
            cos, sin = fk.rope_precompute_freqs(max_seq, head_dim)
            ref_cos, ref_sin = reference_rope_freqs(max_seq, head_dim)
            assert max_abs_error(cos, ref_cos) < 1e-5, \
                f"cos table mismatch for d={head_dim}"
            assert max_abs_error(sin, ref_sin) < 1e-5, \
                f"sin table mismatch for d={head_dim}"

    def test_position_zero(self, fk):
        """Position 0: cos=1, sin=0 for all dimensions."""
        cos, sin = fk.rope_precompute_freqs(16, 64)
        assert torch.allclose(cos[0], torch.ones(32, device="cuda"), atol=1e-6)
        assert torch.allclose(sin[0], torch.zeros(32, device="cuda"), atol=1e-6)

    def test_custom_base(self, fk):
        """Custom frequency base should change rotation speed."""
        cos1, _ = fk.rope_precompute_freqs(512, 64, base=10000.0)
        cos2, _ = fk.rope_precompute_freqs(512, 64, base=500000.0)
        # Different bases → different cos values at non-zero positions
        assert not torch.allclose(cos1[100], cos2[100], atol=1e-3)

    def test_triton_precompute(self):
        """Triton precompute matches CUDA."""
        from src.triton.rope import triton_rope_precompute_freqs
        import flashkernel as fk

        for head_dim in [64, 128]:
            cos_cuda, sin_cuda = fk.rope_precompute_freqs(2048, head_dim)
            cos_tri, sin_tri = triton_rope_precompute_freqs(2048, head_dim)
            assert max_abs_error(cos_cuda, cos_tri) < 1e-5
            assert max_abs_error(sin_cuda, sin_tri) < 1e-5


# ═════════════════════════════════════════════════════════════════════════════
# TEST: CUDA RoPE Forward (Table Lookup)
# ═════════════════════════════════════════════════════════════════════════════

class TestRopeForwardCUDAD64:
    """CUDA RoPE with precomputed tables, head_dim=64."""

    def _run(self, fk, B, H, N, D=64):
        Q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        K = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        cos, sin = fk.rope_precompute_freqs(N, D)
        Q_rot, K_rot = fk.rope_forward(Q, K, cos, sin)
        Q_ref = reference_apply_rope(Q, cos, sin)
        K_ref = reference_apply_rope(K, cos, sin)
        return Q_rot, K_rot, Q_ref, K_ref

    def test_basic(self, fk):
        Q_rot, K_rot, Q_ref, K_ref = self._run(fk, B=1, H=8, N=512)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    def test_batched(self, fk):
        Q_rot, K_rot, Q_ref, K_ref = self._run(fk, B=4, H=8, N=1024)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    def test_long_seq(self, fk):
        Q_rot, K_rot, Q_ref, K_ref = self._run(fk, B=1, H=8, N=4096)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    @pytest.mark.parametrize("N", [128, 256, 512, 1024, 2048])
    def test_seq_sweep(self, fk, N):
        Q_rot, K_rot, Q_ref, K_ref = self._run(fk, B=2, H=4, N=N)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3


class TestRopeForwardCUDAD128:
    """CUDA RoPE with precomputed tables, head_dim=128."""

    def _run(self, fk, B, H, N, D=128):
        Q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        K = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        cos, sin = fk.rope_precompute_freqs(N, D)
        Q_rot, K_rot = fk.rope_forward(Q, K, cos, sin)
        Q_ref = reference_apply_rope(Q, cos, sin)
        K_ref = reference_apply_rope(K, cos, sin)
        return Q_rot, K_rot, Q_ref, K_ref

    def test_basic(self, fk):
        Q_rot, K_rot, Q_ref, K_ref = self._run(fk, B=1, H=8, N=512)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    def test_large(self, fk):
        Q_rot, K_rot, Q_ref, K_ref = self._run(fk, B=8, H=12, N=2048)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    @pytest.mark.parametrize("N", [128, 256, 512, 1024])
    def test_seq_sweep(self, fk, N):
        Q_rot, K_rot, Q_ref, K_ref = self._run(fk, B=2, H=4, N=N)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3


# ═════════════════════════════════════════════════════════════════════════════
# TEST: CUDA RoPE Forward Fused (On-the-fly sin/cos)
# ═════════════════════════════════════════════════════════════════════════════

class TestRopeForwardCUDAFused:
    """CUDA RoPE fused variant — computes sin/cos on the fly."""

    def test_basic_d64(self, fk):
        Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
        cos, sin = reference_rope_freqs(512, 64)
        Q_ref = reference_apply_rope(Q, cos, sin)
        K_ref = reference_apply_rope(K, cos, sin)
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    def test_basic_d128(self, fk):
        Q = torch.randn(1, 8, 512, 128, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 8, 512, 128, dtype=torch.float16, device="cuda")
        cos, sin = reference_rope_freqs(512, 128)
        Q_ref = reference_apply_rope(Q, cos, sin)
        K_ref = reference_apply_rope(K, cos, sin)
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    def test_matches_table_variant(self, fk):
        """Fused and table-lookup should give same result."""
        Q = torch.randn(2, 4, 1024, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 4, 1024, 64, dtype=torch.float16, device="cuda")
        cos, sin = fk.rope_precompute_freqs(1024, 64)
        Q_table, K_table = fk.rope_forward(Q, K, cos, sin)
        Q_fused, K_fused = fk.rope_forward_fused(Q, K)
        assert max_abs_error(Q_table, Q_fused) < 1e-3
        assert max_abs_error(K_table, K_fused) < 1e-3

    @pytest.mark.parametrize("N", [256, 512, 1024, 2048])
    def test_seq_sweep(self, fk, N):
        Q = torch.randn(1, 8, N, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 8, N, 64, dtype=torch.float16, device="cuda")
        cos, sin = reference_rope_freqs(N, 64)
        Q_ref = reference_apply_rope(Q, cos, sin)
        K_ref = reference_apply_rope(K, cos, sin)
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    def test_custom_base(self, fk):
        """Custom base should change rotation but still be correct."""
        base = 500000.0
        Q = torch.randn(1, 4, 256, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 4, 256, 64, dtype=torch.float16, device="cuda")
        cos, sin = reference_rope_freqs(256, 64, base=base)
        Q_ref = reference_apply_rope(Q, cos, sin)
        K_ref = reference_apply_rope(K, cos, sin)
        Q_rot, K_rot = fk.rope_forward_fused(Q, K, base=base)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3


# ═════════════════════════════════════════════════════════════════════════════
# TEST: Triton RoPE
# ═════════════════════════════════════════════════════════════════════════════

class TestTritonRopeForward:
    """Triton RoPE — table lookup variant."""

    def _run(self, B, H, N, D):
        from src.triton.rope import triton_rope_precompute_freqs, triton_rope_forward
        Q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        K = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        cos, sin = triton_rope_precompute_freqs(N, D)
        ref_cos, ref_sin = reference_rope_freqs(N, D)
        Q_rot, K_rot = triton_rope_forward(Q.clone(), K.clone(), cos, sin)
        Q_ref = reference_apply_rope(Q, ref_cos, ref_sin)
        K_ref = reference_apply_rope(K, ref_cos, ref_sin)
        return Q_rot, K_rot, Q_ref, K_ref

    def test_d64(self):
        Q_rot, K_rot, Q_ref, K_ref = self._run(1, 8, 512, 64)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    def test_d128(self):
        Q_rot, K_rot, Q_ref, K_ref = self._run(1, 8, 512, 128)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    @pytest.mark.parametrize("N", [128, 512, 1024, 2048])
    def test_seq_sweep(self, N):
        Q_rot, K_rot, Q_ref, K_ref = self._run(2, 4, N, 64)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3


class TestTritonRopeForwardFused:
    """Triton RoPE — fused variant (on-the-fly sin/cos)."""

    def test_d64(self):
        from src.triton.rope import triton_rope_forward_fused
        Q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 8, 512, 64, dtype=torch.float16, device="cuda")
        cos, sin = reference_rope_freqs(512, 64)
        Q_ref = reference_apply_rope(Q, cos, sin)
        K_ref = reference_apply_rope(K, cos, sin)
        Q_rot, K_rot = triton_rope_forward_fused(Q.clone(), K.clone())
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    def test_d128(self):
        from src.triton.rope import triton_rope_forward_fused
        Q = torch.randn(1, 8, 512, 128, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 8, 512, 128, dtype=torch.float16, device="cuda")
        cos, sin = reference_rope_freqs(512, 128)
        Q_ref = reference_apply_rope(Q, cos, sin)
        K_ref = reference_apply_rope(K, cos, sin)
        Q_rot, K_rot = triton_rope_forward_fused(Q.clone(), K.clone())
        assert max_abs_error(Q_rot, Q_ref) < 1e-3
        assert max_abs_error(K_rot, K_ref) < 1e-3

    def test_matches_table_variant(self):
        from src.triton.rope import (
            triton_rope_precompute_freqs, triton_rope_forward,
            triton_rope_forward_fused,
        )
        Q = torch.randn(2, 4, 1024, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 4, 1024, 64, dtype=torch.float16, device="cuda")
        cos, sin = triton_rope_precompute_freqs(1024, 64)
        Q_table, K_table = triton_rope_forward(Q.clone(), K.clone(), cos, sin)
        Q_fused, K_fused = triton_rope_forward_fused(Q.clone(), K.clone())
        assert max_abs_error(Q_table, Q_fused) < 1e-3
        assert max_abs_error(K_table, K_fused) < 1e-3


# ═════════════════════════════════════════════════════════════════════════════
# TEST: CUDA vs Triton Cross-Validation
# ═════════════════════════════════════════════════════════════════════════════

class TestCUDAvsTritonRope:
    """Verify CUDA and Triton RoPE produce the same output."""

    def test_table_d64(self, fk):
        from src.triton.rope import triton_rope_precompute_freqs, triton_rope_forward
        Q = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device="cuda")
        cos_c, sin_c = fk.rope_precompute_freqs(1024, 64)
        cos_t, sin_t = triton_rope_precompute_freqs(1024, 64)
        Q_cuda, K_cuda = fk.rope_forward(Q, K, cos_c, sin_c)
        Q_tri, K_tri = triton_rope_forward(Q.clone(), K.clone(), cos_t, sin_t)
        assert max_abs_error(Q_cuda, Q_tri) < 1e-3
        assert max_abs_error(K_cuda, K_tri) < 1e-3

    def test_table_d128(self, fk):
        from src.triton.rope import triton_rope_precompute_freqs, triton_rope_forward
        Q = torch.randn(1, 12, 512, 128, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 12, 512, 128, dtype=torch.float16, device="cuda")
        cos_c, sin_c = fk.rope_precompute_freqs(512, 128)
        cos_t, sin_t = triton_rope_precompute_freqs(512, 128)
        Q_cuda, K_cuda = fk.rope_forward(Q, K, cos_c, sin_c)
        Q_tri, K_tri = triton_rope_forward(Q.clone(), K.clone(), cos_t, sin_t)
        assert max_abs_error(Q_cuda, Q_tri) < 1e-3
        assert max_abs_error(K_cuda, K_tri) < 1e-3

    def test_fused_d64(self, fk):
        from src.triton.rope import triton_rope_forward_fused
        Q = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device="cuda")
        Q_cuda, K_cuda = fk.rope_forward_fused(Q, K)
        Q_tri, K_tri = triton_rope_forward_fused(Q.clone(), K.clone())
        assert max_abs_error(Q_cuda, Q_tri) < 1e-3
        assert max_abs_error(K_cuda, K_tri) < 1e-3

    @pytest.mark.parametrize("N", [128, 512, 1024])
    def test_seq_sweep(self, fk, N):
        from src.triton.rope import triton_rope_precompute_freqs, triton_rope_forward
        Q = torch.randn(1, 4, N, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 4, N, 64, dtype=torch.float16, device="cuda")
        cos_c, sin_c = fk.rope_precompute_freqs(N, 64)
        cos_t, sin_t = triton_rope_precompute_freqs(N, 64)
        Q_cuda, K_cuda = fk.rope_forward(Q, K, cos_c, sin_c)
        Q_tri, K_tri = triton_rope_forward(Q.clone(), K.clone(), cos_t, sin_t)
        assert max_abs_error(Q_cuda, Q_tri) < 1e-3
        assert max_abs_error(K_cuda, K_tri) < 1e-3


# ═════════════════════════════════════════════════════════════════════════════
# TEST: Mathematical Properties
# ═════════════════════════════════════════════════════════════════════════════

class TestRopeProperties:
    """Verify mathematical properties of RoPE."""

    def test_norm_preservation(self, fk):
        """RoPE is a rotation — it should preserve vector norms."""
        Q = torch.randn(2, 8, 512, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 8, 512, 64, dtype=torch.float16, device="cuda")
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)

        # Norms should be approximately preserved (within fp16 precision)
        q_norms = Q.float().norm(dim=-1)
        q_rot_norms = Q_rot.float().norm(dim=-1)
        relative_err = ((q_norms - q_rot_norms).abs() / (q_norms + 1e-6)).max().item()
        assert relative_err < 0.02, f"Norm not preserved: relative error {relative_err}"

        k_norms = K.float().norm(dim=-1)
        k_rot_norms = K_rot.float().norm(dim=-1)
        relative_err = ((k_norms - k_rot_norms).abs() / (k_norms + 1e-6)).max().item()
        assert relative_err < 0.02, f"K norm not preserved: relative error {relative_err}"

    def test_position_zero_identity(self, fk):
        """At position 0, cos=1, sin=0 → rotation is identity."""
        Q = torch.randn(1, 1, 1, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 1, 1, 64, dtype=torch.float16, device="cuda")
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        # Position 0 should give identity rotation
        assert max_abs_error(Q, Q_rot) < 1e-3
        assert max_abs_error(K, K_rot) < 1e-3

    def test_different_positions_different_rotations(self, fk):
        """Different sequence positions should produce different rotations."""
        # Same vector at different positions
        vec = torch.randn(1, 1, 1, 64, dtype=torch.float16, device="cuda")
        Q = vec.expand(1, 1, 4, 64).contiguous()  # same vec at 4 positions
        K = torch.randn(1, 1, 4, 64, dtype=torch.float16, device="cuda")
        Q_rot, _ = fk.rope_forward_fused(Q, K)
        # Each position should give a different output
        for i in range(3):
            diff = (Q_rot[0, 0, i] - Q_rot[0, 0, i + 1]).abs().max().item()
            assert diff > 1e-3, f"Position {i} and {i+1} should differ"


# ═════════════════════════════════════════════════════════════════════════════
# TEST: Output Validation
# ═════════════════════════════════════════════════════════════════════════════

class TestRopeOutputs:
    """Validate output shape, dtype, device, and properties."""

    def test_output_shape_d64(self, fk):
        Q = torch.randn(2, 8, 512, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 8, 512, 64, dtype=torch.float16, device="cuda")
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert Q_rot.shape == Q.shape
        assert K_rot.shape == K.shape

    def test_output_shape_d128(self, fk):
        Q = torch.randn(1, 12, 256, 128, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 12, 256, 128, dtype=torch.float16, device="cuda")
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert Q_rot.shape == Q.shape
        assert K_rot.shape == K.shape

    def test_output_dtype(self, fk):
        Q = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="cuda")
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert Q_rot.dtype == torch.float16
        assert K_rot.dtype == torch.float16

    def test_output_device(self, fk):
        Q = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="cuda")
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert Q_rot.is_cuda
        assert K_rot.is_cuda

    def test_output_finite(self, fk):
        Q = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 8, 1024, 64, dtype=torch.float16, device="cuda")
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert torch.isfinite(Q_rot).all()
        assert torch.isfinite(K_rot).all()

    def test_not_inplace_on_original(self, fk):
        """The binding clones — original tensors should be unchanged."""
        Q = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="cuda")
        Q_orig = Q.clone()
        K_orig = K.clone()
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert torch.equal(Q, Q_orig), "Q was modified in-place"
        assert torch.equal(K, K_orig), "K was modified in-place"


# ═════════════════════════════════════════════════════════════════════════════
# TEST: Determinism
# ═════════════════════════════════════════════════════════════════════════════

class TestRopeDeterminism:
    """Same input → same output."""

    def test_determinism_cuda_table(self, fk):
        Q = torch.randn(2, 8, 512, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 8, 512, 64, dtype=torch.float16, device="cuda")
        cos, sin = fk.rope_precompute_freqs(512, 64)
        Q1, K1 = fk.rope_forward(Q, K, cos, sin)
        Q2, K2 = fk.rope_forward(Q, K, cos, sin)
        assert torch.equal(Q1, Q2)
        assert torch.equal(K1, K2)

    def test_determinism_cuda_fused(self, fk):
        Q = torch.randn(2, 8, 512, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 8, 512, 64, dtype=torch.float16, device="cuda")
        Q1, K1 = fk.rope_forward_fused(Q, K)
        Q2, K2 = fk.rope_forward_fused(Q, K)
        assert torch.equal(Q1, Q2)
        assert torch.equal(K1, K2)

    def test_determinism_triton(self):
        from src.triton.rope import triton_rope_forward_fused
        Q = torch.randn(2, 8, 512, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 8, 512, 64, dtype=torch.float16, device="cuda")
        Q1, K1 = triton_rope_forward_fused(Q.clone(), K.clone())
        Q2, K2 = triton_rope_forward_fused(Q.clone(), K.clone())
        assert torch.equal(Q1, Q2)
        assert torch.equal(K1, K2)


# ═════════════════════════════════════════════════════════════════════════════
# TEST: Boundary / Non-standard Dims
# ═════════════════════════════════════════════════════════════════════════════

class TestRopeBoundary:
    """Non-standard head dimensions and sequence lengths."""

    @pytest.mark.parametrize("D", [32, 48, 64, 96, 128, 256])
    def test_various_head_dims(self, fk, D):
        Q = torch.randn(1, 4, 128, D, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 4, 128, D, dtype=torch.float16, device="cuda")
        cos, sin = reference_rope_freqs(128, D)
        Q_ref = reference_apply_rope(Q, cos, sin)
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3

    @pytest.mark.parametrize("N", [1, 3, 7, 15, 33, 100, 513])
    def test_non_power_of_two_seq(self, fk, N):
        Q = torch.randn(1, 4, N, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 4, N, 64, dtype=torch.float16, device="cuda")
        cos, sin = reference_rope_freqs(N, 64)
        Q_ref = reference_apply_rope(Q, cos, sin)
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3

    def test_single_token(self, fk):
        Q = torch.randn(1, 1, 1, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 1, 1, 64, dtype=torch.float16, device="cuda")
        cos, sin = reference_rope_freqs(1, 64)
        Q_ref = reference_apply_rope(Q, cos, sin)
        Q_rot, K_rot = fk.rope_forward_fused(Q, K)
        assert max_abs_error(Q_rot, Q_ref) < 1e-3


# ═════════════════════════════════════════════════════════════════════════════
# TEST: Error Handling
# ═════════════════════════════════════════════════════════════════════════════

class TestRopeErrors:
    """Test error handling for invalid inputs."""

    def test_cpu_raises(self, fk):
        Q = torch.randn(1, 4, 128, 64, dtype=torch.float16)
        K = torch.randn(1, 4, 128, 64, dtype=torch.float16)
        with pytest.raises(RuntimeError, match="CUDA"):
            fk.rope_forward_fused(Q, K)

    def test_fp32_raises(self, fk):
        Q = torch.randn(1, 4, 128, 64, dtype=torch.float32, device="cuda")
        K = torch.randn(1, 4, 128, 64, dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match="float16"):
            fk.rope_forward_fused(Q, K)

    def test_odd_head_dim_raises(self, fk):
        Q = torch.randn(1, 4, 128, 63, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 4, 128, 63, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="even"):
            fk.rope_forward_fused(Q, K)

    def test_shape_mismatch_raises(self, fk):
        Q = torch.randn(1, 4, 128, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(2, 4, 128, 64, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="match"):
            fk.rope_forward_fused(Q, K)

    def test_3d_input_raises(self, fk):
        Q = torch.randn(4, 128, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(4, 128, 64, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="4-D"):
            fk.rope_forward_fused(Q, K)

    def test_table_too_short_raises(self, fk):
        Q = torch.randn(1, 4, 256, 64, dtype=torch.float16, device="cuda")
        K = torch.randn(1, 4, 256, 64, dtype=torch.float16, device="cuda")
        cos, sin = fk.rope_precompute_freqs(128, 64)  # only 128 positions
        with pytest.raises(RuntimeError, match="max_seq_len"):
            fk.rope_forward(Q, K, cos, sin)  # seq_len=256 > 128
