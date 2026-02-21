"""
FlashKernel — Triton Rotary Position Embedding (v1.0.5)

Applies RoPE (Su et al., 2021) to Q and K tensors.

For each position pos and dimension pair (2i, 2i+1):
  q_rot[2i]   = q[2i]   * cos(θ) - q[2i+1] * sin(θ)
  q_rot[2i+1] = q[2i]   * sin(θ) + q[2i+1] * cos(θ)

where θ_i = pos / base^(2i/d), base defaults to 10000.

Two variants:
  1. Table-lookup: precomputed cos/sin, loaded from HBM
  2. Fused: compute cos/sin on the fly (saves bandwidth, slightly more math)

Architecture target: any GPU with Triton support (SM >= 7.0)
"""

import math

import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 1: Apply RoPE with precomputed cos/sin tables
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def _rope_forward_kernel(
    QK_ptr,         # [batch, num_heads, seq_len, head_dim] fp16
    cos_ptr,        # [max_seq_len, half_dim] fp32
    sin_ptr,        # [max_seq_len, half_dim] fp32
    batch,
    num_heads,
    seq_len,
    head_dim: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply RoPE to one tensor using precomputed cos/sin tables."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    total = batch * num_heads * seq_len * HALF_DIM
    mask = offs < total

    # Decompose flat index → (b, h, pos, i)
    i   = offs % HALF_DIM
    rem = offs // HALF_DIM
    pos = rem % seq_len
    rem = rem // seq_len
    # h = rem % num_heads (not needed for table lookup)
    # b = rem // num_heads (not needed for table lookup)

    # Pointer to the head vector element pairs
    # QK layout: [b, h, pos, d] → flat offset = rem * seq_len * head_dim + pos * head_dim
    head_offset = rem * seq_len * head_dim + pos * head_dim

    # Load pair (x_2i, x_{2i+1}) → fp32 for precision
    x0 = tl.load(QK_ptr + head_offset + 2 * i, mask=mask).to(tl.float32)
    x1 = tl.load(QK_ptr + head_offset + 2 * i + 1, mask=mask).to(tl.float32)

    # Load cos/sin from precomputed tables
    table_idx = pos * HALF_DIM + i
    c = tl.load(cos_ptr + table_idx, mask=mask)
    s = tl.load(sin_ptr + table_idx, mask=mask)

    # Apply rotation
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c

    # Store back in fp16
    tl.store(QK_ptr + head_offset + 2 * i, y0.to(tl.float16), mask=mask)
    tl.store(QK_ptr + head_offset + 2 * i + 1, y1.to(tl.float16), mask=mask)


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 2: Fused RoPE — compute cos/sin on the fly
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def _rope_forward_fused_kernel(
    QK_ptr,         # [batch, num_heads, seq_len, head_dim] fp16
    batch,
    num_heads,
    seq_len,
    head_dim: tl.constexpr,
    HALF_DIM: tl.constexpr,
    base: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply RoPE with on-the-fly sin/cos computation."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    total = batch * num_heads * seq_len * HALF_DIM
    mask = offs < total

    # Decompose flat index → (b, h, pos, i)
    i   = offs % HALF_DIM
    rem = offs // HALF_DIM
    pos = rem % seq_len
    rem = rem // seq_len

    # Compute freq and angle in-register
    # freq = 1.0 / base^(2i / head_dim)
    # Using exp/log for better numerical stability than pow:
    #   freq = exp(-2i/head_dim * log(base))
    exponent = (2.0 * i.to(tl.float32)) / head_dim
    log_base = tl.log(base + 0.0)  # ensure float
    freq = tl.exp(-exponent * log_base)
    angle = pos.to(tl.float32) * freq

    c = tl.cos(angle)
    s = tl.sin(angle)

    # Load pair
    head_offset = rem * seq_len * head_dim + pos * head_dim
    x0 = tl.load(QK_ptr + head_offset + 2 * i, mask=mask).to(tl.float32)
    x1 = tl.load(QK_ptr + head_offset + 2 * i + 1, mask=mask).to(tl.float32)

    # Rotate
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c

    # Store
    tl.store(QK_ptr + head_offset + 2 * i, y0.to(tl.float16), mask=mask)
    tl.store(QK_ptr + head_offset + 2 * i + 1, y1.to(tl.float16), mask=mask)


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL 3: Precompute frequency table
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def _rope_precompute_freqs_kernel(
    cos_ptr,        # [max_seq_len, half_dim] fp32
    sin_ptr,        # [max_seq_len, half_dim] fp32
    max_seq_len,
    head_dim: tl.constexpr,
    HALF_DIM: tl.constexpr,
    base: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Precompute cos/sin tables on device."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    total = max_seq_len * HALF_DIM
    mask = offs < total

    pos = offs // HALF_DIM
    i   = offs % HALF_DIM

    exponent = (2.0 * i.to(tl.float32)) / head_dim
    log_base = tl.log(base + 0.0)
    freq = tl.exp(-exponent * log_base)
    angle = pos.to(tl.float32) * freq

    c = tl.cos(angle)
    s = tl.sin(angle)

    tl.store(cos_ptr + offs, c, mask=mask)
    tl.store(sin_ptr + offs, s, mask=mask)


# ═════════════════════════════════════════════════════════════════════════════
# PYTHON WRAPPERS
# ═════════════════════════════════════════════════════════════════════════════

BLOCK_SIZE = 1024


def triton_rope_precompute_freqs(max_seq_len: int, head_dim: int,
                                  base: float = 10000.0,
                                  device: str = "cuda"):
    """
    Precompute RoPE cos/sin tables on device.

    Args:
        max_seq_len: Maximum sequence length
        head_dim: Head dimension (must be even)
        base: Frequency base (default 10000.0)
        device: Device string

    Returns:
        cos_table: [max_seq_len, head_dim/2] fp32
        sin_table: [max_seq_len, head_dim/2] fp32
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"

    half_dim = head_dim // 2
    cos_table = torch.empty(max_seq_len, half_dim, dtype=torch.float32, device=device)
    sin_table = torch.empty(max_seq_len, half_dim, dtype=torch.float32, device=device)

    total = max_seq_len * half_dim
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _rope_precompute_freqs_kernel[grid](
        cos_table, sin_table,
        max_seq_len,
        head_dim, half_dim, base,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return cos_table, sin_table


def triton_rope_forward(Q, K, cos_table, sin_table):
    """
    Apply RoPE to Q and K using precomputed cos/sin tables — Triton.

    Args:
        Q: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor (modified in-place)
        K: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor (modified in-place)
        cos_table: [max_seq_len, head_dim/2] fp32
        sin_table: [max_seq_len, head_dim/2] fp32

    Returns:
        Q, K (modified in-place, returned for convenience)
    """
    # Validate
    assert Q.is_cuda, "Q must be on CUDA device"
    assert K.is_cuda, "K must be on CUDA device"
    assert Q.dtype == torch.float16, "Q must be float16"
    assert K.dtype == torch.float16, "K must be float16"
    assert Q.dim() == 4, "Q must be 4-D [B, H, N, D]"
    assert K.dim() == 4, "K must be 4-D [B, H, N, D]"
    assert Q.is_contiguous(), "Q must be contiguous"
    assert K.is_contiguous(), "K must be contiguous"

    batch, num_heads, seq_len, head_dim = Q.shape
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    assert K.shape == Q.shape, "K shape must match Q shape"
    assert cos_table.shape[0] >= seq_len, "cos_table too short for seq_len"
    assert cos_table.shape[1] == head_dim // 2, "cos_table half_dim mismatch"

    half_dim = head_dim // 2
    total = batch * num_heads * seq_len * half_dim
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Apply to Q
    _rope_forward_kernel[grid](
        Q, cos_table, sin_table,
        batch, num_heads, seq_len,
        head_dim, half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Apply to K
    _rope_forward_kernel[grid](
        K, cos_table, sin_table,
        batch, num_heads, seq_len,
        head_dim, half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return Q, K


def triton_rope_forward_fused(Q, K, base=10000.0):
    """
    Apply RoPE to Q and K with on-the-fly sin/cos — Triton fused variant.

    Computes frequencies in-register per thread, no precomputed table needed.
    Saves HBM bandwidth but does more math per element.

    Args:
        Q: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor (modified in-place)
        K: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor (modified in-place)
        base: Frequency base (default 10000.0)

    Returns:
        Q, K (modified in-place, returned for convenience)
    """
    assert Q.is_cuda, "Q must be on CUDA device"
    assert K.is_cuda, "K must be on CUDA device"
    assert Q.dtype == torch.float16, "Q must be float16"
    assert K.dtype == torch.float16, "K must be float16"
    assert Q.dim() == 4, "Q must be 4-D [B, H, N, D]"
    assert K.dim() == 4, "K must be 4-D [B, H, N, D]"
    assert Q.is_contiguous(), "Q must be contiguous"
    assert K.is_contiguous(), "K must be contiguous"

    batch, num_heads, seq_len, head_dim = Q.shape
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    assert K.shape == Q.shape, "K shape must match Q shape"

    half_dim = head_dim // 2
    total = batch * num_heads * seq_len * half_dim
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Apply to Q
    _rope_forward_fused_kernel[grid](
        Q, batch, num_heads, seq_len,
        head_dim, half_dim, base,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Apply to K
    _rope_forward_fused_kernel[grid](
        K, batch, num_heads, seq_len,
        head_dim, half_dim, base,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return Q, K
