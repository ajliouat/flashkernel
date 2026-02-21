"""
FlashKernel — Paged KV-Cache Triton Kernels (v1.0.6)

Triton implementations of the paged KV-cache append and read (scatter-gather)
operations. These mirror the CUDA kernels in paged_kv_cache.cu.

Pool layout: [num_pages, 2(K/V), num_heads, page_size, head_dim] fp16

The append kernel writes new KV tokens into the pool via a slot mapping.
The read kernel gathers KV from scattered pages into contiguous output.

Architecture target: Any GPU supported by Triton (tested on SM 7.5 / T4)
"""

import torch
import triton
import triton.language as tl


# ─── Append Kernel ───────────────────────────────────────────────────────────
#
# Writes new K, V tokens into the page pool using a pre-computed slot mapping.
# slot_mapping[token_idx] = physical_page * page_size + offset_within_page


@triton.jit
def _paged_kv_append_kernel(
    pool_ptr,
    slot_mapping_ptr,
    new_keys_ptr,
    new_values_ptr,
    total_tokens,
    num_heads,
    page_size,
    head_dim,
    # Pool strides (pre-computed on host for efficiency)
    pool_kv_stride,     # H * S * D
    pool_page_stride,   # 2 * H * S * D
    pool_head_stride,   # S * D
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = total_tokens * num_heads * head_dim
    mask = offsets < total

    # Decode: (token_idx, head, dim)
    d = offsets % head_dim
    h = (offsets // head_dim) % num_heads
    token_idx = offsets // (num_heads * head_dim)

    # Slot mapping lookup
    slot = tl.load(slot_mapping_ptr + token_idx, mask=mask, other=0)
    page = slot // page_size
    offset_in_page = slot % page_size

    # Pool index for K (kv=0)
    pool_k_idx = (page * pool_page_stride
                  + h * pool_head_stride
                  + offset_in_page * head_dim
                  + d)

    # Source index in [total_tokens, num_heads, head_dim]
    src_idx = token_idx * (num_heads * head_dim) + h * head_dim + d

    # Load new K, V
    k_val = tl.load(new_keys_ptr + src_idx, mask=mask, other=0.0)
    v_val = tl.load(new_values_ptr + src_idx, mask=mask, other=0.0)

    # Store K and V
    tl.store(pool_ptr + pool_k_idx, k_val, mask=mask)
    tl.store(pool_ptr + pool_k_idx + pool_kv_stride, v_val, mask=mask)


# ─── Read/Gather Kernel ─────────────────────────────────────────────────────
#
# Gathers KV from scattered pages into contiguous [B, H, N, D] output.
# Each thread handles one output element, looking up the page table.


@triton.jit
def _paged_kv_read_kernel(
    pool_ptr,
    block_table_ptr,
    seq_lens_ptr,
    K_out_ptr,
    V_out_ptr,
    batch,
    num_heads,
    max_seq_len,
    page_size,
    head_dim,
    max_blocks_per_seq,
    # Pool strides
    pool_kv_stride,
    pool_page_stride,
    pool_head_stride,
    # Output strides
    out_batch_stride,   # H * max_seq_len * D
    out_head_stride,    # max_seq_len * D
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch * num_heads * max_seq_len * head_dim
    mask = offsets < total

    # Decode: (b, h, pos, d)
    d   = offsets % head_dim
    pos = (offsets // head_dim) % max_seq_len
    h   = (offsets // (head_dim * max_seq_len)) % num_heads
    b   = offsets // (num_heads * max_seq_len * head_dim)

    # Check valid position
    seq_len = tl.load(seq_lens_ptr + b, mask=mask, other=0)
    valid = mask & (pos < seq_len)

    # Page table lookup
    logical_page = pos // page_size
    offset_in_page = pos % page_size
    bt_idx = b * max_blocks_per_seq + logical_page
    physical_page = tl.load(block_table_ptr + bt_idx, mask=valid, other=0)

    # Pool index for K
    pool_k_idx = (physical_page * pool_page_stride
                  + h * pool_head_stride
                  + offset_in_page * head_dim
                  + d)

    # Load K, V from pool
    k_val = tl.load(pool_ptr + pool_k_idx, mask=valid, other=0.0)
    v_val = tl.load(pool_ptr + pool_k_idx + pool_kv_stride, mask=valid, other=0.0)

    # Output index: [b, h, pos, d]
    out_idx = b * out_batch_stride + h * out_head_stride + pos * head_dim + d

    tl.store(K_out_ptr + out_idx, k_val, mask=valid)
    tl.store(V_out_ptr + out_idx, v_val, mask=valid)


# ─── Python Wrappers ────────────────────────────────────────────────────────

BLOCK_SIZE = 1024  # Triton BLOCK_SIZE (larger than CUDA for coalescing)


def triton_paged_kv_cache_append(
    pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    new_keys: torch.Tensor,
    new_values: torch.Tensor,
) -> None:
    """
    Append new KV tokens to the page pool — Triton kernel.

    Args:
        pool:         [num_pages, 2, num_heads, page_size, head_dim] fp16
        slot_mapping: [total_tokens] int32
        new_keys:     [total_tokens, num_heads, head_dim] fp16
        new_values:   [total_tokens, num_heads, head_dim] fp16

    Modifies pool in-place.
    """
    assert pool.is_cuda and slot_mapping.is_cuda
    assert new_keys.is_cuda and new_values.is_cuda

    total_tokens = new_keys.shape[0]
    num_heads = new_keys.shape[1]
    head_dim = new_keys.shape[2]
    num_pages = pool.shape[0]
    page_size = pool.shape[3]

    if total_tokens == 0:
        return

    # Pre-compute pool strides
    pool_head_stride = page_size * head_dim
    pool_kv_stride = num_heads * pool_head_stride
    pool_page_stride = 2 * pool_kv_stride

    total_elements = total_tokens * num_heads * head_dim
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    _paged_kv_append_kernel[(grid,)](
        pool, slot_mapping, new_keys, new_values,
        total_tokens, num_heads, page_size, head_dim,
        pool_kv_stride, pool_page_stride, pool_head_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def triton_paged_kv_cache_read(
    pool: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter-gather read from paged KV cache — Triton kernel.

    Args:
        pool:        [num_pages, 2, num_heads, page_size, head_dim] fp16
        block_table: [batch, max_blocks_per_seq] int32
        seq_lens:    [batch] int32
        max_seq_len: Max sequence length for output

    Returns:
        K_out: [batch, num_heads, max_seq_len, head_dim] fp16
        V_out: [batch, num_heads, max_seq_len, head_dim] fp16
    """
    assert pool.is_cuda and block_table.is_cuda and seq_lens.is_cuda

    batch = block_table.shape[0]
    max_blocks_per_seq = block_table.shape[1]
    num_heads = pool.shape[2]
    page_size = pool.shape[3]
    head_dim = pool.shape[4]

    # Allocate output (pre-zeroed for padding positions)
    K_out = torch.zeros(batch, num_heads, max_seq_len, head_dim,
                        dtype=pool.dtype, device=pool.device)
    V_out = torch.zeros(batch, num_heads, max_seq_len, head_dim,
                        dtype=pool.dtype, device=pool.device)

    total = batch * num_heads * max_seq_len * head_dim
    if total == 0:
        return K_out, V_out

    # Pool strides
    pool_head_stride = page_size * head_dim
    pool_kv_stride = num_heads * pool_head_stride
    pool_page_stride = 2 * pool_kv_stride

    # Output strides
    out_batch_stride = num_heads * max_seq_len * head_dim
    out_head_stride = max_seq_len * head_dim

    grid = (total + BLOCK_SIZE - 1) // BLOCK_SIZE

    _paged_kv_read_kernel[(grid,)](
        pool, block_table, seq_lens, K_out, V_out,
        batch, num_heads, max_seq_len, page_size, head_dim,
        max_blocks_per_seq,
        pool_kv_stride, pool_page_stride, pool_head_stride,
        out_batch_stride, out_head_stride,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return K_out, V_out
