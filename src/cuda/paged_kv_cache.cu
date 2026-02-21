/**
 * FlashKernel — Paged KV-Cache CUDA kernels (v1.0.6)
 *
 * Two kernels:
 *   1. paged_kv_cache_append — Write new KV tokens into pool via slot mapping
 *   2. paged_kv_cache_read   — Scatter-gather from pool into contiguous output
 *
 * Pool layout: [num_pages, 2(K/V), num_heads, page_size, head_dim] fp16
 *
 * The page allocation logic (free list, block table management) lives in
 * Python (PagedKVCache class). These kernels handle only GPU data movement,
 * maximizing parallelism on the scatter/gather pattern.
 *
 * Architecture: SM 7.5 (Turing / T4)
 * Compiled with: nvcc -O3 --use_fast_math -arch=sm_75
 */

#include "paged_kv_cache.cuh"

namespace flashkernel {

// ─── Constants ──────────────────────────────────────────────────────────────

constexpr int BLOCK_SIZE = 256;

// ─── Pool stride helpers ────────────────────────────────────────────────────
//
// Pool shape: [P, 2, H, S, D]  (P=pages, H=heads, S=page_size, D=head_dim)
//   page_stride = 2 * H * S * D
//   kv_stride   = H * S * D        (offset from K to V within same page)
//   head_stride = S * D
//
// K index: page * page_stride + 0 * kv_stride + head * head_stride + off * D + d
// V index: page * page_stride + 1 * kv_stride + head * head_stride + off * D + d

// ─── Append Kernel ──────────────────────────────────────────────────────────
//
// Grid:  ceil(total_tokens * num_heads * head_dim / BLOCK_SIZE)
// Each thread writes one (token, head, dim) element to both K and V slots.

__global__ void paged_kv_cache_append_kernel(
    half* __restrict__ pool,
    const int* __restrict__ slot_mapping,
    const half* __restrict__ new_keys,
    const half* __restrict__ new_values,
    int total_tokens,
    int num_heads,
    int page_size,
    int head_dim,
    int num_pages
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = total_tokens * num_heads * head_dim;
    if (tid >= total_elements) return;

    // Decode flat index → (token_idx, head, dim)
    const int d         = tid % head_dim;
    const int h         = (tid / head_dim) % num_heads;
    const int token_idx = tid / (num_heads * head_dim);

    // Look up physical slot for this token
    const int slot   = slot_mapping[token_idx];
    const int page   = slot / page_size;
    const int offset = slot % page_size;

    // Bounds check
    if (page < 0 || page >= num_pages) return;

    // Compute pool strides
    const int kv_stride   = num_heads * page_size * head_dim;
    const int page_stride = 2 * kv_stride;
    const int head_stride = page_size * head_dim;

    // Pool index for K (kv=0)
    const int pool_k_idx = page * page_stride
                         + h * head_stride
                         + offset * head_dim
                         + d;

    // Source index in [total_tokens, num_heads, head_dim] layout
    const int src_idx = token_idx * (num_heads * head_dim)
                      + h * head_dim
                      + d;

    // Write K and V
    pool[pool_k_idx]              = new_keys[src_idx];
    pool[pool_k_idx + kv_stride]  = new_values[src_idx];
}


// ─── Read/Gather Kernel ─────────────────────────────────────────────────────
//
// Grid:  ceil(batch * num_heads * max_seq_len * head_dim / BLOCK_SIZE)
// Each thread reads one (batch, head, position, dim) element for both K and V.
// Positions beyond seq_lens[batch] are left as zero (output is pre-zeroed).

__global__ void paged_kv_cache_read_kernel(
    const half* __restrict__ pool,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    half* __restrict__ K_out,
    half* __restrict__ V_out,
    int batch,
    int num_heads,
    int max_seq_len,
    int page_size,
    int head_dim,
    int max_blocks_per_seq
) {
    const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * num_heads * max_seq_len * head_dim;
    if (tid >= total) return;

    // Decode flat index → (b, h, pos, d)
    const int d   = tid % head_dim;
    const int pos = (tid / head_dim) % max_seq_len;
    const int h   = (tid / (head_dim * max_seq_len)) % num_heads;
    const int b   = tid / (num_heads * max_seq_len * head_dim);

    // Skip padding positions
    const int seq_len = seq_lens[b];
    if (pos >= seq_len) return;

    // Page table lookup
    const int logical_page  = pos / page_size;
    const int offset        = pos % page_size;
    const int physical_page = block_table[b * max_blocks_per_seq + logical_page];

    // Pool strides
    const int kv_stride   = num_heads * page_size * head_dim;
    const int page_stride = 2 * kv_stride;
    const int head_stride = page_size * head_dim;

    // Read from pool
    const int pool_k_idx = physical_page * page_stride
                         + h * head_stride
                         + offset * head_dim
                         + d;

    // Output index: [b, h, pos, d]
    const int out_idx = b * (num_heads * max_seq_len * head_dim)
                      + h * (max_seq_len * head_dim)
                      + pos * head_dim
                      + d;

    K_out[out_idx] = pool[pool_k_idx];
    V_out[out_idx] = pool[pool_k_idx + kv_stride];
}


// ─── Host Launchers ─────────────────────────────────────────────────────────

void paged_kv_cache_append(
    half* pool,
    const int* slot_mapping,
    const half* new_keys,
    const half* new_values,
    int total_tokens,
    int num_heads,
    int page_size,
    int head_dim,
    int num_pages,
    cudaStream_t stream
) {
    const int total_elements = total_tokens * num_heads * head_dim;
    if (total_elements == 0) return;

    const int grid = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    paged_kv_cache_append_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        pool, slot_mapping, new_keys, new_values,
        total_tokens, num_heads, page_size, head_dim, num_pages
    );
}


void paged_kv_cache_read(
    const half* pool,
    const int* block_table,
    const int* seq_lens,
    half* K_out,
    half* V_out,
    int batch,
    int num_heads,
    int max_seq_len,
    int page_size,
    int head_dim,
    int max_blocks_per_seq,
    cudaStream_t stream
) {
    const int total = batch * num_heads * max_seq_len * head_dim;
    if (total == 0) return;

    const int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    paged_kv_cache_read_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        pool, block_table, seq_lens, K_out, V_out,
        batch, num_heads, max_seq_len, page_size, head_dim, max_blocks_per_seq
    );
}

}  // namespace flashkernel
