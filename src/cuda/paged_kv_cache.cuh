/**
 * FlashKernel — Paged KV-Cache header (v1.0.6)
 *
 * Block-level KV cache with dynamic page allocation and scatter-gather
 * reads. Eliminates pre-allocated max-length buffers by storing KV data
 * in fixed-size pages and using a page table for indirection.
 *
 * Design:
 *   Pool:       [num_pages, 2(K/V), num_heads, page_size, head_dim] fp16
 *   Page table: [batch, max_blocks_per_seq] int32 → physical page index
 *   Slot map:   logical token → physical_page * page_size + offset
 *
 * Operations:
 *   append  — write new KV tokens to pool via slot mapping
 *   read    — scatter-gather from pool into contiguous output
 *
 * Reference: Kwon et al., "Efficient Memory Management for Large Language
 *            Model Serving with PagedAttention" (vLLM, 2023)
 *
 * Architecture target: SM 7.5 (Turing / T4)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flashkernel {

/**
 * paged_kv_cache_append — Write new KV tokens into the page pool.
 *
 * Each new token is mapped to a physical slot via slot_mapping:
 *   slot = physical_page_index * page_size + offset_within_page
 *
 * The caller (Python-side PagedKVCache) manages page allocation and
 * computes slot_mapping. This kernel only does the data movement.
 *
 * @param pool          [num_pages, 2, num_heads, page_size, head_dim] fp16
 * @param slot_mapping  [total_tokens] int32 — flat physical slot per token
 * @param new_keys      [total_tokens, num_heads, head_dim] fp16
 * @param new_values    [total_tokens, num_heads, head_dim] fp16
 * @param total_tokens  Number of new tokens to append
 * @param num_heads     Number of attention heads
 * @param page_size     Tokens per page
 * @param head_dim      Dimension per head
 * @param num_pages     Total pages in pool (for bounds safety)
 * @param stream        CUDA stream
 */
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
    cudaStream_t stream = 0
);

/**
 * paged_kv_cache_read — Scatter-gather KV from pages into contiguous output.
 *
 * For each (batch, head, position, dim), the kernel looks up:
 *   physical_page = block_table[batch][position / page_size]
 *   offset        = position % page_size
 * and reads K, V from pool[physical_page][kv][head][offset][dim].
 *
 * Positions beyond seq_lens[batch] are left as zero.
 *
 * @param pool              [num_pages, 2, num_heads, page_size, head_dim] fp16
 * @param block_table       [batch, max_blocks_per_seq] int32 — physical page indices
 * @param seq_lens          [batch] int32 — actual sequence length per entry
 * @param K_out             [batch, num_heads, max_seq_len, head_dim] fp16, output
 * @param V_out             [batch, num_heads, max_seq_len, head_dim] fp16, output
 * @param batch             Batch size
 * @param num_heads         Number of attention heads
 * @param max_seq_len       Maximum sequence length for output allocation
 * @param page_size         Tokens per page
 * @param head_dim          Dimension per head
 * @param max_blocks_per_seq Maximum pages per sequence in block_table
 * @param stream            CUDA stream
 */
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
    cudaStream_t stream = 0
);

}  // namespace flashkernel
