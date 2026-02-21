/**
 * FlashKernel — FlashAttention header (v1.0.2)
 *
 * Tiled FlashAttention forward pass:
 *   - Online softmax (no N×N materialization)
 *   - fp16 compute with fp32 accumulators
 *   - Causal masking support
 *
 * Tile sizes:
 *   head_dim=64:  Br=64, Bc=64  (32 KB shared mem)
 *   head_dim=128: Br=32, Bc=64  (44 KB shared mem)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flashkernel {

/**
 * flash_attention_forward — Tiled FlashAttention (CUDA)
 *
 * Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * @param Q         [batch, num_heads, seq_len, head_dim] fp16, row-major
 * @param K         [batch, num_heads, seq_len, head_dim] fp16, row-major
 * @param V         [batch, num_heads, seq_len, head_dim] fp16, row-major
 * @param O         [batch, num_heads, seq_len, head_dim] fp16, output
 * @param L         [batch, num_heads, seq_len]            fp32, log-sum-exp (for backward)
 * @param batch     Batch size
 * @param num_heads Number of attention heads
 * @param seq_len   Sequence length (Q and KV same length)
 * @param head_dim  Head dimension (64 or 128)
 * @param scale     Softmax scale factor (typically 1/sqrt(head_dim))
 * @param is_causal If true, apply causal mask (upper triangle = -inf)
 * @param stream    CUDA stream
 */
void flash_attention_forward(
    const half* Q,
    const half* K,
    const half* V,
    half* O,
    float* L,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal,
    cudaStream_t stream = 0
);

}  // namespace flashkernel
