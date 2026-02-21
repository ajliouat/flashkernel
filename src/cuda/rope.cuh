/**
 * FlashKernel — Rotary Position Embedding header (v1.0.5)
 *
 * Applies Rotary Position Embedding (RoPE) to Q and K tensors:
 *   q_rot[2i]   = q[2i]   * cos(θ) - q[2i+1] * sin(θ)
 *   q_rot[2i+1] = q[2i]   * sin(θ) + q[2i+1] * cos(θ)
 *
 * where θ_i = pos / 10000^(2i/d)
 *
 * Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
 *            Position Embedding" (2021)
 *
 * Architecture target: SM 7.5 (Turing / T4)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flashkernel {

/**
 * rope_precompute_freqs — Precompute sin/cos frequency table on device.
 *
 * Computes:
 *   freq[i] = 1.0 / (base^(2i/d))          for i in [0, d/2)
 *   cos_table[pos][i] = cos(pos * freq[i])  for pos in [0, max_seq_len)
 *   sin_table[pos][i] = sin(pos * freq[i])
 *
 * @param cos_table   [max_seq_len, d/2] fp32, output
 * @param sin_table   [max_seq_len, d/2] fp32, output
 * @param max_seq_len Maximum sequence length
 * @param head_dim    Head dimension (d) — must be even
 * @param base        Frequency base (default: 10000.0)
 * @param stream      CUDA stream
 */
void rope_precompute_freqs(
    float* cos_table,
    float* sin_table,
    int max_seq_len,
    int head_dim,
    float base,
    cudaStream_t stream = 0
);

/**
 * rope_forward — Apply RoPE to Q and K tensors in-place.
 *
 * @param Q           [batch, num_heads, seq_len, head_dim] fp16, modified in-place
 * @param K           [batch, num_heads, seq_len, head_dim] fp16, modified in-place
 * @param cos_table   [max_seq_len, head_dim/2] fp32 — precomputed cos
 * @param sin_table   [max_seq_len, head_dim/2] fp32 — precomputed sin
 * @param batch       Batch size
 * @param num_heads   Number of attention heads
 * @param seq_len     Sequence length (must be <= max_seq_len used for table)
 * @param head_dim    Head dimension (must be even)
 * @param stream      CUDA stream
 */
void rope_forward(
    half* Q,
    half* K,
    const float* cos_table,
    const float* sin_table,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream = 0
);

/**
 * rope_forward_fused — Precompute freqs on-the-fly and apply RoPE.
 *
 * Avoids materializing cos/sin tables in HBM — computes sin/cos
 * in-register per thread. Slightly more compute, but saves bandwidth
 * for one-shot inference.
 *
 * @param Q           [batch, num_heads, seq_len, head_dim] fp16, modified in-place
 * @param K           [batch, num_heads, seq_len, head_dim] fp16, modified in-place
 * @param batch       Batch size
 * @param num_heads   Number of attention heads
 * @param seq_len     Sequence length
 * @param head_dim    Head dimension (must be even)
 * @param base        Frequency base (default: 10000.0)
 * @param stream      CUDA stream
 */
void rope_forward_fused(
    half* Q,
    half* K,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    float base,
    cudaStream_t stream = 0
);

}  // namespace flashkernel
