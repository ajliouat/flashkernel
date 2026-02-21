/**
 * FlashKernel — Rotary Position Embedding CUDA kernel (v1.0.5)
 *
 * Implements RoPE (Su et al., 2021) — the position encoding used in
 * LLaMA, Mistral, GPT-NeoX, and most modern LLMs.
 *
 * Key idea: Encode position information by rotating pairs of dimensions
 * in Q and K using position-dependent angles. This creates a relative
 * position encoding that decays with distance:
 *   <RoPE(q, pos_m), RoPE(k, pos_n)> depends only on (m - n)
 *
 * Implementation strategy:
 *   1. PRECOMPUTED: Build cos/sin tables on device, then apply via lookup.
 *      Better when reusing tables across multiple forward passes.
 *   2. FUSED: Compute sin/cos per-thread on the fly using __sincosf().
 *      Saves HBM bandwidth (no table reads), better for one-shot inference.
 *
 * Both variants apply the rotation in-place on Q and K.
 *
 * Architecture target: SM 7.5 (Turing / T4)
 */

#include "rope.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

namespace flashkernel {

// ─── Constants ──────────────────────────────────────────────────────────────

static constexpr int BLOCK_SIZE = 256;

// ═════════════════════════════════════════════════════════════════════════════
// KERNEL 1: Precompute sin/cos frequency table
// ═════════════════════════════════════════════════════════════════════════════

__global__ void rope_precompute_freqs_kernel(
    float* __restrict__ cos_table,   // [max_seq_len, half_dim]
    float* __restrict__ sin_table,   // [max_seq_len, half_dim]
    int max_seq_len,
    int half_dim,
    float base
) {
    // Each thread handles one (pos, dim_pair) element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max_seq_len * half_dim;

    if (idx >= total) return;

    int pos = idx / half_dim;
    int i   = idx % half_dim;  // dimension pair index

    // θ_i = pos / base^(2i / head_dim)
    // In log space: log(θ) = log(pos) - (2i / head_dim) * log(base)
    // But direct: freq = 1.0 / base^(2i / (2 * half_dim))
    float freq = 1.0f / powf(base, (float)(2 * i) / (float)(2 * half_dim));
    float angle = (float)pos * freq;

    // Use __sincosf for fast simultaneous sin+cos
    float s, c;
    __sincosf(angle, &s, &c);

    cos_table[idx] = c;
    sin_table[idx] = s;
}

// ═════════════════════════════════════════════════════════════════════════════
// KERNEL 2: Apply RoPE with precomputed tables (table lookup)
// ═════════════════════════════════════════════════════════════════════════════

__global__ void rope_forward_kernel(
    half* __restrict__ QK,             // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ cos_table,  // [max_seq_len, half_dim]
    const float* __restrict__ sin_table,  // [max_seq_len, half_dim]
    int batch,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Grid: one thread per (batch, head, pos, dim_pair)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = batch * num_heads * seq_len * half_dim;

    if (idx >= total) return;

    // Decompose flat index → (b, h, pos, i)
    int i   = idx % half_dim;
    int rem = idx / half_dim;
    int pos = rem % seq_len;
    rem     = rem / seq_len;
    int h   = rem % num_heads;
    int b   = rem / num_heads;

    // Pointer to this token's head vector
    int head_offset = ((b * num_heads + h) * seq_len + pos) * head_dim;

    // Load the pair (x_2i, x_{2i+1}) in fp32
    float x0 = __half2float(QK[head_offset + 2 * i]);
    float x1 = __half2float(QK[head_offset + 2 * i + 1]);

    // Lookup precomputed cos/sin
    float c = cos_table[pos * half_dim + i];
    float s = sin_table[pos * half_dim + i];

    // Apply rotation:
    //   x_rot[2i]   = x[2i]   * cos - x[2i+1] * sin
    //   x_rot[2i+1] = x[2i]   * sin + x[2i+1] * cos
    float y0 = x0 * c - x1 * s;
    float y1 = x0 * s + x1 * c;

    // Store back in fp16
    QK[head_offset + 2 * i]     = __float2half(y0);
    QK[head_offset + 2 * i + 1] = __float2half(y1);
}

// ═════════════════════════════════════════════════════════════════════════════
// KERNEL 3: Fused RoPE — compute sin/cos on the fly (no table lookup)
// ═════════════════════════════════════════════════════════════════════════════

__global__ void rope_forward_fused_kernel(
    half* __restrict__ QK,             // [batch, num_heads, seq_len, head_dim]
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    float base
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = batch * num_heads * seq_len * half_dim;

    if (idx >= total) return;

    // Decompose flat index → (b, h, pos, i)
    int i   = idx % half_dim;
    int rem = idx / half_dim;
    int pos = rem % seq_len;
    rem     = rem / seq_len;
    int h   = rem % num_heads;
    int b   = rem / num_heads;

    // Compute frequency in-register
    float freq = 1.0f / powf(base, (float)(2 * i) / (float)(2 * half_dim));
    float angle = (float)pos * freq;

    float s, c;
    __sincosf(angle, &s, &c);

    // Load pair
    int head_offset = ((b * num_heads + h) * seq_len + pos) * head_dim;
    float x0 = __half2float(QK[head_offset + 2 * i]);
    float x1 = __half2float(QK[head_offset + 2 * i + 1]);

    // Rotate
    float y0 = x0 * c - x1 * s;
    float y1 = x0 * s + x1 * c;

    // Store
    QK[head_offset + 2 * i]     = __float2half(y0);
    QK[head_offset + 2 * i + 1] = __float2half(y1);
}

// ═════════════════════════════════════════════════════════════════════════════
// HOST LAUNCHERS
// ═════════════════════════════════════════════════════════════════════════════

void rope_precompute_freqs(
    float* cos_table,
    float* sin_table,
    int max_seq_len,
    int head_dim,
    float base,
    cudaStream_t stream
) {
    int half_dim = head_dim / 2;
    int total = max_seq_len * half_dim;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    rope_precompute_freqs_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        cos_table, sin_table, max_seq_len, half_dim, base
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "rope_precompute_freqs kernel error: %s\n",
                cudaGetErrorString(err));
    }
}

void rope_forward(
    half* Q,
    half* K,
    const float* cos_table,
    const float* sin_table,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    int half_dim = head_dim / 2;
    int total = batch * num_heads * seq_len * half_dim;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Apply to Q
    rope_forward_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        Q, cos_table, sin_table, batch, num_heads, seq_len, head_dim
    );

    // Apply to K (same rotation — they share the same position encoding)
    rope_forward_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        K, cos_table, sin_table, batch, num_heads, seq_len, head_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "rope_forward kernel error: %s\n",
                cudaGetErrorString(err));
    }
}

void rope_forward_fused(
    half* Q,
    half* K,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    float base,
    cudaStream_t stream
) {
    int half_dim = head_dim / 2;
    int total = batch * num_heads * seq_len * half_dim;
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Apply to Q
    rope_forward_fused_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        Q, batch, num_heads, seq_len, head_dim, base
    );

    // Apply to K
    rope_forward_fused_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        K, batch, num_heads, seq_len, head_dim, base
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "rope_forward_fused kernel error: %s\n",
                cudaGetErrorString(err));
    }
}

}  // namespace flashkernel
