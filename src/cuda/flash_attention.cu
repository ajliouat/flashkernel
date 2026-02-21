/**
 * FlashKernel — Tiled FlashAttention Forward Pass (v1.0.2)
 *
 * Algorithm (Dao et al., 2022):
 *   For each Q-block of Br rows:
 *     Initialize O=0, m=-inf, l=0 in registers
 *     For each KV-block of Bc rows:
 *       Load Q_block, K_block, V_block to shared memory
 *       S = Q_block @ K_block^T * scale        (Br × Bc in shared mem)
 *       Apply causal mask if needed
 *       m_new = max(m, rowmax(S))
 *       P = exp(S - m_new)                     (online softmax)
 *       l_new = exp(m - m_new) * l + rowsum(P)
 *       O = exp(m - m_new) * O + P @ V_block   (accumulate in fp32)
 *       m = m_new, l = l_new
 *     O = O / l                                (normalize)
 *     Write O (cast to fp16) and L=m+log(l) to HBM
 *
 * Tile sizes (SM 7.5 / T4, 48 KB shared memory per SM):
 *   head_dim=64:  Br=64, Bc=64  → Q:8K + K:8K + S:8K + V:8K = 32 KB  ✓
 *   head_dim=128: Br=32, Bc=64  → Q:8K + K:16K + S:4K + V:16K = 44 KB ✓
 *
 * Memory layout: [batch, num_heads, seq_len, head_dim] — row-major / contiguous
 * All intermediate computation in fp32 for numerical stability.
 *
 * Architecture target: SM 7.5 (Turing / T4)
 */

#include "flash_attention.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>
#include <algorithm>

namespace flashkernel {

// ─── Constants ──────────────────────────────────────────────────────────────

static constexpr int WARP_SIZE = 32;
static constexpr unsigned FULL_MASK = 0xffffffff;

// ─── Warp Utilities ─────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return __shfl_sync(FULL_MASK, val, 0);  // broadcast max to all lanes
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return __shfl_sync(FULL_MASK, val, 0);  // broadcast sum to all lanes
}

// ═════════════════════════════════════════════════════════════════════════════
// FLASH ATTENTION KERNEL — head_dim=64, Br=64, Bc=64
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Each block handles one (batch, head, q_block) = Br=64 query rows.
 * Block dimensions: (Bc=64 threads)  — one thread per column in the KV tile.
 * Actually: we use (Br=64 threads in x) so each thread handles one query row.
 *
 * Strategy:
 *   - 64 threads in the block, each "owns" one row of Q (one query)
 *   - For each K/V block: compute S row, update running max/sum/output
 *   - Thread i computes the full attention for query row i
 *
 * Shared memory layout (head_dim=64):
 *   Q_smem[Br][d]  = 64 × 64 × 2B = 8 KB   (loaded once per Q block)
 *   K_smem[Bc][d]  = 64 × 64 × 2B = 8 KB   (loaded per KV block)
 *   V_smem[Bc][d]  = 64 × 64 × 2B = 8 KB   (loaded per KV block)
 *   S_smem[Br][Bc] = 64 × 64 × 4B = 16 KB  (scores, fp32)
 *   Total: 40 KB  ✓ (fits T4 48 KB)
 */

// Template parameters for tile sizes
template <int Br, int Bc, int D>
__global__ void flash_attention_forward_kernel(
    const half* __restrict__ Q,       // [B, H, N, D]
    const half* __restrict__ K,       // [B, H, N, D]
    const half* __restrict__ V,       // [B, H, N, D]
    half*       __restrict__ O,       // [B, H, N, D]
    float*      __restrict__ L,       // [B, H, N]
    const int N,                      // seq_len
    const float scale,
    const bool is_causal
) {
    // Block/thread indices
    const int bh    = blockIdx.x;     // batch * head index (flattened)
    const int q_blk = blockIdx.y;     // which Q block (0 .. ceil(N/Br)-1)
    const int tid   = threadIdx.x;    // thread within block: 0..Br-1

    // Each thread "owns" one query row
    const int q_row = q_blk * Br + tid;
    if (q_row >= N) return;

    // Base pointers for this (batch, head)
    const int bh_offset = bh * N * D;
    const half* Q_bh = Q + bh_offset;
    const half* K_bh = K + bh_offset;
    const half* V_bh = V + bh_offset;
    half* O_bh       = O + bh_offset;
    float* L_bh      = L + bh * N;

    // ─── Shared memory ──────────────────────────────────────────────────
    // We use dynamic shared memory, partitioned as:
    //   Q_smem:  Br × D  (half)
    //   K_smem:  Bc × D  (half)
    //   V_smem:  Bc × D  (half)
    extern __shared__ char smem_raw[];

    half* Q_smem = reinterpret_cast<half*>(smem_raw);
    half* K_smem = Q_smem + Br * D;
    half* V_smem = K_smem + Bc * D;

    // ─── Load Q block to shared memory ──────────────────────────────────
    // Each thread loads its own row of Q
    if (q_row < N) {
        for (int d = 0; d < D; d++) {
            Q_smem[tid * D + d] = Q_bh[q_row * D + d];
        }
    }
    __syncthreads();

    // ─── Per-thread accumulators (in registers, fp32) ───────────────────
    float m_i = -FLT_MAX;     // running max
    float l_i = 0.0f;         // running sum of exp
    float o_i[D > 128 ? 1 : D];  // output accumulator — compile-time sized
    // Note: D is a template param so this is stack-allocated
    // Initialize output accumulators
    #pragma unroll
    for (int d = 0; d < D; d++) {
        o_i[d] = 0.0f;
    }

    // ─── Number of KV blocks ────────────────────────────────────────────
    int num_kv_blocks = (N + Bc - 1) / Bc;

    // For causal: only iterate up to the block that contains q_row
    int max_kv_block = num_kv_blocks;
    if (is_causal) {
        max_kv_block = (q_row / Bc) + 1;
    }

    // ─── Main loop over KV blocks ───────────────────────────────────────
    for (int kv_blk = 0; kv_blk < max_kv_block; kv_blk++) {
        const int kv_start = kv_blk * Bc;

        // Load K block to shared memory (cooperatively)
        // Each thread loads one row of K (tid-th row)
        {
            int k_row = kv_start + tid;
            if (k_row < N && tid < Bc) {
                for (int d = 0; d < D; d++) {
                    K_smem[tid * D + d] = K_bh[k_row * D + d];
                }
            } else if (tid < Bc) {
                // Padding for out-of-bounds
                for (int d = 0; d < D; d++) {
                    K_smem[tid * D + d] = __float2half(0.0f);
                }
            }
        }

        // Load V block to shared memory
        {
            int v_row = kv_start + tid;
            if (v_row < N && tid < Bc) {
                for (int d = 0; d < D; d++) {
                    V_smem[tid * D + d] = V_bh[v_row * D + d];
                }
            } else if (tid < Bc) {
                for (int d = 0; d < D; d++) {
                    V_smem[tid * D + d] = __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // ─── Compute S_i = Q_i · K^T * scale for this thread's query ───
        // S_ij = dot(Q[q_row, :], K[kv_start+j, :]) * scale  for j in [0, Bc)
        float row_max = -FLT_MAX;

        float s_vals[Bc > 128 ? 1 : Bc];  // compile-time Bc

        #pragma unroll
        for (int j = 0; j < Bc; j++) {
            int kv_col = kv_start + j;

            // Causal mask: if kv_col > q_row, mask out
            if (is_causal && kv_col > q_row) {
                s_vals[j] = -FLT_MAX;
                continue;
            }

            // Out-of-bounds mask
            if (kv_col >= N) {
                s_vals[j] = -FLT_MAX;
                continue;
            }

            // Dot product Q[tid,:] · K[j,:] in fp32
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                dot += __half2float(Q_smem[tid * D + d]) *
                       __half2float(K_smem[j * D + d]);
            }
            s_vals[j] = dot * scale;

            row_max = fmaxf(row_max, s_vals[j]);
        }

        // ─── Online softmax update ─────────────────────────────────────
        // m_new = max(m_i, row_max)
        float m_new = fmaxf(m_i, row_max);

        // Rescale factor for previous accumulations
        float alpha = expf(m_i - m_new);

        // Compute P = exp(S - m_new) and accumulate
        float l_new = alpha * l_i;

        // Rescale existing output accumulator
        #pragma unroll
        for (int d = 0; d < D; d++) {
            o_i[d] *= alpha;
        }

        // P @ V accumulation
        #pragma unroll
        for (int j = 0; j < Bc; j++) {
            float p_ij = (s_vals[j] > -FLT_MAX + 1.0f)
                         ? expf(s_vals[j] - m_new)
                         : 0.0f;
            l_new += p_ij;

            // Accumulate p_ij * V[j, :]
            #pragma unroll
            for (int d = 0; d < D; d++) {
                o_i[d] += p_ij * __half2float(V_smem[j * D + d]);
            }
        }

        m_i = m_new;
        l_i = l_new;

        __syncthreads();
    }

    // ─── Normalize and write output ─────────────────────────────────────
    if (q_row < N && l_i > 0.0f) {
        float inv_l = 1.0f / l_i;
        for (int d = 0; d < D; d++) {
            O_bh[q_row * D + d] = __float2half(o_i[d] * inv_l);
        }
        // Store log-sum-exp for potential backward pass
        L_bh[q_row] = m_i + logf(l_i);
    } else if (q_row < N) {
        // Edge case: all masked out
        for (int d = 0; d < D; d++) {
            O_bh[q_row * D + d] = __float2half(0.0f);
        }
        L_bh[q_row] = -FLT_MAX;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// HOST WRAPPER
// ═════════════════════════════════════════════════════════════════════════════

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
    cudaStream_t stream
) {
    // Select tile dimensions based on head_dim
    if (head_dim == 64) {
        constexpr int Br = 64;
        constexpr int Bc = 64;
        constexpr int D  = 64;

        // Grid: (batch*heads, ceil(N/Br))
        int num_q_blocks = (seq_len + Br - 1) / Br;
        dim3 grid(batch * num_heads, num_q_blocks);
        dim3 block(Br);  // one thread per query row in the tile

        // Shared memory: Q_smem + K_smem + V_smem (all half)
        size_t smem_bytes = (Br * D + Bc * D + Bc * D) * sizeof(half);

        flash_attention_forward_kernel<Br, Bc, D>
            <<<grid, block, smem_bytes, stream>>>(
                Q, K, V, O, L, seq_len, scale, is_causal
            );

    } else if (head_dim == 128) {
        // Larger head dim: use smaller Br to fit shared memory
        constexpr int Br = 32;
        constexpr int Bc = 64;
        constexpr int D  = 128;

        int num_q_blocks = (seq_len + Br - 1) / Br;
        dim3 grid(batch * num_heads, num_q_blocks);
        dim3 block(Br);

        size_t smem_bytes = (Br * D + Bc * D + Bc * D) * sizeof(half);

        flash_attention_forward_kernel<Br, Bc, D>
            <<<grid, block, smem_bytes, stream>>>(
                Q, K, V, O, L, seq_len, scale, is_causal
            );

    } else {
        fprintf(stderr,
                "flash_attention_forward: unsupported head_dim=%d "
                "(only 64 and 128 supported)\n", head_dim);
        return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attention_forward launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

}  // namespace flashkernel
