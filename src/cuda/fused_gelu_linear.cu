/**
 * FlashKernel — Fused GeLU+Linear CUDA kernel (v1.0.4)
 *
 * Computes: Y = GeLU(X @ W^T + bias)
 *
 * Strategy:
 *   - Tiled GEMM in shared memory (TILE_M x TILE_N accumulation, tiles of K)
 *   - Bias addition in registers after matmul accumulation
 *   - GeLU applied in-register (fp32) before single HBM write
 *   - This eliminates one full HBM round-trip vs unfused (2 kernels):
 *       Unfused: kernel1 writes temp to HBM, kernel2 reads temp back
 *       Fused:   matmul + GeLU in registers, single write to HBM
 *
 * Tile sizes (SM 7.5 / T4, 48 KB shared memory):
 *   TILE_M=64, TILE_N=64, TILE_K=32:
 *     smem_X: 64 * 32 * 2 = 4 KB
 *     smem_W: 64 * 32 * 2 = 4 KB
 *     Total: 8 KB  (well within limit, allows high occupancy)
 *
 * Memory layout: row-major throughout
 * Accumulation in fp32 for numerical stability.
 *
 * Architecture target: SM 7.5 (Turing / T4)
 */

#include "fused_gelu_linear.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

namespace flashkernel {

// ─── Tile dimensions ────────────────────────────────────────────────────────
// Each thread-block computes a TILE_M x TILE_N output tile.
// We iterate over K in chunks of TILE_K.

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 32;

// Thread-block: 16 x 16 = 256 threads
// Each thread computes a 4x4 sub-tile of the TILE_M x TILE_N output
static constexpr int THREADS_X = 16;
static constexpr int THREADS_Y = 16;
static constexpr int THREAD_TILE_M = TILE_M / THREADS_Y;  // 4
static constexpr int THREAD_TILE_N = TILE_N / THREADS_X;  // 4

// ─── GeLU implementations ───────────────────────────────────────────────────

__device__ __forceinline__ float gelu_exact(float x) {
    // GeLU_exact(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));  // 1/sqrt(2)
}

__device__ __forceinline__ float gelu_tanh(float x) {
    // GeLU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float kSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/pi)
    float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// ═════════════════════════════════════════════════════════════════════════════
// MAIN KERNEL
// ═════════════════════════════════════════════════════════════════════════════

template <bool UseTanhApprox, bool HasBias>
__global__ void fused_gelu_linear_kernel(
    const half* __restrict__ X,       // [M, K]
    const half* __restrict__ W,       // [N, K]
    const half* __restrict__ bias,    // [N] or nullptr
    half* __restrict__ Y,             // [M, N]
    int M,
    int N,
    int K
) {
    // ── Block-level output tile coordinates ─────────────────────────────────
    const int bm = blockIdx.y * TILE_M;  // row offset in Y
    const int bn = blockIdx.x * TILE_N;  // col offset in Y

    // ── Thread-level coordinates within the block ───────────────────────────
    const int tx = threadIdx.x;  // 0..15 (col direction)
    const int ty = threadIdx.y;  // 0..15 (row direction)
    const int tid = ty * THREADS_X + tx;

    // ── Shared memory for X and W tiles ─────────────────────────────────────
    __shared__ half smem_X[TILE_M][TILE_K];   // 64 x 32
    __shared__ half smem_W[TILE_N][TILE_K];   // 64 x 32

    // ── Accumulator registers (fp32) — each thread holds 4x4 outputs ────────
    float acc[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // ── Iterate over K tiles ────────────────────────────────────────────────
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    for (int kt = 0; kt < num_k_tiles; kt++) {
        const int k_offset = kt * TILE_K;

        // -- Load X tile: TILE_M rows × TILE_K cols --
        // 256 threads load 64*32 = 2048 elements → 8 elements per thread
        {
            const int elems_per_load = (TILE_M * TILE_K) / (THREADS_X * THREADS_Y);
            #pragma unroll
            for (int e = 0; e < elems_per_load; e++) {
                int flat = tid * elems_per_load + e;
                int r = flat / TILE_K;
                int c = flat % TILE_K;
                int global_r = bm + r;
                int global_c = k_offset + c;
                if (global_r < M && global_c < K) {
                    smem_X[r][c] = X[global_r * K + global_c];
                } else {
                    smem_X[r][c] = __float2half(0.0f);
                }
            }
        }

        // -- Load W tile: TILE_N rows × TILE_K cols --
        // W is [N, K] row-major. We load W[bn..bn+TILE_N, k_offset..k_offset+TILE_K]
        {
            const int elems_per_load = (TILE_N * TILE_K) / (THREADS_X * THREADS_Y);
            #pragma unroll
            for (int e = 0; e < elems_per_load; e++) {
                int flat = tid * elems_per_load + e;
                int r = flat / TILE_K;
                int c = flat % TILE_K;
                int global_r = bn + r;
                int global_c = k_offset + c;
                if (global_r < N && global_c < K) {
                    smem_W[r][c] = W[global_r * K + global_c];
                } else {
                    smem_W[r][c] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // -- Accumulate: each thread computes its 4x4 sub-tile --
        // acc[i][j] += sum_k(X[bm + ty*4 + i, k] * W[bn + tx*4 + j, k])
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            // Load X values for this thread's rows
            float x_val[THREAD_TILE_M];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                x_val[i] = __half2float(smem_X[ty * THREAD_TILE_M + i][kk]);
            }
            // Load W values for this thread's columns
            float w_val[THREAD_TILE_N];
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {
                w_val[j] = __half2float(smem_W[tx * THREAD_TILE_N + j][kk]);
            }
            // Outer product accumulate
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    acc[i][j] += x_val[i] * w_val[j];
                }
            }
        }

        __syncthreads();
    }

    // ── Apply bias + GeLU and store ─────────────────────────────────────────
    // This is the key fusion: bias add + GeLU happen in registers,
    // avoiding the intermediate HBM round-trip.

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        int row = bm + ty * THREAD_TILE_M + i;
        if (row >= M) continue;

        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int col = bn + tx * THREAD_TILE_N + j;
            if (col >= N) continue;

            float val = acc[i][j];

            // Add bias (in register, no HBM read for intermediate)
            if constexpr (HasBias) {
                val += __half2float(bias[col]);
            }

            // Apply GeLU (in register)
            if constexpr (UseTanhApprox) {
                val = gelu_tanh(val);
            } else {
                val = gelu_exact(val);
            }

            // Single HBM write (fused result)
            Y[row * N + col] = __float2half(val);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// HOST LAUNCHER
// ═════════════════════════════════════════════════════════════════════════════

void fused_gelu_linear(
    const half* X,
    const half* W,
    const half* bias,
    half* Y,
    int M,
    int N,
    int K,
    bool use_tanh_approx,
    cudaStream_t stream
) {
    dim3 block(THREADS_X, THREADS_Y);  // 16 x 16 = 256 threads
    dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M
    );

    // Template dispatch: 4 variants (tanh/exact) x (bias/no-bias)
    if (use_tanh_approx) {
        if (bias != nullptr) {
            fused_gelu_linear_kernel<true, true><<<grid, block, 0, stream>>>(
                X, W, bias, Y, M, N, K);
        } else {
            fused_gelu_linear_kernel<true, false><<<grid, block, 0, stream>>>(
                X, W, bias, Y, M, N, K);
        }
    } else {
        if (bias != nullptr) {
            fused_gelu_linear_kernel<false, true><<<grid, block, 0, stream>>>(
                X, W, bias, Y, M, N, K);
        } else {
            fused_gelu_linear_kernel<false, false><<<grid, block, 0, stream>>>(
                X, W, bias, Y, M, N, K);
        }
    }

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "fused_gelu_linear kernel launch error: %s\n",
                cudaGetErrorString(err));
    }
}

}  // namespace flashkernel
