/**
 * FlashKernel — Parallel Reduction Kernels (v1.0.1)
 *
 * Three-level reduction strategy:
 *   1. Warp-level: __shfl_down_sync (no shared memory, no sync)
 *   2. Block-level: Shared memory to combine warp results
 *   3. Grid-level:  Multi-block with atomic or two-pass reduce
 *
 * Design choices:
 *   - fp16 inputs are promoted to fp32 for accumulation (numerical stability)
 *   - Row-wise reduction uses one block per row for small cols, multiple for large
 *   - Full reduction uses two-pass approach (no atomics, deterministic)
 *
 * Architecture target: SM 7.5 (Turing / T4)
 *   - 32 threads per warp
 *   - 48 KB shared memory per SM (configurable up to 64 KB)
 *   - 64 warps per SM max
 */

#include "reduce.cuh"
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdio>

namespace flashkernel {

// ─── Constants ──────────────────────────────────────────────────────────────

static constexpr int WARP_SIZE = 32;
static constexpr int BLOCK_SIZE = 256;  // 8 warps per block
static constexpr unsigned FULL_MASK = 0xffffffff;

// ─── Warp-Level Primitives ──────────────────────────────────────────────────

/**
 * Warp reduction via shuffle-down.
 * Every thread in the warp must participate (full mask).
 * After this, lane 0 holds the reduced value.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

// ─── Block-Level Reduction ──────────────────────────────────────────────────

/**
 * Block-level sum: each warp reduces internally, then lane-0 threads
 * write to shared memory. The first warp then reduces those partial sums.
 *
 * Returns the block sum in thread 0.
 */
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[WARP_SIZE];  // one slot per warp (max 32 warps)

    int lane   = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Step 1: intra-warp reduce
    val = warp_reduce_sum(val);

    // Step 2: lane-0 of each warp stores to shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Step 3: first warp reduces the partial sums
    // Only participate if this thread's index < number of warps
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

__device__ float block_reduce_max(float val) {
    __shared__ float shared[WARP_SIZE];

    int lane    = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

// ═════════════════════════════════════════════════════════════════════════════
// FULL REDUCTION KERNELS (entire array → single scalar)
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Pass 1: Each block reduces a chunk of the input into one partial sum.
 * Grid-stride loop so each block can handle more elements than blockDim.x.
 */
__global__ void reduce_sum_f32_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    const int n
) {
    float sum = 0.0f;

    // Grid-stride loop: each thread accumulates multiple elements
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        sum += input[i];
    }

    // Block-level reduce
    sum = block_reduce_sum(sum);

    // Thread 0 of each block writes its partial sum
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

/**
 * fp16 input → fp32 accumulation → fp16 output.
 */
__global__ void reduce_sum_f16_kernel(
    const half* __restrict__ input,
    float*      __restrict__ output,
    const int n
) {
    float sum = 0.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        sum += __half2float(input[i]);
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

/**
 * Pass 2: Reduce the partial sums from pass 1 into a single value.
 * Single block, exactly num_partials threads (capped at BLOCK_SIZE).
 */
__global__ void reduce_finalize_f32_kernel(
    const float* __restrict__ partials,
    float*       __restrict__ output,
    const int num_partials
) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
        sum += partials[i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[0] = sum;
    }
}

__global__ void reduce_finalize_f16_kernel(
    const float* __restrict__ partials,
    half*        __restrict__ output,
    const int num_partials
) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
        sum += partials[i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        output[0] = __float2half(sum);
    }
}

// ─── Max kernels ────────────────────────────────────────────────────────────

__global__ void reduce_max_f32_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    const int n
) {
    float mx = -FLT_MAX;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        mx = fmaxf(mx, input[i]);
    }

    mx = block_reduce_max(mx);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = mx;
    }
}

__global__ void reduce_max_finalize_f32_kernel(
    const float* __restrict__ partials,
    float*       __restrict__ output,
    const int num_partials
) {
    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
        mx = fmaxf(mx, partials[i]);
    }
    mx = block_reduce_max(mx);
    if (threadIdx.x == 0) {
        output[0] = mx;
    }
}

__global__ void reduce_max_f16_kernel(
    const half* __restrict__ input,
    float*      __restrict__ output,
    const int n
) {
    float mx = -FLT_MAX;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x) {
        mx = fmaxf(mx, __half2float(input[i]));
    }

    mx = block_reduce_max(mx);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = mx;
    }
}

__global__ void reduce_max_finalize_f16_kernel(
    const float* __restrict__ partials,
    half*        __restrict__ output,
    const int num_partials
) {
    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
        mx = fmaxf(mx, partials[i]);
    }
    mx = block_reduce_max(mx);
    if (threadIdx.x == 0) {
        output[0] = __float2half(mx);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ROW-WISE REDUCTION KERNELS (matrix → vector: reduce last dim)
// ═════════════════════════════════════════════════════════════════════════════

/**
 * One block per row. Each block reduces `cols` elements.
 * Grid-stride within the row if cols > blockDim.x.
 */
__global__ void reduce_sum_rows_f32_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    const int rows,
    const int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = input + row * cols;
    float sum = 0.0f;

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        sum += row_ptr[c];
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        output[row] = sum;
    }
}

__global__ void reduce_sum_rows_f16_kernel(
    const half* __restrict__ input,
    half*       __restrict__ output,
    const int rows,
    const int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const half* row_ptr = input + row * cols;
    float sum = 0.0f;

    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        sum += __half2float(row_ptr[c]);
    }

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0) {
        output[row] = __float2half(sum);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// HOST WRAPPERS
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Compute grid size for full reduction pass 1.
 * Cap at 1024 blocks to keep pass-2 cheap.
 */
static int compute_grid_size(int n, int block_size) {
    int blocks = (n + block_size - 1) / block_size;
    return min(blocks, 1024);
}

// ─── Full Sum ───────────────────────────────────────────────────────────────

void reduce_sum_f32(const float* input, float* output, int n,
                    cudaStream_t stream) {
    if (n <= 0) return;

    // Single-block path: no temporary buffer needed
    if (n <= BLOCK_SIZE) {
        reduce_sum_f32_kernel<<<1, BLOCK_SIZE, 0, stream>>>(input, output, n);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "reduce_sum_f32 launch failed: %s\n",
                    cudaGetErrorString(err));
        }
        return;
    }

    // Two-pass reduction
    int grid = compute_grid_size(n, BLOCK_SIZE);

    // Allocate temporary buffer for partial sums
    float* partials = nullptr;
    cudaMallocAsync(&partials, grid * sizeof(float), stream);

    // Pass 1: each block produces one partial sum
    reduce_sum_f32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        input, partials, n);

    // Pass 2: single block reduces partial sums
    int finalize_threads = min(grid, BLOCK_SIZE);
    // Round up to next multiple of WARP_SIZE
    finalize_threads = ((finalize_threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    reduce_finalize_f32_kernel<<<1, finalize_threads, 0, stream>>>(
        partials, output, grid);

    cudaFreeAsync(partials, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "reduce_sum_f32 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

void reduce_sum_f16(const half* input, half* output, int n,
                    cudaStream_t stream) {
    if (n <= 0) return;

    if (n <= BLOCK_SIZE) {
        // Single block: still need temporary fp32 buffer for finalization
        float* partials = nullptr;
        cudaMallocAsync(&partials, sizeof(float), stream);
        reduce_sum_f16_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            input, partials, n);
        reduce_finalize_f16_kernel<<<1, WARP_SIZE, 0, stream>>>(
            partials, output, 1);
        cudaFreeAsync(partials, stream);
    } else {
        int grid = compute_grid_size(n, BLOCK_SIZE);
        float* partials = nullptr;
        cudaMallocAsync(&partials, grid * sizeof(float), stream);

        reduce_sum_f16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            input, partials, n);

        int finalize_threads = ((min(grid, BLOCK_SIZE) + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        reduce_finalize_f16_kernel<<<1, finalize_threads, 0, stream>>>(
            partials, output, grid);

        cudaFreeAsync(partials, stream);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "reduce_sum_f16 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

// ─── Full Max ───────────────────────────────────────────────────────────────

void reduce_max_f32(const float* input, float* output, int n,
                    cudaStream_t stream) {
    if (n <= 0) return;

    if (n <= BLOCK_SIZE) {
        reduce_max_f32_kernel<<<1, BLOCK_SIZE, 0, stream>>>(input, output, n);
    } else {
        int grid = compute_grid_size(n, BLOCK_SIZE);
        float* partials = nullptr;
        cudaMallocAsync(&partials, grid * sizeof(float), stream);

        reduce_max_f32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            input, partials, n);

        int finalize_threads = ((min(grid, BLOCK_SIZE) + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        reduce_max_finalize_f32_kernel<<<1, finalize_threads, 0, stream>>>(
            partials, output, grid);

        cudaFreeAsync(partials, stream);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "reduce_max_f32 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

void reduce_max_f16(const half* input, half* output, int n,
                    cudaStream_t stream) {
    if (n <= 0) return;

    if (n <= BLOCK_SIZE) {
        float* partials = nullptr;
        cudaMallocAsync(&partials, sizeof(float), stream);
        reduce_max_f16_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            input, partials, n);
        reduce_max_finalize_f16_kernel<<<1, WARP_SIZE, 0, stream>>>(
            partials, output, 1);
        cudaFreeAsync(partials, stream);
    } else {
        int grid = compute_grid_size(n, BLOCK_SIZE);
        float* partials = nullptr;
        cudaMallocAsync(&partials, grid * sizeof(float), stream);

        reduce_max_f16_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            input, partials, n);

        int finalize_threads = ((min(grid, BLOCK_SIZE) + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        reduce_max_finalize_f16_kernel<<<1, finalize_threads, 0, stream>>>(
            partials, output, grid);

        cudaFreeAsync(partials, stream);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "reduce_max_f16 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

// ─── Row-wise Sum ───────────────────────────────────────────────────────────

void reduce_sum_rows_f32(const float* input, float* output,
                         int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;

    // One block per row, BLOCK_SIZE threads per block
    int threads = min(cols, BLOCK_SIZE);
    // Round up to warp size for clean warp reduction
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    threads = min(threads, BLOCK_SIZE);

    reduce_sum_rows_f32_kernel<<<rows, threads, 0, stream>>>(
        input, output, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "reduce_sum_rows_f32 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

void reduce_sum_rows_f16(const half* input, half* output,
                         int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;

    int threads = min(cols, BLOCK_SIZE);
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    threads = min(threads, BLOCK_SIZE);

    reduce_sum_rows_f16_kernel<<<rows, threads, 0, stream>>>(
        input, output, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "reduce_sum_rows_f16 launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

}  // namespace flashkernel
