/**
 * FlashKernel — Parallel reduction header (v1.0.1)
 *
 * Exposes warp-shuffle + shared-memory tree reduction for fp32 and fp16.
 * Supports arbitrary-length 1-D reductions (row-wise sum over the last dim).
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flashkernel {

// ─── Reduction Operations ───────────────────────────────────────────────────

enum class ReduceOp {
    SUM,
    MAX,
    MIN,
};

// ─── Kernel Wrappers ────────────────────────────────────────────────────────

/**
 * reduce_sum_f32 — Full reduction of n elements to a single scalar.
 *
 * @param input   Device pointer, n elements (fp32)
 * @param output  Device pointer, 1 element (fp32)
 * @param n       Number of elements
 * @param stream  CUDA stream
 */
void reduce_sum_f32(const float* input, float* output, int n,
                    cudaStream_t stream = 0);

/**
 * reduce_sum_f16 — Full reduction of n elements to a single scalar.
 *
 * Accumulation is done in fp32 internally for numerical stability,
 * then the final result is stored in fp16.
 *
 * @param input   Device pointer, n elements (fp16)
 * @param output  Device pointer, 1 element (fp16)
 * @param n       Number of elements
 * @param stream  CUDA stream
 */
void reduce_sum_f16(const half* input, half* output, int n,
                    cudaStream_t stream = 0);

/**
 * reduce_sum_rows_f32 — Per-row sum reduction over the last dimension.
 *
 * Given a (rows x cols) matrix, produces (rows,) output.
 *
 * @param input   Device pointer, rows * cols elements (fp32)
 * @param output  Device pointer, rows elements (fp32)
 * @param rows    Number of rows
 * @param cols    Number of columns (reduction dimension)
 * @param stream  CUDA stream
 */
void reduce_sum_rows_f32(const float* input, float* output,
                         int rows, int cols, cudaStream_t stream = 0);

/**
 * reduce_sum_rows_f16 — Per-row sum reduction over the last dimension.
 *
 * Accumulation in fp32 internally.
 */
void reduce_sum_rows_f16(const half* input, half* output,
                         int rows, int cols, cudaStream_t stream = 0);

/**
 * reduce_max_f32 — Full reduction: max of n elements.
 */
void reduce_max_f32(const float* input, float* output, int n,
                    cudaStream_t stream = 0);

/**
 * reduce_max_f16 — Full reduction: max of n elements.
 */
void reduce_max_f16(const half* input, half* output, int n,
                    cudaStream_t stream = 0);

}  // namespace flashkernel
