/**
 * FlashKernel — Fused GeLU+Linear header (v1.0.4)
 *
 * Fuses y = GeLU(x @ W^T + b) into a single kernel:
 *   1. Tiled matmul in shared memory
 *   2. Add bias in registers
 *   3. Apply GeLU activation in registers (no extra HBM round-trip)
 *   4. Write result to HBM once
 *
 * GeLU variants:
 *   - Exact:  x * 0.5 * (1 + erf(x / sqrt(2)))
 *   - Tanh:   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * Architecture target: SM 7.5 (Turing / T4)
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace flashkernel {

/**
 * fused_gelu_linear — Fused linear projection + GeLU activation (CUDA)
 *
 * Computes: Y = GeLU(X @ W^T + bias)
 *
 * @param X         [M, K] fp16, row-major — input activations
 * @param W         [N, K] fp16, row-major — weight matrix (transposed layout)
 * @param bias      [N]    fp16 — bias vector (may be nullptr for no bias)
 * @param Y         [M, N] fp16, output
 * @param M         Number of rows in X (batch dimension)
 * @param N         Output features (columns of Y)
 * @param K         Inner dimension (columns of X, columns of W)
 * @param use_tanh_approx  If true, use fast tanh GeLU; else exact erf GeLU
 * @param stream    CUDA stream
 */
void fused_gelu_linear(
    const half* X,
    const half* W,
    const half* bias,
    half* Y,
    int M,
    int N,
    int K,
    bool use_tanh_approx,
    cudaStream_t stream = 0
);

}  // namespace flashkernel
