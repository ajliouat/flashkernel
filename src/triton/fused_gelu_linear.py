"""
FlashKernel — Triton Fused GeLU+Linear (v1.0.4)

Computes: Y = GeLU(X @ W^T + bias) in a single fused kernel.

Key idea (same as CUDA v1.0.4):
  Unfused: 2 kernels, 2 HBM round-trips
    temp = X @ W^T + bias   (write temp to HBM)
    Y = GeLU(temp)           (read temp from HBM, write Y)
  Fused: 1 kernel, 1 HBM write
    matmul + bias + GeLU all in SRAM/registers -> single HBM write

GeLU variants:
  exact:  x * 0.5 * (1 + erf(x / sqrt(2)))
  tanh:   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Architecture target: any GPU with Triton support (SM >= 7.0)
"""

import math

import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
# AUTOTUNE CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

_FUSED_GELU_CONFIGS = [
    # Match CUDA TILE_M=64, TILE_N=64, TILE_K=32
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=8, num_stages=2,
    ),
    # Larger tiles for bigger matrices
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=8, num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4, num_stages=3,
    ),
    # Smaller tiles for small matrices
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
        num_warps=4, num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
        num_warps=4, num_stages=2,
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# TRITON KERNEL
# ═════════════════════════════════════════════════════════════════════════════

@triton.autotune(configs=_FUSED_GELU_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fused_gelu_linear_kernel(
    # Pointers
    X_ptr,     # [M, K]
    W_ptr,     # [N, K]
    bias_ptr,  # [N] or nullptr
    Y_ptr,     # [M, N]
    # Dimensions
    M,
    N,
    K,
    # Strides (X)
    stride_xm,
    stride_xk,
    # Strides (W)
    stride_wn,
    stride_wk,
    # Strides (Y)
    stride_ym,
    stride_yn,
    # Flags
    HAS_BIAS: tl.constexpr,
    USE_TANH_APPROX: tl.constexpr,
    # Tile sizes (autotuned)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused GeLU+Linear kernel.

    Each program instance computes a BLOCK_M x BLOCK_N tile of the output Y.
    """
    # ── Program IDs ──────────────────────────────────────────────────────────
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # ── Offsets ──────────────────────────────────────────────────────────────
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    # ── Accumulator (fp32) ───────────────────────────────────────────────────
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── Iterate over K dimension ─────────────────────────────────────────────
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k  # [BLOCK_K]

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile: [BLOCK_N, BLOCK_K] -> we need W^T contribution
        # W is [N, K], we load W[offs_n, k_offs] and transpose for matmul
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate: acc += X_tile @ W_tile^T  -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(x, tl.trans(w))

    # ── Add bias (in-register, no HBM round-trip) ───────────────────────────
    if HAS_BIAS:
        bias_ptrs = bias_ptr + offs_n
        bias_mask = offs_n < N
        b = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
        acc += b[None, :]

    # ── Apply GeLU (in-register, no HBM round-trip) ─────────────────────────
    if USE_TANH_APPROX:
        # GeLU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        kSqrt2OverPi: tl.constexpr = 0.7978845608028654
        inner = kSqrt2OverPi * (acc + 0.044715 * acc * acc * acc)
        acc = 0.5 * acc * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    else:
        # GeLU_exact(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
        kInvSqrt2: tl.constexpr = 0.7071067811865476
        acc = acc * 0.5 * (1.0 + tl.extra.cuda.libdevice.erf(acc * kInvSqrt2))

    # ── Store result (single HBM write — the fusion payoff) ─────────────────
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)


# ═════════════════════════════════════════════════════════════════════════════
# PYTHON WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

def triton_fused_gelu_linear(
    X: torch.Tensor,
    W: torch.Tensor,
    bias: torch.Tensor | None = None,
    use_tanh_approx: bool = False,
) -> torch.Tensor:
    """
    Fused GeLU+Linear: Y = GeLU(X @ W^T + bias)

    All matmul + bias + GeLU computation happens in a single kernel pass.
    The intermediate matmul result is never written to HBM.

    Args:
        X: [M, K] fp16 CUDA tensor — input activations
        W: [N, K] fp16 CUDA tensor — weight matrix
        bias: [N] fp16 CUDA tensor or None — optional bias
        use_tanh_approx: If True, use fast tanh GeLU approximation;
                         otherwise use exact erf GeLU.

    Returns:
        Y: [M, N] fp16 — GeLU(X @ W^T + bias)
    """
    # ── Input validation ─────────────────────────────────────────────────────
    if not X.is_cuda:
        raise RuntimeError("X must be on CUDA device")
    if not W.is_cuda:
        raise RuntimeError("W must be on CUDA device")
    if X.dtype != torch.float16:
        raise RuntimeError(f"X must be float16, got {X.dtype}")
    if W.dtype != torch.float16:
        raise RuntimeError(f"W must be float16, got {W.dtype}")
    if X.dim() != 2:
        raise RuntimeError(f"X must be 2-D [M, K], got {X.dim()}-D")
    if W.dim() != 2:
        raise RuntimeError(f"W must be 2-D [N, K], got {W.dim()}-D")
    if X.shape[1] != W.shape[1]:
        raise RuntimeError(
            f"X columns ({X.shape[1]}) must match W columns ({W.shape[1]})"
        )
    if bias is not None:
        if not bias.is_cuda:
            raise RuntimeError("bias must be on CUDA device")
        if bias.dtype != torch.float16:
            raise RuntimeError(f"bias must be float16, got {bias.dtype}")
        if bias.dim() != 1 or bias.shape[0] != W.shape[0]:
            raise RuntimeError(
                f"bias must be 1-D of size N={W.shape[0]}, got shape {bias.shape}"
            )

    # Make contiguous
    X = X.contiguous()
    W = W.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    M, K = X.shape
    N = W.shape[0]

    # Allocate output
    Y = torch.empty((M, N), dtype=torch.float16, device=X.device)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _fused_gelu_linear_kernel[grid](
        X, W,
        bias if bias is not None else X,  # dummy ptr when no bias
        Y,
        M, N, K,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        Y.stride(0), Y.stride(1),
        HAS_BIAS=(bias is not None),
        USE_TANH_APPROX=use_tanh_approx,
    )

    return Y
