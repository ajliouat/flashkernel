"""
FlashKernel — Triton FlashAttention Forward Pass (v1.0.3)

Same algorithm as the CUDA kernel (v1.0.2) implemented in Triton:
  - Tiled Q/K/V with online softmax (Dao et al., 2022)
  - No N×N attention matrix materialization
  - Causal masking with early KV-loop termination
  - Autotuned over BLOCK_M, BLOCK_N, num_warps

Tile configs include CUDA-matching sizes for fair comparison:
  head_dim=64:  CUDA uses Br=64, Bc=64
  head_dim=128: CUDA uses Br=32, Bc=64

Architecture target: any GPU with Triton support (SM >= 7.0)
"""

import math

import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
# AUTOTUNE CONFIGURATIONS
# ═════════════════════════════════════════════════════════════════════════════

# Include configs matching CUDA v1.0.2 tile sizes for fair head-to-head
# comparison, plus additional configs that Triton may prefer.

_AUTOTUNE_CONFIGS = [
    # --- Matching CUDA v1.0.2 tile sizes ---
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    # --- Wider exploration ---
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}, num_warps=4, num_stages=2),
]


# ═════════════════════════════════════════════════════════════════════════════
# TRITON KERNEL
# ═════════════════════════════════════════════════════════════════════════════

@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N", "D_MODEL"])
@triton.jit
def _flash_attn_fwd_kernel(
    # Tensor pointers
    Q, K, V, Out, LSE,
    # Strides — Q/K/V/Out: [B, H, N, D]
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    # Strides — LSE: [B, H, N]
    stride_lz, stride_lh, stride_lm,
    # Scalar arguments
    H,          # num_heads
    N,          # seq_len
    scale,      # 1 / sqrt(d)
    # Compile-time constants
    IS_CAUSAL: tl.constexpr,
    D_MODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton FlashAttention forward kernel.

    Each program handles one (batch, head, q_block) = BLOCK_M query rows.
    Inner loop iterates over KV blocks of BLOCK_N rows.
    Online softmax: running max (m), sum (l), and output (O) in fp32.

    Grid: (B * H, cdiv(N, BLOCK_M))
    """
    # ---- grid mapping ----
    pid_bh = tl.program_id(0)          # batch * head (flattened)
    pid_m  = tl.program_id(1)          # Q-block index

    # Separate batch and head
    b = pid_bh // H
    h = pid_bh % H

    # Base pointer offsets for this (b, h)
    q_off = b * stride_qz + h * stride_qh
    k_off = b * stride_kz + h * stride_kh
    v_off = b * stride_vz + h * stride_vh
    o_off = b * stride_oz + h * stride_oh
    l_off = b * stride_lz + h * stride_lh

    # ---- Q block coordinates ----
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)     # [BLOCK_M]
    offs_d = tl.arange(0, D_MODEL)                # [D_MODEL]

    # Load Q block: [BLOCK_M, D_MODEL]  (fp16 from HBM)
    q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

    # ---- accumulators (fp32 for numerical stability) ----
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # running row max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                  # running exp sum
    o_i = tl.zeros([BLOCK_M, D_MODEL], dtype=tl.float32)         # output accumulator

    # ---- KV iteration bounds ----
    # Causal: only attend to positions <= last Q position in this block
    if IS_CAUSAL:
        kv_end = (pid_m + 1) * BLOCK_M
    else:
        kv_end = N

    # ---- main loop over KV blocks ----
    for n_start in range(0, kv_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)       # [BLOCK_N]

        # Load K block: [BLOCK_N, D_MODEL]
        k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)

        # Load V block: [BLOCK_N, D_MODEL]
        v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

        # Compute scores: S = Q @ K^T * scale  →  [BLOCK_M, BLOCK_N]
        s = tl.dot(q, tl.trans(k)) * scale

        # Boundary mask: positions beyond seq_len → -inf
        s = tl.where(offs_n[None, :] < N, s, float("-inf"))

        # Causal mask: future positions → -inf
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        # ---- online softmax update ----
        # New row-wise max
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # Rescale factor for previous accumulations
        alpha = tl.exp(m_i - m_new)
        # Softmax numerator
        p = tl.exp(s - m_new[:, None])

        # Update running sum
        l_i = alpha * l_i + tl.sum(p, axis=1)
        # Update output: rescale old + accumulate new
        # Cast p to fp16 for tl.dot (Tensor Core dispatch)
        o_i = alpha[:, None] * o_i + tl.dot(p.to(q.dtype), v)
        # Update running max
        m_i = m_new

    # ---- normalize output ----
    o_i = o_i / l_i[:, None]

    # ---- log-sum-exp (for backward pass) ----
    lse = m_i + tl.log(l_i)

    # ---- store O: [BLOCK_M, D_MODEL] → HBM (fp16) ----
    o_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, o_i.to(q.dtype), mask=offs_m[:, None] < N)

    # ---- store LSE: [BLOCK_M] → HBM (fp32) ----
    l_ptrs = LSE + l_off + offs_m * stride_lm
    tl.store(l_ptrs, lse, mask=offs_m < N)


# ═════════════════════════════════════════════════════════════════════════════
# PYTHON WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

def triton_flash_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton FlashAttention forward — same algorithm as CUDA v1.0.2.

    Tiled Q/K/V with online softmax. No N×N attention matrix materialized
    in HBM. Autotuned over tile sizes and warp counts.

    Args:
        Q: [B, H, N, D] fp16 CUDA tensor
        K: [B, H, N, D] fp16 CUDA tensor
        V: [B, H, N, D] fp16 CUDA tensor
        scale: Softmax scale (default: 1/sqrt(D))
        is_causal: Apply causal attention mask

    Returns:
        O: [B, H, N, D] fp16 — attention output
        L: [B, H, N]    fp32 — log-sum-exp (for backward pass)

    Supported head_dim: 64, 128. Tile sizes selected by Triton autotune.
    """
    # ---- input validation ----
    if not Q.is_cuda or not K.is_cuda or not V.is_cuda:
        raise RuntimeError("All inputs must be on CUDA device")
    if Q.dtype != torch.float16:
        raise RuntimeError(f"Expected float16, got {Q.dtype}")
    if Q.dim() != 4:
        raise RuntimeError(f"Expected 4-D [B, H, N, D], got {Q.dim()}-D")
    if Q.shape != K.shape or Q.shape != V.shape:
        raise RuntimeError(
            f"Shape mismatch: Q={Q.shape}, K={K.shape}, V={V.shape}"
        )

    B, H, N, D = Q.shape
    if D not in (64, 128):
        raise RuntimeError(f"Unsupported head_dim={D} (only 64, 128)")

    # Ensure contiguous layout
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # ---- allocate outputs ----
    O = torch.empty_like(Q)
    L = torch.empty(B, H, N, device=Q.device, dtype=torch.float32)

    # ---- launch kernel ----
    # Grid: (B*H, ceil(N/BLOCK_M)) — BLOCK_M from autotune
    grid = lambda META: (B * H, triton.cdiv(N, META["BLOCK_M"]))

    _flash_attn_fwd_kernel[grid](
        Q, K, V, O, L,
        # Q strides
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        # K strides
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        # V strides
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        # O strides
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        # L strides
        L.stride(0), L.stride(1), L.stride(2),
        # Scalar args
        H=H,
        N=N,
        scale=scale,
        # Compile-time constants
        IS_CAUSAL=is_causal,
        D_MODEL=D,
    )

    return O, L
