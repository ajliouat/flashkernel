"""
FlashKernel — Triton parallel reduction kernels (v1.0.1)

Triton equivalents of the CUDA reduction kernels in src/cuda/reduce.cu.
These serve as:
  1. Reference implementations for correctness comparison
  2. Demonstration of Triton vs CUDA productivity tradeoff
  3. Building blocks for later fused kernels (softmax, layernorm)

Architecture target: any GPU with Triton support (SM >= 7.0)
"""

import torch
import triton
import triton.language as tl


# ═════════════════════════════════════════════════════════════════════════════
# FULL REDUCTION (entire tensor → scalar)
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def _reduce_sum_kernel(
    input_ptr,
    partial_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program instance reduces a chunk of BLOCK_SIZE elements.
    Grid-stride: program i handles elements [i*BLOCK_SIZE, (i+1)*BLOCK_SIZE).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    partial_sum = tl.sum(x, axis=0)
    tl.store(partial_ptr + pid, partial_sum)


@triton.jit
def _reduce_finalize_kernel(
    partial_ptr,
    output_ptr,
    num_partials,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-program kernel that reduces partial sums to a scalar."""
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_partials
    x = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.store(output_ptr, result)


def triton_reduce_sum(input: torch.Tensor) -> torch.Tensor:
    """
    Full sum reduction using Triton.

    Args:
        input: 1-D or N-D CUDA tensor (fp32 or fp16). Flattened internally.

    Returns:
        Scalar tensor with the sum.
    """
    assert input.is_cuda, "Input must be on CUDA device"
    flat = input.contiguous().view(-1)
    n = flat.numel()

    # For fp16, accumulate in fp32 then cast back
    compute_dtype = torch.float32
    if flat.dtype == torch.float16:
        flat = flat.to(torch.float32)

    BLOCK_SIZE = 1024
    grid_size = (n + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Cap grid at a triton-friendly value
    grid_size = min(grid_size, 65535)

    partials = torch.empty(grid_size, device=flat.device, dtype=compute_dtype)

    _reduce_sum_kernel[(grid_size,)](
        flat, partials, n, BLOCK_SIZE=BLOCK_SIZE,
    )

    # Pass 2: reduce partials
    # Use next power of 2 for BLOCK_SIZE
    finalize_block = triton.next_power_of_2(grid_size)
    finalize_block = max(finalize_block, 32)  # minimum block
    output = torch.empty(1, device=flat.device, dtype=compute_dtype)
    _reduce_finalize_kernel[(1,)](
        partials, output, grid_size, BLOCK_SIZE=finalize_block,
    )

    result = output.squeeze(0)
    if input.dtype == torch.float16:
        result = result.to(torch.float16)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# MAX REDUCTION
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def _reduce_max_kernel(
    input_ptr,
    partial_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))
    partial_max = tl.max(x, axis=0)
    tl.store(partial_ptr + pid, partial_max)


@triton.jit
def _reduce_max_finalize_kernel(
    partial_ptr,
    output_ptr,
    num_partials,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_partials
    x = tl.load(partial_ptr + offsets, mask=mask, other=float("-inf"))
    result = tl.max(x, axis=0)
    tl.store(output_ptr, result)


def triton_reduce_max(input: torch.Tensor) -> torch.Tensor:
    """
    Full max reduction using Triton.

    Args:
        input: 1-D or N-D CUDA tensor.

    Returns:
        Scalar tensor with the max value.
    """
    assert input.is_cuda, "Input must be on CUDA device"
    flat = input.contiguous().view(-1)
    n = flat.numel()

    compute_dtype = torch.float32
    if flat.dtype == torch.float16:
        flat = flat.to(torch.float32)

    BLOCK_SIZE = 1024
    grid_size = min((n + BLOCK_SIZE - 1) // BLOCK_SIZE, 65535)

    partials = torch.empty(grid_size, device=flat.device, dtype=compute_dtype)
    _reduce_max_kernel[(grid_size,)](
        flat, partials, n, BLOCK_SIZE=BLOCK_SIZE,
    )

    finalize_block = max(triton.next_power_of_2(grid_size), 32)
    output = torch.empty(1, device=flat.device, dtype=compute_dtype)
    _reduce_max_finalize_kernel[(1,)](
        partials, output, grid_size, BLOCK_SIZE=finalize_block,
    )

    result = output.squeeze(0)
    if input.dtype == torch.float16:
        result = result.to(torch.float16)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# ROW-WISE SUM REDUCTION
# ═════════════════════════════════════════════════════════════════════════════

@triton.jit
def _reduce_sum_rows_kernel(
    input_ptr,
    output_ptr,
    cols,
    BLOCK_COLS: tl.constexpr,
):
    """
    One program per row. Reduces `cols` elements via grid-stride within row.
    """
    row = tl.program_id(0)
    row_start = row * cols

    acc = tl.zeros([BLOCK_COLS], dtype=tl.float32)
    for start in range(0, cols, BLOCK_COLS):
        offsets = start + tl.arange(0, BLOCK_COLS)
        mask = offsets < cols
        x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)
        acc += x.to(tl.float32)

    result = tl.sum(acc, axis=0)
    tl.store(output_ptr + row, result)


def triton_reduce_sum_rows(
    input: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Row-wise (last-dim) sum reduction using Triton.

    Args:
        input: 2-D+ CUDA tensor.
        dim: Dimension to reduce (default: last).

    Returns:
        Tensor with the reduction dimension removed.
    """
    assert input.is_cuda, "Input must be on CUDA device"

    ndim = input.dim()
    if dim < 0:
        dim = ndim + dim
    assert 0 <= dim < ndim

    # Permute target dim to last if needed
    if dim != ndim - 1:
        perm = list(range(ndim))
        perm.remove(dim)
        perm.append(dim)
        input = input.permute(perm).contiguous()

    cols = input.size(-1)
    rows = input.numel() // cols
    flat = input.contiguous().view(rows, cols)

    out_shape = list(input.shape[:-1])
    if not out_shape:
        out_shape = [1]

    output = torch.empty(rows, device=input.device, dtype=input.dtype)

    BLOCK_COLS = triton.next_power_of_2(min(cols, 4096))
    BLOCK_COLS = max(BLOCK_COLS, 32)

    _reduce_sum_rows_kernel[(rows,)](
        flat, output, cols, BLOCK_COLS=BLOCK_COLS,
    )

    return output.view(out_shape)
