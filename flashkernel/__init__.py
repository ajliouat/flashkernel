"""
FlashKernel — Custom CUDA kernels for transformer inference.

Build from source:
    pip install -e ".[dev]"

Requires CUDA toolkit and PyTorch with CUDA support.
"""

__version__ = "1.0.2"


def _load_extension():
    """
    Load the compiled C++/CUDA extension.

    Available kernels:
      v1.0.0:
        - vector_add(a, b) -> Tensor
        - device_info(device_id=0) -> dict
      v1.0.1:
        - reduce_sum(input, dim=-1) -> Tensor
        - reduce_max(input) -> Tensor
      v1.0.2:
        - flash_attention_forward(Q, K, V, scale, is_causal) -> (O, L)

    Future versions will add:
      - fused_gelu_linear (v1.0.4)
      - rope_forward (v1.0.5)
      - paged_kv_cache_append / _read (v1.0.6)
    """
    try:
        from flashkernel._flashkernel_C import (
            vector_add, device_info,
            reduce_sum, reduce_max,
            flash_attention_forward as _flash_attn_fwd,
        )
        return {
            "vector_add": vector_add,
            "device_info": device_info,
            "reduce_sum": reduce_sum,
            "reduce_max": reduce_max,
            "flash_attention_forward": _flash_attn_fwd,
        }
    except ImportError:
        return None


_ext = _load_extension()

if _ext is not None:
    vector_add = _ext["vector_add"]
    device_info = _ext["device_info"]
    reduce_sum = _ext["reduce_sum"]
    reduce_max = _ext["reduce_max"]
    _flash_attention_forward_c = _ext["flash_attention_forward"]
else:
    vector_add = None
    device_info = None
    reduce_sum = None
    reduce_max = None
    _flash_attention_forward_c = None


def _not_compiled(*args, **kwargs):
    raise RuntimeError(
        "FlashKernel C++ extension not compiled. "
        "Build with: pip install -e '.[dev]' "
        "(requires CUDA toolkit)"
    )

# Fallback stubs
if vector_add is None:
    vector_add = _not_compiled
if device_info is None:
    device_info = _not_compiled
if reduce_sum is None:
    reduce_sum = _not_compiled
if reduce_max is None:
    reduce_max = _not_compiled


def flash_attention_forward(Q, K, V, scale=None, is_causal=False):
    """
    FlashAttention forward pass — tiled attention with online softmax.

    No N×N attention matrix materialized in HBM.

    Args:
        Q: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor
        K: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor
        V: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor
        scale: Softmax scale (default: 1/sqrt(head_dim))
        is_causal: Apply causal mask (upper triangle = -inf)

    Returns:
        O: [batch, num_heads, seq_len, head_dim] fp16 — attention output
        L: [batch, num_heads, seq_len] fp32 — log-sum-exp (for backward)

    Supported head_dim: 64 (tiles 64×64), 128 (tiles 32×64)
    """
    if _flash_attention_forward_c is None:
        _not_compiled()

    import math
    if scale is None or scale < 0:
        scale = 1.0 / math.sqrt(Q.shape[-1])

    return _flash_attention_forward_c(Q, K, V, scale, is_causal)


# ─── Triton kernels (pure Python, no compilation needed) ────────────────────

def triton_reduce_sum(input, dim=None):
    """Triton-based sum reduction. Falls back to row-wise if dim specified."""
    try:
        from src.triton.reduce import triton_reduce_sum as _triton_sum
        from src.triton.reduce import triton_reduce_sum_rows as _triton_sum_rows
    except ImportError:
        from flashkernel._triton.reduce import triton_reduce_sum as _triton_sum
        from flashkernel._triton.reduce import triton_reduce_sum_rows as _triton_sum_rows

    if dim is None:
        return _triton_sum(input)
    return _triton_sum_rows(input, dim=dim)


def triton_reduce_max(input):
    """Triton-based max reduction."""
    try:
        from src.triton.reduce import triton_reduce_max as _triton_max
    except ImportError:
        from flashkernel._triton.reduce import triton_reduce_max as _triton_max
    return _triton_max(input)
