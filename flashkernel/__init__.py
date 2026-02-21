"""
FlashKernel — Custom CUDA kernels for transformer inference.

Build from source:
    pip install -e ".[dev]"

Requires CUDA toolkit and PyTorch with CUDA support.
"""

__version__ = "1.0.1"


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

    Future versions will add:
      - flash_attention_forward (v1.0.2)
      - fused_gelu_linear (v1.0.4)
      - rope_forward (v1.0.5)
      - paged_kv_cache_append / _read (v1.0.6)
    """
    try:
        from flashkernel._flashkernel_C import (
            vector_add, device_info,
            reduce_sum, reduce_max,
        )
        return {
            "vector_add": vector_add,
            "device_info": device_info,
            "reduce_sum": reduce_sum,
            "reduce_max": reduce_max,
        }
    except ImportError:
        return None


_ext = _load_extension()

if _ext is not None:
    vector_add = _ext["vector_add"]
    device_info = _ext["device_info"]
    reduce_sum = _ext["reduce_sum"]
    reduce_max = _ext["reduce_max"]
else:
    def vector_add(*args, **kwargs):
        raise RuntimeError(
            "FlashKernel C++ extension not compiled. "
            "Build with: pip install -e '.[dev]' "
            "(requires CUDA toolkit)"
        )

    def device_info(*args, **kwargs):
        raise RuntimeError(
            "FlashKernel C++ extension not compiled. "
            "Build with: pip install -e '.[dev]' "
            "(requires CUDA toolkit)"
        )

    def reduce_sum(*args, **kwargs):
        raise RuntimeError(
            "FlashKernel C++ extension not compiled. "
            "Build with: pip install -e '.[dev]' "
            "(requires CUDA toolkit)"
        )

    def reduce_max(*args, **kwargs):
        raise RuntimeError(
            "FlashKernel C++ extension not compiled. "
            "Build with: pip install -e '.[dev]' "
            "(requires CUDA toolkit)"
        )


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
