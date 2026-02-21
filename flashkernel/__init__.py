"""
FlashKernel — Custom CUDA kernels for transformer inference.

Build from source:
    pip install -e ".[dev]"

Requires CUDA toolkit and PyTorch with CUDA support.
"""

__version__ = "1.0.0"


def _load_extension():
    """
    Load the compiled C++/CUDA extension.

    In v1.0.0, this provides:
      - vector_add(a, b) -> Tensor
      - device_info(device_id=0) -> dict

    Future versions will add:
      - parallel_reduce (v1.0.1)
      - flash_attention_forward (v1.0.2)
      - fused_gelu_linear (v1.0.4)
      - rope_forward (v1.0.5)
      - paged_kv_cache_append / _read (v1.0.6)
    """
    try:
        from flashkernel._flashkernel_C import vector_add, device_info
        return {"vector_add": vector_add, "device_info": device_info}
    except ImportError:
        return None


_ext = _load_extension()

if _ext is not None:
    vector_add = _ext["vector_add"]
    device_info = _ext["device_info"]
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
