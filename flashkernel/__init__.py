"""
FlashKernel — Custom CUDA kernels for transformer inference.

Build from source:
    pip install -e ".[dev]"

Requires CUDA toolkit and PyTorch with CUDA support.
"""

__version__ = "1.0.5"


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

      v1.0.3:
        - triton_flash_attention_forward(Q, K, V, scale, is_causal) -> (O, L)

      v1.0.4:
        - fused_gelu_linear(X, W, bias, use_tanh_approx) -> Y
        - triton_fused_gelu_linear(X, W, bias, use_tanh_approx) -> Y

      v1.0.5:
        - rope_precompute_freqs(max_seq_len, head_dim, base) -> (cos, sin)
        - rope_forward(Q, K, cos_table, sin_table) -> (Q_rot, K_rot)
        - rope_forward_fused(Q, K, base) -> (Q_rot, K_rot)
        - triton_rope_forward(Q, K, cos_table, sin_table) -> (Q_rot, K_rot)
        - triton_rope_forward_fused(Q, K, base) -> (Q_rot, K_rot)

    Future versions will add:
      - paged_kv_cache_append / _read (v1.0.6)
    """
    try:
        from flashkernel._flashkernel_C import (
            vector_add, device_info,
            reduce_sum, reduce_max,
            flash_attention_forward as _flash_attn_fwd,
            fused_gelu_linear as _fused_gelu_linear_c,
            rope_precompute_freqs as _rope_precompute_freqs_c,
            rope_forward as _rope_forward_c,
            rope_forward_fused as _rope_forward_fused_c,
        )
        return {
            "vector_add": vector_add,
            "device_info": device_info,
            "reduce_sum": reduce_sum,
            "reduce_max": reduce_max,
            "flash_attention_forward": _flash_attn_fwd,
            "fused_gelu_linear": _fused_gelu_linear_c,
            "rope_precompute_freqs": _rope_precompute_freqs_c,
            "rope_forward": _rope_forward_c,
            "rope_forward_fused": _rope_forward_fused_c,
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
    _fused_gelu_linear_c = _ext["fused_gelu_linear"]
    _rope_precompute_freqs_c = _ext["rope_precompute_freqs"]
    _rope_forward_c = _ext["rope_forward"]
    _rope_forward_fused_c = _ext["rope_forward_fused"]
else:
    vector_add = None
    device_info = None
    reduce_sum = None
    reduce_max = None
    _flash_attention_forward_c = None
    _fused_gelu_linear_c = None
    _rope_precompute_freqs_c = None
    _rope_forward_c = None
    _rope_forward_fused_c = None


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


def triton_flash_attention_forward(Q, K, V, scale=None, is_causal=False):
    """
    Triton FlashAttention forward — same tiled online-softmax algorithm
    as the CUDA kernel, autotuned over tile sizes and warp counts.

    Args:
        Q: [B, H, N, D] fp16 CUDA tensor
        K: [B, H, N, D] fp16 CUDA tensor
        V: [B, H, N, D] fp16 CUDA tensor
        scale: Softmax scale (default: 1/sqrt(D))
        is_causal: Apply causal attention mask

    Returns:
        O: [B, H, N, D] fp16 — attention output
        L: [B, H, N]    fp32 — log-sum-exp
    """
    try:
        from src.triton.flash_attention import triton_flash_attention_forward as _tfn
    except ImportError:
        from flashkernel._triton.flash_attention import triton_flash_attention_forward as _tfn
    return _tfn(Q, K, V, scale=scale, is_causal=is_causal)


# ─── v1.0.4: Fused GeLU+Linear ─────────────────────────────────────────────

def fused_gelu_linear(X, W, bias=None, use_tanh_approx=False):
    """
    Fused GeLU+Linear: Y = GeLU(X @ W^T + bias)  — CUDA kernel.

    Eliminates one HBM round-trip by fusing matmul + bias + GeLU into
    a single kernel. The intermediate linear output is never written to HBM.

    Args:
        X: [M, K] fp16 CUDA tensor — input activations
        W: [N, K] fp16 CUDA tensor — weight matrix
        bias: [N] fp16 CUDA tensor or None — optional bias
        use_tanh_approx: If True, use fast tanh GeLU; else exact erf GeLU.

    Returns:
        Y: [M, N] fp16 — GeLU(X @ W^T + bias)
    """
    if _fused_gelu_linear_c is None:
        _not_compiled()
    return _fused_gelu_linear_c(X, W, bias, use_tanh_approx)


def triton_fused_gelu_linear(X, W, bias=None, use_tanh_approx=False):
    """
    Fused GeLU+Linear: Y = GeLU(X @ W^T + bias)  — Triton kernel.

    Same fusion as the CUDA kernel, but implemented in Triton with
    autotune over tile sizes.

    Args:
        X: [M, K] fp16 CUDA tensor — input activations
        W: [N, K] fp16 CUDA tensor — weight matrix
        bias: [N] fp16 CUDA tensor or None — optional bias
        use_tanh_approx: If True, use fast tanh GeLU; else exact erf GeLU.

    Returns:
        Y: [M, N] fp16 — GeLU(X @ W^T + bias)
    """
    try:
        from src.triton.fused_gelu_linear import triton_fused_gelu_linear as _tfgl
    except ImportError:
        from flashkernel._triton.fused_gelu_linear import triton_fused_gelu_linear as _tfgl
    return _tfgl(X, W, bias=bias, use_tanh_approx=use_tanh_approx)


# ─── v1.0.5: RoPE Embedding ────────────────────────────────────────────────

def rope_precompute_freqs(max_seq_len, head_dim, base=10000.0):
    """
    Precompute RoPE cos/sin frequency tables on device — CUDA kernel.

    Args:
        max_seq_len: Maximum sequence length
        head_dim: Head dimension (must be even)
        base: Frequency base (default: 10000.0)

    Returns:
        cos_table: [max_seq_len, head_dim/2] fp32
        sin_table: [max_seq_len, head_dim/2] fp32
    """
    if _rope_precompute_freqs_c is None:
        _not_compiled()
    return _rope_precompute_freqs_c(max_seq_len, head_dim, base)


def rope_forward(Q, K, cos_table, sin_table):
    """
    Apply Rotary Position Embedding to Q and K — CUDA kernel with table lookup.

    Uses precomputed cos/sin tables. Best when reusing tables across
    multiple forward passes (e.g., during decoding).

    Args:
        Q: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor
        K: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor
        cos_table: [max_seq_len, head_dim/2] fp32 — from rope_precompute_freqs
        sin_table: [max_seq_len, head_dim/2] fp32 — from rope_precompute_freqs

    Returns:
        Q_rot: [batch, num_heads, seq_len, head_dim] fp16 — rotated Q
        K_rot: [batch, num_heads, seq_len, head_dim] fp16 — rotated K

    head_dim must be even.
    """
    if _rope_forward_c is None:
        _not_compiled()
    return _rope_forward_c(Q, K, cos_table, sin_table)


def rope_forward_fused(Q, K, base=10000.0):
    """
    Apply RoPE to Q and K with on-the-fly sin/cos — CUDA fused variant.

    Computes sin/cos per-thread in registers. No precomputed table needed.
    Saves HBM bandwidth but does slightly more compute per element.
    Best for one-shot inference or when tables can't be cached.

    Args:
        Q: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor
        K: [batch, num_heads, seq_len, head_dim] fp16 CUDA tensor
        base: Frequency base (default: 10000.0)

    Returns:
        Q_rot: [batch, num_heads, seq_len, head_dim] fp16 — rotated Q
        K_rot: [batch, num_heads, seq_len, head_dim] fp16 — rotated K

    head_dim must be even.
    """
    if _rope_forward_fused_c is None:
        _not_compiled()
    return _rope_forward_fused_c(Q, K, base)


def triton_rope_forward(Q, K, cos_table, sin_table):
    """
    Apply RoPE to Q and K using precomputed tables — Triton kernel.

    Args:
        Q: [B, H, N, D] fp16 CUDA tensor (modified in-place)
        K: [B, H, N, D] fp16 CUDA tensor (modified in-place)
        cos_table: [max_seq_len, D/2] fp32
        sin_table: [max_seq_len, D/2] fp32

    Returns:
        Q, K (modified in-place, returned for convenience)
    """
    try:
        from src.triton.rope import triton_rope_forward as _trf
    except ImportError:
        from flashkernel._triton.rope import triton_rope_forward as _trf
    return _trf(Q, K, cos_table, sin_table)


def triton_rope_forward_fused(Q, K, base=10000.0):
    """
    Apply RoPE to Q and K with on-the-fly sin/cos — Triton fused variant.

    Args:
        Q: [B, H, N, D] fp16 CUDA tensor (modified in-place)
        K: [B, H, N, D] fp16 CUDA tensor (modified in-place)
        base: Frequency base (default: 10000.0)

    Returns:
        Q, K (modified in-place, returned for convenience)
    """
    try:
        from src.triton.rope import triton_rope_forward_fused as _trff
    except ImportError:
        from flashkernel._triton.rope import triton_rope_forward_fused as _trff
    return _trff(Q, K, base=base)
