"""
FlashKernel — Custom CUDA kernels for transformer inference.

Build from source:
    pip install -e ".[dev]"

Requires CUDA toolkit and PyTorch with CUDA support.
"""

__version__ = "1.0.7"


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

      v1.0.6:
        - paged_kv_cache_append(pool, slot_mapping, new_keys, new_values) -> None
        - paged_kv_cache_read(pool, block_table, seq_lens, max_seq_len) -> (K, V)
        - triton_paged_kv_cache_append(...) -> None
        - triton_paged_kv_cache_read(...) -> (K, V)
        - PagedKVCache class — high-level wrapper with page management

      v1.0.7 (integration — no new C kernels):
        - GPT-2 end-to-end integration via monkey-patching
        - src.integration.gpt2_custom_kernels module

    Future versions will add:
      - Custom multi-head attention layer (v1.0.8)
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
            paged_kv_cache_append as _paged_kv_append_c,
            paged_kv_cache_read as _paged_kv_read_c,
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
            "paged_kv_cache_append": _paged_kv_append_c,
            "paged_kv_cache_read": _paged_kv_read_c,
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
    _paged_kv_cache_append_c = _ext["paged_kv_cache_append"]
    _paged_kv_cache_read_c = _ext["paged_kv_cache_read"]
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
    _paged_kv_cache_append_c = None
    _paged_kv_cache_read_c = None


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


# ─── v1.0.6: Paged KV-Cache ────────────────────────────────────────────────

def paged_kv_cache_append(pool, slot_mapping, new_keys, new_values):
    """
    Append new KV tokens to the page pool — CUDA kernel.

    Each token is mapped to a physical slot via slot_mapping:
      slot = physical_page * page_size + offset_within_page

    Args:
        pool:         [num_pages, 2, num_heads, page_size, head_dim] fp16
        slot_mapping: [total_tokens] int32 — flat physical slot per token
        new_keys:     [total_tokens, num_heads, head_dim] fp16
        new_values:   [total_tokens, num_heads, head_dim] fp16

    Modifies pool in-place.
    """
    if _paged_kv_cache_append_c is None:
        _not_compiled()
    return _paged_kv_cache_append_c(pool, slot_mapping, new_keys, new_values)


def paged_kv_cache_read(pool, block_table, seq_lens, max_seq_len):
    """
    Scatter-gather read from paged KV cache — CUDA kernel.

    Gathers KV from scattered pages into contiguous output tensors.
    Positions beyond seq_lens[batch] are zero-padded.

    Args:
        pool:        [num_pages, 2, num_heads, page_size, head_dim] fp16
        block_table: [batch, max_blocks_per_seq] int32
        seq_lens:    [batch] int32
        max_seq_len: Max sequence length for output allocation

    Returns:
        K_out: [batch, num_heads, max_seq_len, head_dim] fp16
        V_out: [batch, num_heads, max_seq_len, head_dim] fp16
    """
    if _paged_kv_cache_read_c is None:
        _not_compiled()
    return _paged_kv_cache_read_c(pool, block_table, seq_lens, max_seq_len)


def triton_paged_kv_cache_append(pool, slot_mapping, new_keys, new_values):
    """
    Append new KV tokens to the page pool — Triton kernel.

    Args:
        pool:         [num_pages, 2, num_heads, page_size, head_dim] fp16
        slot_mapping: [total_tokens] int32
        new_keys:     [total_tokens, num_heads, head_dim] fp16
        new_values:   [total_tokens, num_heads, head_dim] fp16

    Modifies pool in-place.
    """
    try:
        from src.triton.paged_kv_cache import triton_paged_kv_cache_append as _tkva
    except ImportError:
        from flashkernel._triton.paged_kv_cache import triton_paged_kv_cache_append as _tkva
    return _tkva(pool, slot_mapping, new_keys, new_values)


def triton_paged_kv_cache_read(pool, block_table, seq_lens, max_seq_len):
    """
    Scatter-gather read from paged KV cache — Triton kernel.

    Args:
        pool:        [num_pages, 2, num_heads, page_size, head_dim] fp16
        block_table: [batch, max_blocks_per_seq] int32
        seq_lens:    [batch] int32
        max_seq_len: Max sequence length for output allocation

    Returns:
        K_out: [batch, num_heads, max_seq_len, head_dim] fp16
        V_out: [batch, num_heads, max_seq_len, head_dim] fp16
    """
    try:
        from src.triton.paged_kv_cache import triton_paged_kv_cache_read as _tkvr
    except ImportError:
        from flashkernel._triton.paged_kv_cache import triton_paged_kv_cache_read as _tkvr
    return _tkvr(pool, block_table, seq_lens, max_seq_len)


# ─── PagedKVCache High-Level Class ──────────────────────────────────────────

class PagedKVCache:
    """
    Block-level KV cache with dynamic page allocation.

    Eliminates pre-allocated max-length buffers by storing KV data in
    fixed-size pages and using a page table for indirection. Page
    allocation and block table management happen on CPU (like vLLM's
    block manager); data movement uses CUDA/Triton kernels.

    Usage:
        cache = PagedKVCache(num_pages=1024, page_size=256,
                             num_heads=12, head_dim=64)
        cache.append(batch_idx=0, new_keys=k, new_values=v)
        K, V = cache.read(batch_indices=[0], max_seq_len=512)
        cache.free_sequence(batch_idx=0)

    Args:
        num_pages:  Total pages in the memory pool
        page_size:  Tokens per page (default: 256)
        num_heads:  Number of attention heads
        head_dim:   Dimension per head
        dtype:      Data type (default: torch.float16)
        backend:    'cuda' or 'triton' (default: 'cuda')
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_heads: int,
        head_dim: int,
        dtype=None,
        backend: str = "cuda",
    ):
        import torch

        if dtype is None:
            dtype = torch.float16

        self.num_pages = num_pages
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.backend = backend

        # Memory pool: [num_pages, 2(K/V), num_heads, page_size, head_dim]
        self.pool = torch.zeros(
            num_pages, 2, num_heads, page_size, head_dim,
            dtype=dtype, device="cuda",
        )

        # Page management (CPU-side, like vLLM's block manager)
        self._free_pages = list(range(num_pages - 1, -1, -1))   # stack
        self._block_tables: dict[int, list[int]] = {}   # batch_idx → [phys_page, ...]
        self._seq_lens: dict[int, int] = {}              # batch_idx → current length

    # ─── Page Management ─────────────────────────────────────────────────

    def allocate_page(self) -> int:
        """Allocate a physical page from the free pool."""
        if not self._free_pages:
            raise RuntimeError(
                f"Page pool exhausted ({self.num_pages} pages allocated). "
                "Free sequences or increase num_pages."
            )
        return self._free_pages.pop()

    def free_page(self, page_idx: int) -> None:
        """Return a physical page to the free pool."""
        self._free_pages.append(page_idx)

    def free_sequence(self, batch_idx: int) -> None:
        """Free all pages allocated for a sequence."""
        if batch_idx in self._block_tables:
            for page_idx in self._block_tables[batch_idx]:
                self.free_page(page_idx)
            del self._block_tables[batch_idx]
            del self._seq_lens[batch_idx]

    # ─── Append ──────────────────────────────────────────────────────────

    def append(self, batch_idx: int, new_keys, new_values) -> None:
        """
        Append new KV tokens for a sequence.

        Args:
            batch_idx: Sequence index in the batch
            new_keys:  [T, num_heads, head_dim] fp16 CUDA tensor
            new_values: [T, num_heads, head_dim] fp16 CUDA tensor
        """
        import torch

        T = new_keys.shape[0]
        if T == 0:
            return

        # Initialize sequence if new
        if batch_idx not in self._block_tables:
            self._block_tables[batch_idx] = []
            self._seq_lens[batch_idx] = 0

        cur_len = self._seq_lens[batch_idx]

        # Compute slot mapping for each new token
        slot_mapping = []
        for i in range(T):
            token_pos = cur_len + i
            logical_page = token_pos // self.page_size
            offset = token_pos % self.page_size

            # Allocate new page if needed
            while logical_page >= len(self._block_tables[batch_idx]):
                self._block_tables[batch_idx].append(self.allocate_page())

            physical_page = self._block_tables[batch_idx][logical_page]
            slot_mapping.append(physical_page * self.page_size + offset)

        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, device="cuda")

        # Call kernel
        if self.backend == "triton":
            triton_paged_kv_cache_append(
                self.pool, slot_mapping_t, new_keys, new_values
            )
        else:
            paged_kv_cache_append(
                self.pool, slot_mapping_t, new_keys, new_values
            )

        self._seq_lens[batch_idx] = cur_len + T

    # ─── Read ────────────────────────────────────────────────────────────

    def read(self, batch_indices: list[int], max_seq_len: int = None):
        """
        Read cached KV for given batch entries.

        Args:
            batch_indices: List of sequence indices to read
            max_seq_len:   Max output sequence length (default: max of seq_lens)

        Returns:
            K: [len(batch_indices), num_heads, max_seq_len, head_dim] fp16
            V: [len(batch_indices), num_heads, max_seq_len, head_dim] fp16
        """
        import torch

        if not batch_indices:
            raise ValueError("batch_indices must be non-empty")

        # Determine max sequence length and max blocks needed
        seq_lens_list = [self._seq_lens.get(b, 0) for b in batch_indices]
        if max_seq_len is None:
            max_seq_len = max(seq_lens_list) if seq_lens_list else 0
        if max_seq_len == 0:
            B = len(batch_indices)
            z = torch.zeros(B, self.num_heads, 0, self.head_dim,
                            dtype=self.dtype, device="cuda")
            return z, z.clone()

        max_blocks = (max_seq_len + self.page_size - 1) // self.page_size

        # Build block table and seq_lens tensors
        batch = len(batch_indices)
        block_table = torch.full((batch, max_blocks), 0, dtype=torch.int32, device="cuda")
        seq_lens_t = torch.zeros(batch, dtype=torch.int32, device="cuda")

        for i, b in enumerate(batch_indices):
            sl = self._seq_lens.get(b, 0)
            seq_lens_t[i] = sl
            bt = self._block_tables.get(b, [])
            for j, page in enumerate(bt):
                if j < max_blocks:
                    block_table[i, j] = page

        # Call kernel
        if self.backend == "triton":
            return triton_paged_kv_cache_read(
                self.pool, block_table, seq_lens_t, max_seq_len
            )
        else:
            return paged_kv_cache_read(
                self.pool, block_table, seq_lens_t, max_seq_len
            )

    # ─── Properties ──────────────────────────────────────────────────────

    @property
    def num_free_pages(self) -> int:
        """Number of unallocated pages."""
        return len(self._free_pages)

    @property
    def num_allocated_pages(self) -> int:
        """Number of pages currently in use."""
        return self.num_pages - len(self._free_pages)

    @property
    def memory_used_mb(self) -> float:
        """Memory used by allocated pages (MB)."""
        element_size = 2 if self.dtype in (None,) else self.pool.element_size()
        bytes_per_page = 2 * self.num_heads * self.page_size * self.head_dim * element_size
        return self.num_allocated_pages * bytes_per_page / (1024 * 1024)

    @property
    def memory_total_mb(self) -> float:
        """Total pool memory (MB)."""
        return self.pool.nelement() * self.pool.element_size() / (1024 * 1024)

    def get_seq_len(self, batch_idx: int) -> int:
        """Current sequence length for a batch entry."""
        return self._seq_lens.get(batch_idx, 0)

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(num_pages={self.num_pages}, page_size={self.page_size}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"free={self.num_free_pages}, backend='{self.backend}')"
        )
