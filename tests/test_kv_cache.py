"""
FlashKernel — Paged KV-Cache Tests (v1.0.6)

Validates correctness of the paged KV-cache against a naive contiguous
KV-cache implementation.

Test matrix:
  | Config                                     | Check                           |
  |--------------------------------------------|---------------------------------|
  | Single sequence, single page               | append → read == contiguous     |
  | Single sequence, multi-page                | page boundary crossing          |
  | Batch of sequences, variable lengths       | padding + per-seq correctness   |
  | Sequential token-by-token append           | incremental append == bulk      |
  | Page allocation & deallocation             | Free list correctness           |
  | CUDA vs Triton agreement                   | Cross-validation (append + read)|
  | PagedKVCache class interface               | High-level API correctness      |

Additional tests:
  - Pool occupancy and memory accounting
  - Error handling (pool exhaustion, invalid inputs)
  - Determinism (same input → same output)
  - Edge cases (seq_len = page_size, seq_len = 1, empty batch)
"""

import math
import pytest
import torch

# Skip entire module if no CUDA GPU available
pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def fk():
    """Import flashkernel."""
    import flashkernel
    return flashkernel


# ─── Test Constants ──────────────────────────────────────────────────────────

NUM_HEADS = 8
HEAD_DIM = 64
PAGE_SIZE = 16      # Small page size for testing (faster, exercises page boundaries)
NUM_PAGES = 64      # Enough pages for tests


# ─── Helper: Build contiguous reference KV ───────────────────────────────────

def build_contiguous_kv(tokens_per_seq, num_heads, head_dim, device="cuda"):
    """
    Build random K, V tensors as a contiguous cache would.

    Returns:
      keys: list of [T_i, H, D] tensors (one per sequence)
      values: list of [T_i, H, D] tensors
    """
    keys = []
    values = []
    for T in tokens_per_seq:
        k = torch.randn(T, num_heads, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(T, num_heads, head_dim, dtype=torch.float16, device=device)
        keys.append(k)
        values.append(v)
    return keys, values


def reference_contiguous_read(keys_list, values_list, max_seq_len, num_heads, head_dim):
    """
    Build contiguous [B, H, max_seq_len, D] output from per-sequence KV lists.
    This is the ground truth that paged cache must match.
    """
    B = len(keys_list)
    K_ref = torch.zeros(B, num_heads, max_seq_len, head_dim,
                        dtype=torch.float16, device="cuda")
    V_ref = torch.zeros(B, num_heads, max_seq_len, head_dim,
                        dtype=torch.float16, device="cuda")
    for i in range(B):
        T = keys_list[i].shape[0]
        # keys_list[i] is [T, H, D] → transpose to [H, T, D]
        K_ref[i, :, :T, :] = keys_list[i].permute(1, 0, 2)
        V_ref[i, :, :T, :] = values_list[i].permute(1, 0, 2)
    return K_ref, V_ref


# ─── Helper: Low-level CUDA cache operations ────────────────────────────────

def create_pool(num_pages, num_heads, page_size, head_dim):
    """Create a zeroed page pool."""
    return torch.zeros(num_pages, 2, num_heads, page_size, head_dim,
                       dtype=torch.float16, device="cuda")


def compute_slot_mapping(seq_lens_before, new_token_counts, page_size, block_tables):
    """
    Compute the flat slot mapping for new tokens.
    Returns slot_mapping tensor and updated block_tables.
    """
    all_slots = []
    for i in range(len(seq_lens_before)):
        for t in range(new_token_counts[i]):
            pos = seq_lens_before[i] + t
            logical_page = pos // page_size
            offset = pos % page_size
            physical_page = block_tables[i][logical_page]
            all_slots.append(physical_page * page_size + offset)
    return torch.tensor(all_slots, dtype=torch.int32, device="cuda")


class SimplePageAllocator:
    """Minimal page allocator for tests."""

    def __init__(self, num_pages):
        self._free = list(range(num_pages - 1, -1, -1))

    def allocate(self):
        return self._free.pop()

    def free(self, idx):
        self._free.append(idx)

    @property
    def num_free(self):
        return len(self._free)


def ensure_pages(allocator, block_table, needed_pages):
    """Ensure block_table has enough pages, allocating as needed."""
    while len(block_table) < needed_pages:
        block_table.append(allocator.allocate())


# ═════════════════════════════════════════════════════════════════════════════
# Test Classes
# ═════════════════════════════════════════════════════════════════════════════


class TestPagedKVAppendReadBasic:
    """Basic append + read: single sequence fits in one page."""

    def test_single_token_append_read(self, fk):
        """Append 1 token, read it back."""
        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        alloc = SimplePageAllocator(NUM_PAGES)
        bt = [alloc.allocate()]  # 1 page

        k = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        slot_mapping = torch.tensor([bt[0] * PAGE_SIZE + 0],
                                    dtype=torch.int32, device="cuda")
        fk.paged_kv_cache_append(pool, slot_mapping, k, v)

        block_table = torch.tensor([bt + [0] * (3)], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([1], dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, 1)

        assert K_out.shape == (1, NUM_HEADS, 1, HEAD_DIM)
        assert torch.equal(K_out[0, :, 0, :], k[0].unsqueeze(0).permute(1, 0, 2).squeeze(-2))
        # More direct check: K_out[0, h, 0, d] == k[0, h, d]
        assert torch.allclose(K_out[0, :, 0, :], k[0], atol=0)
        assert torch.allclose(V_out[0, :, 0, :], v[0], atol=0)

    def test_full_page_append_read(self, fk):
        """Fill exactly one page."""
        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        alloc = SimplePageAllocator(NUM_PAGES)
        bt = [alloc.allocate()]

        T = PAGE_SIZE
        k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        slots = [bt[0] * PAGE_SIZE + i for i in range(T)]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
        fk.paged_kv_cache_append(pool, slot_mapping, k, v)

        block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, T)

        # K_out[0, h, t, d] should match k[t, h, d]
        K_ref = k.permute(1, 0, 2).unsqueeze(0)  # [1, H, T, D]
        V_ref = v.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)

    def test_partial_page(self, fk):
        """Append fewer tokens than page_size."""
        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        alloc = SimplePageAllocator(NUM_PAGES)
        bt = [alloc.allocate()]

        T = PAGE_SIZE // 2
        k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        slots = [bt[0] * PAGE_SIZE + i for i in range(T)]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
        fk.paged_kv_cache_append(pool, slot_mapping, k, v)

        block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, T)

        K_ref = k.permute(1, 0, 2).unsqueeze(0)
        V_ref = v.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)


class TestPagedKVMultiPage:
    """Sequences spanning multiple pages."""

    @pytest.mark.parametrize("seq_len", [
        PAGE_SIZE + 1,      # Just over one page
        PAGE_SIZE * 2,      # Exactly two pages
        PAGE_SIZE * 3 + 5,  # Multiple pages + partial
    ])
    def test_multi_page_correctness(self, fk, seq_len):
        """Append tokens spanning multiple pages, verify read matches."""
        num_pages_needed = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        alloc = SimplePageAllocator(NUM_PAGES)
        bt = [alloc.allocate() for _ in range(num_pages_needed)]

        k = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        # Build slot mapping
        slots = []
        for t in range(seq_len):
            lp = t // PAGE_SIZE
            off = t % PAGE_SIZE
            slots.append(bt[lp] * PAGE_SIZE + off)
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")

        fk.paged_kv_cache_append(pool, slot_mapping, k, v)

        # Pad block table to max_blocks
        bt_padded = bt + [0] * (num_pages_needed - len(bt))
        block_table = torch.tensor([bt_padded], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, seq_len)

        K_ref = k.permute(1, 0, 2).unsqueeze(0)
        V_ref = v.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)


class TestPagedKVSequentialAppend:
    """Append tokens incrementally (one at a time, then in chunks)."""

    def test_token_by_token_append(self, fk):
        """Append tokens one at a time, verify final read matches bulk."""
        total_tokens = PAGE_SIZE + 5  # Crosses page boundary
        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        alloc = SimplePageAllocator(NUM_PAGES)
        bt = []

        all_keys = torch.randn(total_tokens, NUM_HEADS, HEAD_DIM,
                                dtype=torch.float16, device="cuda")
        all_values = torch.randn(total_tokens, NUM_HEADS, HEAD_DIM,
                                  dtype=torch.float16, device="cuda")

        for t in range(total_tokens):
            lp = t // PAGE_SIZE
            off = t % PAGE_SIZE
            while lp >= len(bt):
                bt.append(alloc.allocate())
            slot = bt[lp] * PAGE_SIZE + off
            slot_mapping = torch.tensor([slot], dtype=torch.int32, device="cuda")
            fk.paged_kv_cache_append(
                pool, slot_mapping,
                all_keys[t:t+1], all_values[t:t+1]
            )

        num_blocks = len(bt)
        block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([total_tokens], dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(
            pool, block_table, seq_lens, total_tokens
        )

        K_ref = all_keys.permute(1, 0, 2).unsqueeze(0)
        V_ref = all_values.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)

    def test_chunked_append(self, fk):
        """Append in variable-size chunks."""
        chunk_sizes = [3, 7, PAGE_SIZE, 2, PAGE_SIZE - 1]
        total = sum(chunk_sizes)

        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        alloc = SimplePageAllocator(NUM_PAGES)
        bt = []

        all_keys = torch.randn(total, NUM_HEADS, HEAD_DIM,
                                dtype=torch.float16, device="cuda")
        all_values = torch.randn(total, NUM_HEADS, HEAD_DIM,
                                  dtype=torch.float16, device="cuda")

        cur_pos = 0
        for chunk in chunk_sizes:
            slots = []
            for i in range(chunk):
                pos = cur_pos + i
                lp = pos // PAGE_SIZE
                off = pos % PAGE_SIZE
                while lp >= len(bt):
                    bt.append(alloc.allocate())
                slots.append(bt[lp] * PAGE_SIZE + off)

            slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
            fk.paged_kv_cache_append(
                pool, slot_mapping,
                all_keys[cur_pos:cur_pos+chunk],
                all_values[cur_pos:cur_pos+chunk],
            )
            cur_pos += chunk

        block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([total], dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, total)

        K_ref = all_keys.permute(1, 0, 2).unsqueeze(0)
        V_ref = all_values.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)


class TestPagedKVVariableLengthBatch:
    """Batch of sequences with different lengths."""

    @pytest.mark.parametrize("seq_lens_list", [
        [5, 10, 3],
        [PAGE_SIZE, PAGE_SIZE + 1, 1],
        [PAGE_SIZE * 2 + 3, 7, PAGE_SIZE - 1, PAGE_SIZE],
        [1, 1, 1, 1, 1],
    ])
    def test_variable_length_batch(self, fk, seq_lens_list):
        """Multiple sequences with different lengths in one batch."""
        B = len(seq_lens_list)
        max_seq = max(seq_lens_list)

        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        alloc = SimplePageAllocator(NUM_PAGES)

        # Generate per-sequence KV and append
        all_keys = []
        all_values = []
        block_tables = []

        for b in range(B):
            T = seq_lens_list[b]
            k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            all_keys.append(k)
            all_values.append(v)

            bt = []
            slots = []
            for t in range(T):
                lp = t // PAGE_SIZE
                off = t % PAGE_SIZE
                while lp >= len(bt):
                    bt.append(alloc.allocate())
                slots.append(bt[lp] * PAGE_SIZE + off)

            slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
            fk.paged_kv_cache_append(pool, slot_mapping, k, v)
            block_tables.append(bt)

        # Build block table tensor (pad to uniform width)
        max_blocks = max(len(bt) for bt in block_tables)
        bt_tensor = torch.zeros(B, max_blocks, dtype=torch.int32, device="cuda")
        for b in range(B):
            for j, p in enumerate(block_tables[b]):
                bt_tensor[b, j] = p

        seq_lens_t = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(pool, bt_tensor, seq_lens_t, max_seq)

        # Verify each sequence
        K_ref, V_ref = reference_contiguous_read(
            all_keys, all_values, max_seq, NUM_HEADS, HEAD_DIM
        )
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)


class TestPagedKVPaddingZeroFill:
    """Verify that positions beyond seq_len are zero in read output."""

    def test_padding_is_zero(self, fk):
        """Read with max_seq_len > actual seq_len → padding should be zero."""
        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        alloc = SimplePageAllocator(NUM_PAGES)
        bt = [alloc.allocate()]

        T = 5
        k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        slots = [bt[0] * PAGE_SIZE + i for i in range(T)]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
        fk.paged_kv_cache_append(pool, slot_mapping, k, v)

        max_out = PAGE_SIZE * 2  # Much larger than T
        block_table = torch.tensor([bt + [0]], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, max_out)

        # Valid region matches
        K_ref = k.permute(1, 0, 2)  # [H, T, D]
        assert torch.allclose(K_out[0, :, :T, :], K_ref, atol=0)

        # Padding is zero
        assert torch.all(K_out[0, :, T:, :] == 0)
        assert torch.all(V_out[0, :, T:, :] == 0)


class TestPagedKVDeterminism:
    """Same inputs → same outputs (deterministic)."""

    def test_deterministic_append_read(self, fk):
        """Two identical append+read cycles should produce identical results."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
            alloc = SimplePageAllocator(NUM_PAGES)
            bt = [alloc.allocate()]

            T = 10
            k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

            slots = [bt[0] * PAGE_SIZE + i for i in range(T)]
            slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
            fk.paged_kv_cache_append(pool, slot_mapping, k, v)

            block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
            seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
            K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, T)
            results.append((K_out.clone(), V_out.clone()))

        assert torch.equal(results[0][0], results[1][0])
        assert torch.equal(results[0][1], results[1][1])


class TestPagedKVNonContiguousPages:
    """Pages are not in physical order — tests true scatter-gather."""

    def test_reversed_page_order(self, fk):
        """Allocate pages in reverse order, verify correct gather."""
        seq_len = PAGE_SIZE * 3
        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)

        # Manually assign pages in reverse order
        bt = [NUM_PAGES - 1, NUM_PAGES - 2, NUM_PAGES - 3]

        k = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        slots = []
        for t in range(seq_len):
            lp = t // PAGE_SIZE
            off = t % PAGE_SIZE
            slots.append(bt[lp] * PAGE_SIZE + off)
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
        fk.paged_kv_cache_append(pool, slot_mapping, k, v)

        block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, seq_len)

        K_ref = k.permute(1, 0, 2).unsqueeze(0)
        V_ref = v.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)

    def test_interleaved_pages_two_sequences(self, fk):
        """Two sequences with interleaved physical pages."""
        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)

        # Seq 0 gets pages [0, 2, 4], seq 1 gets pages [1, 3, 5]
        bt0 = [0, 2, 4]
        bt1 = [1, 3, 5]

        T0 = PAGE_SIZE * 3
        T1 = PAGE_SIZE * 2 + 7

        k0 = torch.randn(T0, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v0 = torch.randn(T0, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        k1 = torch.randn(T1, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v1 = torch.randn(T1, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        # Append seq 0
        slots0 = [bt0[t // PAGE_SIZE] * PAGE_SIZE + t % PAGE_SIZE for t in range(T0)]
        fk.paged_kv_cache_append(
            pool, torch.tensor(slots0, dtype=torch.int32, device="cuda"), k0, v0
        )

        # Append seq 1
        slots1 = [bt1[t // PAGE_SIZE] * PAGE_SIZE + t % PAGE_SIZE for t in range(T1)]
        fk.paged_kv_cache_append(
            pool, torch.tensor(slots1, dtype=torch.int32, device="cuda"), k1, v1
        )

        # Read both
        max_seq = max(T0, T1)
        max_blocks = 3
        block_table = torch.zeros(2, max_blocks, dtype=torch.int32, device="cuda")
        block_table[0, :3] = torch.tensor(bt0)
        block_table[1, :3] = torch.tensor(bt1)
        seq_lens = torch.tensor([T0, T1], dtype=torch.int32, device="cuda")

        K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, max_seq)

        K_ref, V_ref = reference_contiguous_read(
            [k0, k1], [v0, v1], max_seq, NUM_HEADS, HEAD_DIM
        )
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)


class TestPagedKVTriton:
    """Triton kernels match CUDA kernels."""

    def test_triton_append_read_matches_cuda(self, fk):
        """Triton append + Triton read should match CUDA results."""
        seq_len = PAGE_SIZE * 2 + 5
        num_pages_needed = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE

        # CUDA path
        pool_cuda = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        alloc = SimplePageAllocator(NUM_PAGES)
        bt = [alloc.allocate() for _ in range(num_pages_needed)]

        k = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        slots = []
        for t in range(seq_len):
            lp = t // PAGE_SIZE
            off = t % PAGE_SIZE
            slots.append(bt[lp] * PAGE_SIZE + off)
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")

        fk.paged_kv_cache_append(pool_cuda, slot_mapping, k, v)

        block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
        K_cuda, V_cuda = fk.paged_kv_cache_read(pool_cuda, block_table, seq_lens, seq_len)

        # Triton path (same data, same page assignments)
        pool_triton = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        fk.triton_paged_kv_cache_append(pool_triton, slot_mapping, k, v)
        K_triton, V_triton = fk.triton_paged_kv_cache_read(
            pool_triton, block_table, seq_lens, seq_len
        )

        assert torch.allclose(K_cuda, K_triton, atol=0)
        assert torch.allclose(V_cuda, V_triton, atol=0)

    def test_triton_variable_length_batch(self, fk):
        """Triton handles variable-length batch correctly."""
        seq_lens_list = [PAGE_SIZE + 3, 7, PAGE_SIZE * 2]
        B = len(seq_lens_list)
        max_seq = max(seq_lens_list)

        pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
        alloc = SimplePageAllocator(NUM_PAGES)

        all_keys, all_values, block_tables = [], [], []

        for b in range(B):
            T = seq_lens_list[b]
            k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            all_keys.append(k)
            all_values.append(v)

            bt = []
            slots = []
            for t in range(T):
                lp = t // PAGE_SIZE
                off = t % PAGE_SIZE
                while lp >= len(bt):
                    bt.append(alloc.allocate())
                slots.append(bt[lp] * PAGE_SIZE + off)

            slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
            fk.triton_paged_kv_cache_append(pool, slot_mapping, k, v)
            block_tables.append(bt)

        max_blocks = max(len(bt) for bt in block_tables)
        bt_tensor = torch.zeros(B, max_blocks, dtype=torch.int32, device="cuda")
        for b in range(B):
            for j, p in enumerate(block_tables[b]):
                bt_tensor[b, j] = p

        seq_lens_t = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
        K_out, V_out = fk.triton_paged_kv_cache_read(pool, bt_tensor, seq_lens_t, max_seq)

        K_ref, V_ref = reference_contiguous_read(
            all_keys, all_values, max_seq, NUM_HEADS, HEAD_DIM
        )
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)


class TestPagedKVCacheClass:
    """High-level PagedKVCache class tests."""

    def test_basic_append_and_read(self, fk):
        """PagedKVCache append + read matches contiguous reference."""
        cache = fk.PagedKVCache(
            num_pages=NUM_PAGES, page_size=PAGE_SIZE,
            num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        )

        T = PAGE_SIZE + 5
        k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        cache.append(batch_idx=0, new_keys=k, new_values=v)
        assert cache.get_seq_len(0) == T

        K_out, V_out = cache.read([0])

        K_ref = k.permute(1, 0, 2).unsqueeze(0)
        V_ref = v.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)

    def test_incremental_append(self, fk):
        """Multiple appends to same sequence, then read."""
        cache = fk.PagedKVCache(
            num_pages=NUM_PAGES, page_size=PAGE_SIZE,
            num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        )

        chunks = [5, 3, PAGE_SIZE, 2]
        all_k = []
        all_v = []
        for c in chunks:
            k = torch.randn(c, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            v = torch.randn(c, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            cache.append(0, k, v)
            all_k.append(k)
            all_v.append(v)

        total = sum(chunks)
        assert cache.get_seq_len(0) == total

        K_out, V_out = cache.read([0])

        full_k = torch.cat(all_k, dim=0)
        full_v = torch.cat(all_v, dim=0)
        K_ref = full_k.permute(1, 0, 2).unsqueeze(0)
        V_ref = full_v.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)

    def test_multiple_sequences(self, fk):
        """Multiple independent sequences in the cache."""
        cache = fk.PagedKVCache(
            num_pages=NUM_PAGES, page_size=PAGE_SIZE,
            num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        )

        lens = [10, PAGE_SIZE + 3, 5]
        keys_list = []
        vals_list = []
        for i, T in enumerate(lens):
            k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            cache.append(i, k, v)
            keys_list.append(k)
            vals_list.append(v)

        max_seq = max(lens)
        K_out, V_out = cache.read([0, 1, 2], max_seq_len=max_seq)

        K_ref, V_ref = reference_contiguous_read(
            keys_list, vals_list, max_seq, NUM_HEADS, HEAD_DIM
        )
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)

    def test_free_sequence(self, fk):
        """Free a sequence and verify pages are returned."""
        cache = fk.PagedKVCache(
            num_pages=NUM_PAGES, page_size=PAGE_SIZE,
            num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        )

        initial_free = cache.num_free_pages
        T = PAGE_SIZE * 3
        k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        cache.append(0, k, v)

        pages_used = cache.num_allocated_pages
        assert pages_used == 3
        assert cache.num_free_pages == initial_free - 3

        cache.free_sequence(0)
        assert cache.num_free_pages == initial_free
        assert cache.get_seq_len(0) == 0

    def test_triton_backend(self, fk):
        """PagedKVCache with backend='triton' works correctly."""
        cache = fk.PagedKVCache(
            num_pages=NUM_PAGES, page_size=PAGE_SIZE,
            num_heads=NUM_HEADS, head_dim=HEAD_DIM,
            backend="triton",
        )

        T = PAGE_SIZE + 7
        k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        cache.append(0, k, v)
        K_out, V_out = cache.read([0])

        K_ref = k.permute(1, 0, 2).unsqueeze(0)
        V_ref = v.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)

    def test_memory_accounting(self, fk):
        """Memory used/total properties are correct."""
        cache = fk.PagedKVCache(
            num_pages=NUM_PAGES, page_size=PAGE_SIZE,
            num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        )
        assert cache.memory_used_mb == 0.0
        assert cache.memory_total_mb > 0.0

        T = PAGE_SIZE * 2
        k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        cache.append(0, k, v)

        assert cache.memory_used_mb > 0.0
        assert cache.num_allocated_pages == 2

    def test_repr(self, fk):
        """__repr__ works."""
        cache = fk.PagedKVCache(
            num_pages=32, page_size=16,
            num_heads=4, head_dim=64,
        )
        r = repr(cache)
        assert "PagedKVCache" in r
        assert "num_pages=32" in r
        assert "page_size=16" in r


class TestPagedKVEdgeCases:
    """Edge cases and error handling."""

    def test_exact_page_boundary(self, fk):
        """Sequence length exactly equal to page_size × N."""
        for n in [1, 2, 3]:
            T = PAGE_SIZE * n
            pool = create_pool(NUM_PAGES, NUM_HEADS, PAGE_SIZE, HEAD_DIM)
            alloc = SimplePageAllocator(NUM_PAGES)
            bt = [alloc.allocate() for _ in range(n)]

            k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

            slots = [bt[t // PAGE_SIZE] * PAGE_SIZE + t % PAGE_SIZE for t in range(T)]
            slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
            fk.paged_kv_cache_append(pool, slot_mapping, k, v)

            block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
            seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
            K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, T)

            K_ref = k.permute(1, 0, 2).unsqueeze(0)
            assert torch.allclose(K_out, K_ref, atol=0)

    def test_single_head(self, fk):
        """Works with num_heads=1."""
        NH = 1
        pool = torch.zeros(NUM_PAGES, 2, NH, PAGE_SIZE, HEAD_DIM,
                           dtype=torch.float16, device="cuda")
        alloc = SimplePageAllocator(NUM_PAGES)
        bt = [alloc.allocate()]

        T = 5
        k = torch.randn(T, NH, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NH, HEAD_DIM, dtype=torch.float16, device="cuda")

        slots = [bt[0] * PAGE_SIZE + i for i in range(T)]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
        fk.paged_kv_cache_append(pool, slot_mapping, k, v)

        block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, T)

        K_ref = k.permute(1, 0, 2).unsqueeze(0)
        V_ref = v.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)

    @pytest.mark.parametrize("head_dim", [32, 64, 128])
    def test_various_head_dims(self, fk, head_dim):
        """Works with different head dimensions."""
        pool = torch.zeros(NUM_PAGES, 2, NUM_HEADS, PAGE_SIZE, head_dim,
                           dtype=torch.float16, device="cuda")
        alloc = SimplePageAllocator(NUM_PAGES)
        bt = [alloc.allocate()]

        T = 7
        k = torch.randn(T, NUM_HEADS, head_dim, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, head_dim, dtype=torch.float16, device="cuda")

        slots = [bt[0] * PAGE_SIZE + i for i in range(T)]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")
        fk.paged_kv_cache_append(pool, slot_mapping, k, v)

        block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
        seq_lens = torch.tensor([T], dtype=torch.int32, device="cuda")
        K_out, V_out = fk.paged_kv_cache_read(pool, block_table, seq_lens, T)

        K_ref = k.permute(1, 0, 2).unsqueeze(0)
        V_ref = v.permute(1, 0, 2).unsqueeze(0)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)


class TestPagedKVCacheClassPoolExhaustion:
    """Pool exhaustion handling."""

    def test_pool_exhaustion_raises(self, fk):
        """Allocating more pages than available raises RuntimeError."""
        cache = fk.PagedKVCache(
            num_pages=2, page_size=PAGE_SIZE,
            num_heads=NUM_HEADS, head_dim=HEAD_DIM,
        )

        # Fill all pages
        T = PAGE_SIZE * 2
        k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        cache.append(0, k, v)
        assert cache.num_free_pages == 0

        # Should raise on next allocation
        with pytest.raises(RuntimeError, match="Page pool exhausted"):
            k2 = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            v2 = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
            cache.append(1, k2, v2)


class TestPagedKVLargeScale:
    """Larger-scale tests with realistic configurations."""

    @pytest.mark.slow
    @pytest.mark.parametrize("config", [
        {"B": 4, "H": 12, "D": 64, "seqs": [512, 256, 128, 1024]},
        {"B": 2, "H": 8, "D": 128, "seqs": [2048, 1024]},
    ])
    def test_large_batch(self, fk, config):
        """Larger batch with realistic dimensions."""
        B = config["B"]
        H = config["H"]
        D = config["D"]
        seqs = config["seqs"]

        total_tokens = sum(seqs)
        pages_needed = sum((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seqs)
        num_pages = pages_needed + 10  # headroom

        cache = fk.PagedKVCache(
            num_pages=num_pages, page_size=PAGE_SIZE,
            num_heads=H, head_dim=D,
        )

        keys_list = []
        vals_list = []
        for i, T in enumerate(seqs):
            k = torch.randn(T, H, D, dtype=torch.float16, device="cuda")
            v = torch.randn(T, H, D, dtype=torch.float16, device="cuda")
            cache.append(i, k, v)
            keys_list.append(k)
            vals_list.append(v)

        max_seq = max(seqs)
        K_out, V_out = cache.read(list(range(B)), max_seq_len=max_seq)

        K_ref, V_ref = reference_contiguous_read(keys_list, vals_list, max_seq, H, D)
        assert torch.allclose(K_out, K_ref, atol=0)
        assert torch.allclose(V_out, V_ref, atol=0)
