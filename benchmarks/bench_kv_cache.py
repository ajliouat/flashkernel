"""
FlashKernel — Paged KV-Cache Benchmark (v1.0.6)

Benchmarks:
  1. Memory savings: paged vs contiguous at seq=[256, 512, 1024, 2048, 4096]
  2. Read latency: scatter-gather from pages vs contiguous memcpy
  3. Append latency: token insertion
  4. CUDA vs Triton comparison for both append and read
  5. Batch sweep: B=[1, 4, 8] at seq=1024

Key insight: Paged KV-cache saves memory proportional to (max_seq - avg_seq),
but scatter-gather reads have higher latency than contiguous. This benchmark
quantifies the trade-off.

Outputs:
  - Console tables with timing, memory, and comparison
  - benchmarks/results/kv_cache.csv
"""

import os
import sys
import math
import argparse

import torch

# Add project root for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks.harness import BenchmarkRunner, BenchmarkResult, compare_results


def check_cuda():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"VRAM: {mem_gb:.1f} GB")
    print()


# ─── Helpers ─────────────────────────────────────────────────────────────────

PAGE_SIZE = 256   # Tokens per page (production default)
NUM_HEADS = 12
HEAD_DIM = 64


def create_paged_cache(num_pages, page_size=PAGE_SIZE):
    """Create pool and allocator state."""
    pool = torch.zeros(num_pages, 2, NUM_HEADS, page_size, HEAD_DIM,
                       dtype=torch.float16, device="cuda")
    return pool


def fill_paged_cache(fk, pool, seq_len, page_size=PAGE_SIZE, use_triton=False):
    """
    Fill a paged cache for one sequence, return block_table and seq_lens tensors.
    """
    num_pages_needed = (seq_len + page_size - 1) // page_size
    bt = list(range(num_pages_needed))  # sequential allocation for simplicity

    k = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
    v = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

    slots = [bt[t // page_size] * page_size + t % page_size for t in range(seq_len)]
    slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")

    if use_triton:
        fk.triton_paged_kv_cache_append(pool, slot_mapping, k, v)
    else:
        fk.paged_kv_cache_append(pool, slot_mapping, k, v)

    block_table = torch.tensor([bt], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")

    return block_table, seq_lens, k, v


def fill_paged_cache_batch(fk, pool, seq_lens_list, page_size=PAGE_SIZE, use_triton=False):
    """Fill paged cache for a batch of sequences."""
    B = len(seq_lens_list)
    block_tables = []
    page_offset = 0

    for b in range(B):
        T = seq_lens_list[b]
        num_pages_needed = (T + page_size - 1) // page_size
        bt = list(range(page_offset, page_offset + num_pages_needed))
        page_offset += num_pages_needed

        k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        slots = [bt[t // page_size] * page_size + t % page_size for t in range(T)]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")

        if use_triton:
            fk.triton_paged_kv_cache_append(pool, slot_mapping, k, v)
        else:
            fk.paged_kv_cache_append(pool, slot_mapping, k, v)

        block_tables.append(bt)

    max_blocks = max(len(bt) for bt in block_tables)
    bt_tensor = torch.zeros(B, max_blocks, dtype=torch.int32, device="cuda")
    for b in range(B):
        for j, p in enumerate(block_tables[b]):
            bt_tensor[b, j] = p

    seq_lens_t = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
    return bt_tensor, seq_lens_t


# ─── Benchmark 1: Memory Savings ────────────────────────────────────────────

def bench_memory_savings(fk, runner):
    """Compare memory usage: paged vs contiguous at various sequence lengths."""
    print("\n" + "=" * 80)
    print("  Benchmark 1: Memory Savings — Paged vs Contiguous")
    print("=" * 80)

    batch_size = 4
    max_seq_len = 4096  # Pre-allocated contiguous max

    # Variable actual lengths (simulating real workloads)
    configs = [
        {"actual_seqs": [256] * batch_size, "label": "all=256"},
        {"actual_seqs": [512] * batch_size, "label": "all=512"},
        {"actual_seqs": [1024] * batch_size, "label": "all=1024"},
        {"actual_seqs": [2048] * batch_size, "label": "all=2048"},
        {"actual_seqs": [4096] * batch_size, "label": "all=4096"},
        {"actual_seqs": [256, 512, 1024, 2048], "label": "mixed"},
        {"actual_seqs": [128, 128, 128, 4096], "label": "skewed"},
    ]

    element_bytes = 2  # fp16

    print(f"\n  {'Config':<20} {'Contiguous MB':>14} {'Paged MB':>12} {'Savings':>10} {'Ratio':>8}")
    print("  " + "─" * 66)

    results = []

    for cfg in configs:
        actual_seqs = cfg["actual_seqs"]
        label = cfg["label"]

        # Contiguous: batch × max_seq_len × 2(K/V) × H × D × elem_size
        contig_bytes = batch_size * max_seq_len * 2 * NUM_HEADS * HEAD_DIM * element_bytes
        contig_mb = contig_bytes / (1024 * 1024)

        # Paged: sum of actual pages needed + page table overhead
        total_pages = sum((s + PAGE_SIZE - 1) // PAGE_SIZE for s in actual_seqs)
        paged_data_bytes = total_pages * 2 * NUM_HEADS * PAGE_SIZE * HEAD_DIM * element_bytes
        # Page table overhead
        max_blocks = max((s + PAGE_SIZE - 1) // PAGE_SIZE for s in actual_seqs)
        pt_bytes = batch_size * max_blocks * 4  # int32
        paged_bytes = paged_data_bytes + pt_bytes
        paged_mb = paged_bytes / (1024 * 1024)

        savings_pct = (1.0 - paged_mb / contig_mb) * 100 if contig_mb > 0 else 0
        ratio = contig_mb / paged_mb if paged_mb > 0 else float('inf')

        print(f"  {label:<20} {contig_mb:>12.2f}  {paged_mb:>12.2f} {savings_pct:>8.1f}% {ratio:>7.1f}×")

        result = BenchmarkResult(
            name=f"memory_{label}",
            n_warmup=0, n_timed=1, times_ms=[0],
        )
        result.extra = {
            "contiguous_mb": round(contig_mb, 2),
            "paged_mb": round(paged_mb, 2),
            "savings_pct": round(savings_pct, 1),
        }
        result.compute_stats()
        results.append(result)

    return results


# ─── Benchmark 2: Read Latency ──────────────────────────────────────────────

def bench_read_latency(fk, runner):
    """Compare latency: paged scatter-gather read vs contiguous read."""
    print("\n" + "=" * 80)
    print("  Benchmark 2: Read Latency — Paged Gather vs Contiguous")
    print("=" * 80)

    seq_lens_sweep = [256, 512, 1024, 2048]
    batch = 4
    results = []

    for seq_len in seq_lens_sweep:
        total_pages_needed = batch * ((seq_len + PAGE_SIZE - 1) // PAGE_SIZE)
        num_pages = total_pages_needed + 10
        pool = create_paged_cache(num_pages)

        seq_lens_list = [seq_len] * batch
        bt_tensor, seq_lens_t = fill_paged_cache_batch(fk, pool, seq_lens_list)

        # Paged read benchmark
        result_paged = runner.run(
            lambda: fk.paged_kv_cache_read(pool, bt_tensor, seq_lens_t, seq_len),
            name=f"paged_read_N={seq_len}",
        )
        results.append(result_paged)

        # Contiguous read baseline (just a tensor copy for comparison)
        K_contig = torch.randn(batch, NUM_HEADS, seq_len, HEAD_DIM,
                               dtype=torch.float16, device="cuda")
        V_contig = torch.randn(batch, NUM_HEADS, seq_len, HEAD_DIM,
                               dtype=torch.float16, device="cuda")
        result_contig = runner.run(
            lambda: (K_contig.clone(), V_contig.clone()),
            name=f"contig_read_N={seq_len}",
        )
        results.append(result_contig)

        del pool
        torch.cuda.empty_cache()

    compare_results(results)
    return results


# ─── Benchmark 3: Append Latency ────────────────────────────────────────────

def bench_append_latency(fk, runner):
    """Measure append latency for different token counts."""
    print("\n" + "=" * 80)
    print("  Benchmark 3: Append Latency")
    print("=" * 80)

    token_counts = [1, 16, 64, 128, 256]
    num_pages = 256
    results = []

    for T in token_counts:
        pool = create_paged_cache(num_pages)
        bt = [0]  # single page (we'll overwrite same slot for benchmarking)

        k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
        v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

        # Use sequential slots
        num_pages_needed = (T + PAGE_SIZE - 1) // PAGE_SIZE
        bt_list = list(range(num_pages_needed))
        slots = [bt_list[t // PAGE_SIZE] * PAGE_SIZE + t % PAGE_SIZE for t in range(T)]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")

        result = runner.run(
            lambda: fk.paged_kv_cache_append(pool, slot_mapping, k, v),
            name=f"append_T={T}",
        )
        results.append(result)
        del pool
        torch.cuda.empty_cache()

    compare_results(results)
    return results


# ─── Benchmark 4: CUDA vs Triton ────────────────────────────────────────────

def bench_cuda_vs_triton(fk, runner):
    """Compare CUDA and Triton implementations."""
    print("\n" + "=" * 80)
    print("  Benchmark 4: CUDA vs Triton")
    print("=" * 80)

    seq_len = 1024
    batch = 4
    total_pages_needed = batch * ((seq_len + PAGE_SIZE - 1) // PAGE_SIZE)
    num_pages = total_pages_needed + 10

    results = []

    # -- Append comparison --
    T = 64
    k = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
    v = torch.randn(T, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
    slots = list(range(T))
    slot_mapping = torch.tensor(slots, dtype=torch.int32, device="cuda")

    for backend, fn_name in [("CUDA", "paged_kv_cache_append"),
                              ("Triton", "triton_paged_kv_cache_append")]:
        pool = create_paged_cache(num_pages)
        fn = getattr(fk, fn_name)
        result = runner.run(
            lambda: fn(pool, slot_mapping, k, v),
            name=f"append_{backend}",
        )
        results.append(result)
        del pool
        torch.cuda.empty_cache()

    # -- Read comparison --
    for backend, use_triton in [("CUDA", False), ("Triton", True)]:
        pool = create_paged_cache(num_pages)
        seq_lens_list = [seq_len] * batch
        bt_tensor, seq_lens_t = fill_paged_cache_batch(
            fk, pool, seq_lens_list, use_triton=use_triton
        )

        read_fn = fk.triton_paged_kv_cache_read if use_triton else fk.paged_kv_cache_read
        result = runner.run(
            lambda: read_fn(pool, bt_tensor, seq_lens_t, seq_len),
            name=f"read_{backend}",
        )
        results.append(result)
        del pool
        torch.cuda.empty_cache()

    compare_results(results, baseline_name="read_CUDA")
    return results


# ─── Benchmark 5: Batch Sweep ───────────────────────────────────────────────

def bench_batch_sweep(fk, runner):
    """Read latency scaling with batch size."""
    print("\n" + "=" * 80)
    print("  Benchmark 5: Batch Sweep (N=1024)")
    print("=" * 80)

    seq_len = 1024
    batch_sizes = [1, 4, 8, 16]
    results = []

    for batch in batch_sizes:
        total_pages_needed = batch * ((seq_len + PAGE_SIZE - 1) // PAGE_SIZE)
        num_pages = total_pages_needed + 10
        pool = create_paged_cache(num_pages)

        seq_lens_list = [seq_len] * batch
        bt_tensor, seq_lens_t = fill_paged_cache_batch(fk, pool, seq_lens_list)

        result = runner.run(
            lambda: fk.paged_kv_cache_read(pool, bt_tensor, seq_lens_t, seq_len),
            name=f"read_B={batch}",
        )
        results.append(result)
        del pool
        torch.cuda.empty_cache()

    compare_results(results)
    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FlashKernel KV-Cache Benchmark")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--timed", type=int, default=200, help="Timed iterations")
    parser.add_argument("--csv", type=str, default="benchmarks/results/kv_cache.csv",
                        help="CSV output path")
    args = parser.parse_args()

    check_cuda()

    import flashkernel as fk

    runner = BenchmarkRunner(warmup=args.warmup, timed=args.timed)

    all_results = []

    # Memory analysis (no GPU timing)
    all_results.extend(bench_memory_savings(fk, runner))

    # Read latency
    all_results.extend(bench_read_latency(fk, runner))

    # Append latency
    all_results.extend(bench_append_latency(fk, runner))

    # CUDA vs Triton
    all_results.extend(bench_cuda_vs_triton(fk, runner))

    # Batch sweep
    all_results.extend(bench_batch_sweep(fk, runner))

    # Export to CSV
    csv_path = args.csv
    for i, result in enumerate(all_results):
        result.to_csv(csv_path, append=(i > 0))

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
