"""
FlashKernel — FlashAttention Benchmark (v1.0.2)

Benchmarks:
  1. Latency sweep: seq=[128, 256, 512, 1024, 2048, 4096], head_dim=[64, 128]
  2. Batch sweep: batch=[1, 4, 8], fixed seq=1024
  3. Comparison: FlashKernel vs PyTorch SDPA vs naive matmul attention
  4. Memory usage: peak GPU memory per config

Outputs:
  - Console tables
  - benchmarks/results/attention_cuda.csv
"""

import os
import sys
import math
import argparse

import torch
import torch.nn.functional as F

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


def naive_attention(Q, K, V, scale):
    """Standard O(N²) attention — materializes the full attention matrix."""
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, V.float())
    return out.half()


def get_memory_mb():
    """Get current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / (1024 * 1024)


def get_peak_memory_mb():
    """Get peak GPU memory allocated in MB."""
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


# ═════════════════════════════════════════════════════════════════════════════
# SEQUENCE LENGTH SWEEP
# ═════════════════════════════════════════════════════════════════════════════

def bench_seq_sweep(warmup=50, iters=200):
    """Sweep sequence lengths for FlashKernel vs PyTorch SDPA."""
    import flashkernel

    configs = [
        # (B, H, N, D)
        (1, 8, 128, 64),
        (1, 8, 256, 64),
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (1, 8, 4096, 64),
        # head_dim=128
        (1, 8, 128, 128),
        (1, 8, 256, 128),
        (1, 8, 512, 128),
        (1, 8, 1024, 128),
        (1, 8, 2048, 128),
    ]

    results = []
    runner = BenchmarkRunner(warmup=warmup, timed=iters)

    for B, H, N, D in configs:
        scale = 1.0 / math.sqrt(D)
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        # FlashKernel
        torch.cuda.reset_peak_memory_stats()
        def fk_fn():
            return flashkernel.flash_attention_forward(Q, K, V, scale=scale)
        fk_result = runner.run(fk_fn)
        fk_peak = get_peak_memory_mb()
        fk_result.name = f"FlashKernel B={B} H={H} N={N} D={D}"

        # PyTorch SDPA
        torch.cuda.reset_peak_memory_stats()
        def sdpa_fn():
            return F.scaled_dot_product_attention(
                Q.float(), K.float(), V.float(), scale=scale
            )
        sdpa_result = runner.run(sdpa_fn)
        sdpa_peak = get_peak_memory_mb()
        sdpa_result.name = f"PyTorch SDPA B={B} H={H} N={N} D={D}"

        # Naive (only for small N to avoid OOM)
        if N <= 2048:
            torch.cuda.reset_peak_memory_stats()
            def naive_fn():
                return naive_attention(Q, K, V, scale)
            naive_result = runner.run(naive_fn)
            naive_peak = get_peak_memory_mb()
            naive_result.name = f"Naive O(N²) B={B} H={H} N={N} D={D}"
            batch_results = [fk_result, sdpa_result, naive_result]
        else:
            batch_results = [fk_result, sdpa_result]

        results.extend(batch_results)

        print(f"\n--- B={B}, H={H}, N={N}, D={D} ---")
        compare_results(batch_results)
        print(f"  Peak memory: FK={fk_peak:.0f}MB, SDPA={sdpa_peak:.0f}MB")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# BATCH SIZE SWEEP
# ═════════════════════════════════════════════════════════════════════════════

def bench_batch_sweep(warmup=50, iters=200):
    """Sweep batch sizes at fixed seq_len=1024."""
    import flashkernel

    configs = [
        (1, 8, 1024, 64),
        (4, 8, 1024, 64),
        (8, 8, 1024, 64),
        (1, 12, 1024, 128),
        (4, 12, 1024, 128),
        (8, 12, 1024, 128),
    ]

    results = []
    runner = BenchmarkRunner(warmup=warmup, timed=iters)

    for B, H, N, D in configs:
        scale = 1.0 / math.sqrt(D)
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        def fk_fn():
            return flashkernel.flash_attention_forward(Q, K, V, scale=scale)
        fk_result = runner.run(fk_fn)
        fk_result.name = f"FlashKernel B={B} H={H} N={N} D={D}"

        def sdpa_fn():
            return F.scaled_dot_product_attention(
                Q.float(), K.float(), V.float(), scale=scale
            )
        sdpa_result = runner.run(sdpa_fn)
        sdpa_result.name = f"PyTorch SDPA B={B} H={H} N={N} D={D}"

        results.extend([fk_result, sdpa_result])

        print(f"\n--- B={B}, H={H}, N={N}, D={D} ---")
        compare_results([fk_result, sdpa_result])

    return results


# ═════════════════════════════════════════════════════════════════════════════
# CAUSAL vs NON-CAUSAL
# ═════════════════════════════════════════════════════════════════════════════

def bench_causal(warmup=50, iters=200):
    """Compare causal vs non-causal FlashAttention."""
    import flashkernel

    configs = [
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
    ]

    results = []
    runner = BenchmarkRunner(warmup=warmup, timed=iters)

    for B, H, N, D in configs:
        scale = 1.0 / math.sqrt(D)
        Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
        V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

        def fk_noncausal():
            return flashkernel.flash_attention_forward(Q, K, V, scale=scale, is_causal=False)
        nc_result = runner.run(fk_noncausal)
        nc_result.name = f"Non-causal N={N}"

        def fk_causal():
            return flashkernel.flash_attention_forward(Q, K, V, scale=scale, is_causal=True)
        c_result = runner.run(fk_causal)
        c_result.name = f"Causal N={N}"

        results.extend([nc_result, c_result])

        print(f"\n--- N={N} causal comparison ---")
        compare_results([nc_result, c_result])
        speedup = nc_result.mean_ms / c_result.mean_ms if c_result.mean_ms > 0 else 0
        print(f"  Causal speedup: {speedup:.2f}x (expected ~2x for large N)")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FlashKernel attention benchmarks")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", type=str,
                        default=os.path.join(ROOT, "benchmarks", "results", "attention_cuda.csv"))
    args = parser.parse_args()

    check_cuda()

    all_results = []

    print("=" * 70)
    print("SEQUENCE LENGTH SWEEP")
    print("=" * 70)
    all_results.extend(bench_seq_sweep(args.warmup, args.iters))

    print("\n" + "=" * 70)
    print("BATCH SIZE SWEEP")
    print("=" * 70)
    all_results.extend(bench_batch_sweep(args.warmup, args.iters))

    print("\n" + "=" * 70)
    print("CAUSAL vs NON-CAUSAL")
    print("=" * 70)
    all_results.extend(bench_causal(args.warmup, args.iters))

    # Export
    if all_results:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        for i, r in enumerate(all_results):
            r.to_csv(args.output, append=(i > 0))
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
