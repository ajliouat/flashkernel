"""
FlashKernel — Reduction kernel benchmarks (v1.0.1)

Benchmarks:
  1. Full sum reduction: CUDA vs Triton vs PyTorch torch.sum
  2. Full max reduction: CUDA vs Triton vs PyTorch torch.max
  3. Row-wise sum: CUDA vs Triton vs PyTorch, varying (rows, cols)

Sweep: 1K → 100M elements (powers of 10)

Outputs:
  - Console table
  - benchmarks/results/reduce.csv
"""

import os
import sys
import argparse

import torch

# Add project root for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks.harness import BenchmarkRunner, BenchmarkResult, compare_results


def check_cuda():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Benchmarks require a GPU.")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# FULL SUM REDUCTION BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════

def bench_reduce_sum(warmup: int = 100, iters: int = 1000):
    """Benchmark full sum reduction across all three backends."""
    import flashkernel
    from src.triton.reduce import triton_reduce_sum

    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    results = []

    runner = BenchmarkRunner(warmup=warmup, timed=iters)

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        # PyTorch baseline
        def pytorch_fn():
            return x.sum()
        pt_result = runner.run(pytorch_fn)
        pt_result.name = f"torch.sum n={n}"

        # CUDA kernel
        def cuda_fn():
            return flashkernel.reduce_sum(x)
        cuda_result = runner.run(cuda_fn)
        cuda_result.name = f"CUDA reduce n={n}"

        # Triton kernel
        def triton_fn():
            return triton_reduce_sum(x)
        triton_result = runner.run(triton_fn)
        triton_result.name = f"Triton reduce n={n}"

        # Effective bandwidth: read n floats
        bytes_read = n * 4  # fp32
        for r in [pt_result, cuda_result, triton_result]:
            r.extra["n"] = n
            r.extra["dtype"] = "fp32"
            r.extra["op"] = "sum"
            r.extra["bandwidth_gb_s"] = (bytes_read / r.mean_ms / 1e6)

        results.extend([pt_result, cuda_result, triton_result])

        print(f"\n--- n = {n:,} ---")
        compare_results([pt_result, cuda_result, triton_result])

    return results


# ═════════════════════════════════════════════════════════════════════════════
# FULL MAX REDUCTION BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════

def bench_reduce_max(warmup: int = 100, iters: int = 1000):
    """Benchmark full max reduction."""
    import flashkernel
    from src.triton.reduce import triton_reduce_max

    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    results = []

    runner = BenchmarkRunner(warmup=warmup, timed=iters)

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        def pytorch_fn():
            return x.max()
        pt_result = runner.run(pytorch_fn)
        pt_result.name = f"torch.max n={n}"

        def cuda_fn():
            return flashkernel.reduce_max(x)
        cuda_result = runner.run(cuda_fn)
        cuda_result.name = f"CUDA max n={n}"

        def triton_fn():
            return triton_reduce_max(x)
        triton_result = runner.run(triton_fn)
        triton_result.name = f"Triton max n={n}"

        bytes_read = n * 4
        for r in [pt_result, cuda_result, triton_result]:
            r.extra["n"] = n
            r.extra["dtype"] = "fp32"
            r.extra["op"] = "max"
            r.extra["bandwidth_gb_s"] = (bytes_read / r.mean_ms / 1e6)

        results.extend([pt_result, cuda_result, triton_result])

        print(f"\n--- max n = {n:,} ---")
        compare_results([pt_result, cuda_result, triton_result])

    return results


# ═════════════════════════════════════════════════════════════════════════════
# ROW-WISE SUM BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════

def bench_reduce_rows(warmup: int = 100, iters: int = 1000):
    """Benchmark row-wise sum reduction."""
    import flashkernel
    from src.triton.reduce import triton_reduce_sum_rows

    shapes = [
        (128, 4096),     # typical attention row
        (512, 4096),     # batch of attention rows
        (1024, 2048),    # medium
        (4096, 1024),    # many short rows
        (8192, 512),     # many very short rows
    ]
    results = []

    runner = BenchmarkRunner(warmup=warmup, timed=iters)

    for rows, cols in shapes:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        def pytorch_fn():
            return x.sum(dim=-1)
        pt_result = runner.run(pytorch_fn)
        pt_result.name = f"torch.sum(dim=-1) {rows}x{cols}"

        def cuda_fn():
            return flashkernel.reduce_sum(x, dim=1)
        cuda_result = runner.run(cuda_fn)
        cuda_result.name = f"CUDA rows {rows}x{cols}"

        def triton_fn():
            return triton_reduce_sum_rows(x, dim=-1)
        triton_result = runner.run(triton_fn)
        triton_result.name = f"Triton rows {rows}x{cols}"

        bytes_read = rows * cols * 4
        for r in [pt_result, cuda_result, triton_result]:
            r.extra["rows"] = rows
            r.extra["cols"] = cols
            r.extra["dtype"] = "fp32"
            r.extra["op"] = "sum_rows"
            r.extra["bandwidth_gb_s"] = (bytes_read / r.mean_ms / 1e6)

        results.extend([pt_result, cuda_result, triton_result])

        print(f"\n--- rows {rows}x{cols} ---")
        compare_results([pt_result, cuda_result, triton_result])

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="FlashKernel reduction benchmarks")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=1000, help="Timed iterations")
    parser.add_argument("--output", type=str,
                        default=os.path.join(ROOT, "benchmarks", "results", "reduce.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    check_cuda()

    all_results = []

    print("=" * 70)
    print("FULL SUM REDUCTION")
    print("=" * 70)
    all_results.extend(bench_reduce_sum(args.warmup, args.iters))

    print("\n" + "=" * 70)
    print("FULL MAX REDUCTION")
    print("=" * 70)
    all_results.extend(bench_reduce_max(args.warmup, args.iters))

    print("\n" + "=" * 70)
    print("ROW-WISE SUM REDUCTION")
    print("=" * 70)
    all_results.extend(bench_reduce_rows(args.warmup, args.iters))

    # Export to CSV
    if all_results:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        all_results[0].to_csv(args.output, all_results)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
