"""
FlashKernel — Stub benchmark (v1.0.0)

Benchmarks the trivial vector_add kernel to verify the timing harness works.
This also establishes the HBM bandwidth baseline for the T4.

Usage:
    python benchmarks/bench_stub.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from benchmarks.harness import BenchmarkRunner, compare_results


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping benchmarks")
        return

    device = torch.device("cuda")
    runner = BenchmarkRunner(warmup=100, timed=1000)
    results = []

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    # ─── Sweep over sizes ────────────────────────────────────────────────
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    for n in sizes:
        a = torch.randn(n, device=device, dtype=torch.float32)
        b = torch.randn(n, device=device, dtype=torch.float32)

        # Baseline: PyTorch eager
        result_pt = runner.run(
            lambda: torch.add(a, b),
            name=f"torch.add n={n:_}",
        )
        results.append(result_pt)

        # Our kernel
        import flashkernel
        result_fk = runner.run(
            lambda: flashkernel.vector_add(a, b),
            name=f"flashkernel.add n={n:_}",
        )
        results.append(result_fk)

        # Bandwidth calculation
        bytes_rw = n * 4 * 3  # read a + read b + write c, float32 = 4 bytes
        bandwidth_gbps = (bytes_rw / 1e9) / (result_fk.mean_ms / 1e3)
        print(f"n={n:>12_}  ours={result_fk.mean_ms:.4f}ms  "
              f"torch={result_pt.mean_ms:.4f}ms  "
              f"bandwidth={bandwidth_gbps:.1f} GB/s")

    # ─── Comparison table ────────────────────────────────────────────────
    compare_results(results)

    # ─── Save CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), "results", "stub.csv")
    for r in results:
        r.to_csv(csv_path)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
