"""
FlashKernel — Benchmark harness (v1.0.0)

Provides a reusable timing framework for all kernel benchmarks.
This module is the foundation — every v1.0.x benchmark imports from here.

Usage:
    from benchmarks.harness import BenchmarkRunner

    runner = BenchmarkRunner(warmup=100, timed=1000)
    result = runner.run(lambda: my_kernel(a, b))
    result.print()
    result.to_csv("results/my_kernel.csv")
"""

import time
import csv
import os
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class BenchmarkResult:
    """Container for benchmark timing results."""

    name: str
    n_warmup: int
    n_timed: int
    times_ms: list[float] = field(default_factory=list)

    # Populated after run
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    peak_mem_mb: float = 0.0

    def compute_stats(self):
        """Compute statistics from raw timing data."""
        if not self.times_ms:
            return
        self.times_ms.sort()
        self.mean_ms = statistics.mean(self.times_ms)
        self.std_ms = statistics.stdev(self.times_ms) if len(self.times_ms) > 1 else 0.0
        self.min_ms = self.times_ms[0]
        self.max_ms = self.times_ms[-1]
        n = len(self.times_ms)
        self.p50_ms = self.times_ms[int(n * 0.50)]
        self.p95_ms = self.times_ms[int(n * 0.95)]
        self.p99_ms = self.times_ms[int(n * 0.99)]

    def print(self):
        """Print formatted results to stdout."""
        print(f"\n{'─' * 60}")
        print(f"  {self.name}")
        print(f"{'─' * 60}")
        print(f"  Warmup:  {self.n_warmup} iterations (discarded)")
        print(f"  Timed:   {self.n_timed} iterations")
        print(f"  Mean:    {self.mean_ms:.4f} ms  (±{self.std_ms:.4f})")
        print(f"  Min:     {self.min_ms:.4f} ms")
        print(f"  Max:     {self.max_ms:.4f} ms")
        print(f"  p50:     {self.p50_ms:.4f} ms")
        print(f"  p95:     {self.p95_ms:.4f} ms")
        print(f"  p99:     {self.p99_ms:.4f} ms")
        if self.peak_mem_mb > 0:
            print(f"  Peak mem: {self.peak_mem_mb:.1f} MB")
        print(f"{'─' * 60}\n")

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "name": self.name,
            "n_warmup": self.n_warmup,
            "n_timed": self.n_timed,
            "mean_ms": round(self.mean_ms, 4),
            "std_ms": round(self.std_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "peak_mem_mb": round(self.peak_mem_mb, 1),
        }

    def to_csv(self, path: str, append: bool = True):
        """Write result to CSV file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        row = self.to_dict()
        file_exists = os.path.exists(path)
        mode = "a" if (append and file_exists) else "w"
        with open(path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if mode == "w" or not file_exists:
                writer.writeheader()
            writer.writerow(row)


class BenchmarkRunner:
    """
    GPU-aware benchmark runner using CUDA events for accurate timing.

    Usage:
        runner = BenchmarkRunner(warmup=100, timed=1000)
        result = runner.run(lambda: kernel(a, b), name="my_kernel")
    """

    def __init__(self, warmup: int = 100, timed: int = 1000):
        self.warmup = warmup
        self.timed = timed

    def run(
        self,
        fn: Callable,
        name: str = "kernel",
        track_memory: bool = True,
    ) -> BenchmarkResult:
        """
        Time a kernel function using CUDA events.

        Args:
            fn: Callable that runs the kernel (no args — use closures).
            name: Human-readable name for the benchmark.
            track_memory: Whether to track peak GPU memory.

        Returns:
            BenchmarkResult with timing statistics.
        """
        assert torch.cuda.is_available(), "CUDA required for benchmarking"

        # Reset memory tracking
        if track_memory:
            torch.cuda.reset_peak_memory_stats()

        # Warmup — don't time these
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()

        # Timed iterations — use CUDA events for accurate GPU timing
        times = []
        for _ in range(self.timed):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            fn()
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # milliseconds

        result = BenchmarkResult(
            name=name,
            n_warmup=self.warmup,
            n_timed=self.timed,
            times_ms=times,
        )

        if track_memory:
            result.peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        result.compute_stats()
        return result


def compare_results(results: list[BenchmarkResult], baseline_name: Optional[str] = None):
    """
    Print a comparison table of multiple benchmark results.

    Args:
        results: List of BenchmarkResult objects to compare.
        baseline_name: Name of the baseline result for speedup calculation.
    """
    baseline_mean = None
    if baseline_name:
        for r in results:
            if r.name == baseline_name:
                baseline_mean = r.mean_ms
                break

    print(f"\n{'═' * 80}")
    print(f"  {'Name':<30} {'Mean (ms)':>10} {'Std':>10} {'p95':>10} {'Speedup':>10}")
    print(f"{'═' * 80}")

    for r in results:
        speedup = ""
        if baseline_mean and baseline_mean > 0:
            speedup = f"{baseline_mean / r.mean_ms:.2f}×"
        print(f"  {r.name:<30} {r.mean_ms:>10.4f} {r.std_ms:>10.4f} {r.p95_ms:>10.4f} {speedup:>10}")

    print(f"{'═' * 80}\n")
