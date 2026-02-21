"""
FlashKernel — Fused GeLU+Linear Benchmark (v1.0.4)

Benchmarks:
  1. Dimension sweep: M=[128,512,2048], N=[768,3072], K=[768,3072]
  2. Fused vs unfused comparison (key metric: speedup from fusion)
  3. CUDA vs Triton comparison
  4. GeLU variant comparison (exact vs tanh)
  5. Memory usage tracking

The primary goal is demonstrating >=1.3x speedup from fusion on at least one config.

Outputs:
  - Console tables with speedup ratios
  - benchmarks/results/fused_gelu.csv
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


# ─── Reference implementations ──────────────────────────────────────────────

def unfused_gelu_linear(X, W, bias, approximate="none"):
    """Unfused: F.linear + F.gelu (2 kernels, 2 HBM round-trips)."""
    out = F.linear(X, W, bias)
    return F.gelu(out, approximate=approximate)


def torch_compile_fused(X, W, bias, approximate="none"):
    """torch.compile version — let compiler find the fusion."""
    return _compiled_gelu_linear(X, W, bias, approximate)


@torch.compile(mode="reduce-overhead")
def _compiled_gelu_linear(X, W, bias, approximate):
    out = F.linear(X, W, bias)
    return F.gelu(out, approximate=approximate)


# ═════════════════════════════════════════════════════════════════════════════
# DIMENSION SWEEP — FUSED vs UNFUSED
# ═════════════════════════════════════════════════════════════════════════════

def bench_fused_vs_unfused(runner, M, N, K):
    """Compare fused CUDA kernel vs unfused PyTorch for one config."""
    import flashkernel

    X = torch.randn(M, K, dtype=torch.float16, device="cuda")
    W = torch.randn(N, K, dtype=torch.float16, device="cuda")
    bias = torch.randn(N, dtype=torch.float16, device="cuda")

    results = []

    # 1. Unfused PyTorch (2 kernels)
    r = runner.run(
        lambda: unfused_gelu_linear(X, W, bias),
        name=f"unfused M={M} N={N} K={K}",
    )
    r.extra["backend"] = "unfused"
    r.extra["M"] = M
    r.extra["N"] = N
    r.extra["K"] = K
    results.append(r)

    # 2. torch.compile
    # Warmup compile
    _compiled_gelu_linear(X, W, bias, "none")
    torch.cuda.synchronize()
    r = runner.run(
        lambda: torch_compile_fused(X, W, bias),
        name=f"compile M={M} N={N} K={K}",
    )
    r.extra["backend"] = "torch.compile"
    r.extra["M"] = M
    r.extra["N"] = N
    r.extra["K"] = K
    results.append(r)

    # 3. CUDA fused kernel
    r = runner.run(
        lambda: flashkernel.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False),
        name=f"cuda_fused M={M} N={N} K={K}",
    )
    r.extra["backend"] = "cuda_fused"
    r.extra["M"] = M
    r.extra["N"] = N
    r.extra["K"] = K
    results.append(r)

    # 4. Triton fused kernel
    r = runner.run(
        lambda: flashkernel.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False),
        name=f"triton_fused M={M} N={N} K={K}",
    )
    r.extra["backend"] = "triton_fused"
    r.extra["M"] = M
    r.extra["N"] = N
    r.extra["K"] = K
    results.append(r)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# GeLU VARIANT COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def bench_gelu_variants(runner, M=512, N=3072, K=768):
    """Compare exact vs tanh GeLU for CUDA and Triton fused kernels."""
    import flashkernel

    X = torch.randn(M, K, dtype=torch.float16, device="cuda")
    W = torch.randn(N, K, dtype=torch.float16, device="cuda")
    bias = torch.randn(N, dtype=torch.float16, device="cuda")

    results = []

    for backend_name, fn in [
        ("cuda_exact", lambda: flashkernel.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)),
        ("cuda_tanh", lambda: flashkernel.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)),
        ("triton_exact", lambda: flashkernel.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=False)),
        ("triton_tanh", lambda: flashkernel.triton_fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)),
    ]:
        r = runner.run(fn, name=f"{backend_name} M={M} N={N} K={K}")
        r.extra["backend"] = backend_name
        r.extra["M"] = M
        r.extra["N"] = N
        r.extra["K"] = K
        results.append(r)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fused GeLU+Linear benchmarks")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--timed", type=int, default=200, help="Timed iterations")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results/fused_gelu.csv",
        help="CSV output path",
    )
    args = parser.parse_args()

    check_cuda()
    runner = BenchmarkRunner(warmup=args.warmup, timed=args.timed)

    all_results = []

    # ── 1. Dimension sweep: fused vs unfused ─────────────────────────────────
    print("=" * 70)
    print("  FUSED vs UNFUSED — Dimension Sweep")
    print("=" * 70)
    print()

    # ROADMAP-specified dimensions
    sweep_configs = [
        (128, 768, 768),
        (128, 3072, 768),
        (128, 768, 3072),
        (512, 768, 768),
        (512, 3072, 768),
        (512, 768, 3072),
        (2048, 768, 768),
        (2048, 3072, 768),
        (2048, 768, 3072),
        (2048, 3072, 3072),
    ]

    for M, N, K in sweep_configs:
        print(f"\n--- M={M}, N={N}, K={K} ---")
        try:
            results = bench_fused_vs_unfused(runner, M, N, K)
            compare_results(results, baseline_name=f"unfused M={M} N={N} K={K}")
            all_results.extend(results)
        except Exception as e:
            print(f"  SKIPPED: {e}")
            continue

    # ── 2. GeLU variant comparison ───────────────────────────────────────────
    print()
    print("=" * 70)
    print("  GeLU VARIANT COMPARISON (exact vs tanh)")
    print("=" * 70)
    print()

    try:
        variant_results = bench_gelu_variants(runner)
        compare_results(variant_results, baseline_name="cuda_exact M=512 N=3072 K=768")
        all_results.extend(variant_results)
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # ── 3. Summary table ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SUMMARY — Best Speedup from Fusion")
    print("=" * 70)
    print()

    # Find best speedup across all configs
    best_speedup = 0.0
    best_config = ""
    for r in all_results:
        if r.extra.get("backend") == "unfused":
            M, N, K = r.extra["M"], r.extra["N"], r.extra["K"]
            unfused_mean = r.mean_ms
            # Find matching CUDA fused
            for r2 in all_results:
                if (r2.extra.get("backend") == "cuda_fused" and
                    r2.extra.get("M") == M and r2.extra.get("N") == N and r2.extra.get("K") == K):
                    speedup = unfused_mean / r2.mean_ms if r2.mean_ms > 0 else 0
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_config = f"M={M} N={N} K={K}"

    if best_speedup > 0:
        print(f"  Best CUDA fusion speedup: {best_speedup:.2f}x ({best_config})")
        if best_speedup >= 1.3:
            print("  STATUS: PASS (>= 1.3x target)")
        else:
            print(f"  STATUS: Below 1.3x target (got {best_speedup:.2f}x)")
    print()

    # ── 4. CSV export ────────────────────────────────────────────────────────
    if all_results:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        for i, r in enumerate(all_results):
            r.to_csv(args.output, append=(i > 0))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
