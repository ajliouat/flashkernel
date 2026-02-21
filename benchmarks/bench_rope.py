"""
FlashKernel — RoPE Embedding Benchmark (v1.0.5)

Benchmarks:
  1. Sequence length sweep: N=[512, 1024, 2048, 4096], d=[64, 128]
  2. Table-lookup vs fused (on-the-fly sin/cos) comparison
  3. CUDA vs Triton comparison
  4. Batch sweep: B=[1, 4, 8] at N=1024
  5. Effective bandwidth calculation

The key metric is whether the fused variant (no table reads) is faster
than the table-lookup variant, and how our kernels compare to a pure
PyTorch reference.

Outputs:
  - Console tables with timing and bandwidth
  - benchmarks/results/rope.csv
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


# ─── Reference implementations ──────────────────────────────────────────────

def reference_rope(Q, K, base=10000.0):
    """
    Pure PyTorch RoPE — the baseline.
    Q, K: [B, H, N, D] fp16
    """
    B, H, N, D = Q.shape
    half_dim = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32, device=Q.device) / D))
    positions = torch.arange(N, dtype=torch.float32, device=Q.device)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [N, half_dim]
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # [1, 1, N, half_dim]
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    def apply(x):
        x_fp32 = x.float()
        x0 = x_fp32[..., 0::2]
        x1 = x_fp32[..., 1::2]
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        return torch.stack([y0, y1], dim=-1).flatten(-2).half()

    return apply(Q), apply(K)


def torch_compile_rope(Q, K, base=10000.0):
    """torch.compile version — let compiler optimize."""
    return _compiled_rope(Q, K, base)


@torch.compile(mode="reduce-overhead")
def _compiled_rope(Q, K, base):
    B, H, N, D = Q.shape
    half_dim = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32, device=Q.device) / D))
    positions = torch.arange(N, dtype=torch.float32, device=Q.device)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    def apply(x):
        x_fp32 = x.float()
        x0 = x_fp32[..., 0::2]
        x1 = x_fp32[..., 1::2]
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        return torch.stack([y0, y1], dim=-1).flatten(-2).half()

    return apply(Q), apply(K)


# ─── Benchmark helpers ───────────────────────────────────────────────────────

def effective_bandwidth_gbps(B, H, N, D, time_ms):
    """
    Compute effective bandwidth for RoPE.
    Reads: Q + K (fp16), cos + sin table (fp32)
    Writes: Q_rot + K_rot (fp16)
    """
    qk_bytes = 2 * B * H * N * D * 2  # Q + K, fp16
    table_bytes = N * (D // 2) * 4 * 2  # cos + sin tables, fp32
    total_bytes = qk_bytes * 2 + table_bytes  # read + write Q/K + read tables
    return total_bytes / (time_ms / 1000.0) / 1e9


# ─── Main benchmark functions ───────────────────────────────────────────────

def bench_seq_sweep(runner, csv_path):
    """Benchmark across sequence lengths for both head dims."""
    import flashkernel as fk
    from src.triton.rope import (
        triton_rope_precompute_freqs, triton_rope_forward,
        triton_rope_forward_fused,
    )

    print("\n" + "=" * 80)
    print("  BENCHMARK 1: Sequence Length Sweep")
    print("=" * 80)

    for D in [64, 128]:
        print(f"\n--- head_dim = {D} ---")

        for N in [512, 1024, 2048, 4096]:
            B, H = 4, 8
            Q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
            K = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")

            results = []

            # PyTorch reference
            result = runner.run(
                lambda: reference_rope(Q, K),
                name=f"PyTorch_N{N}_D{D}"
            )
            result.extra = {"backend": "pytorch", "B": B, "H": H, "N": N, "D": D}
            result.to_csv(csv_path, append=True)
            results.append(result)

            # CUDA table-lookup
            cos, sin = fk.rope_precompute_freqs(N, D)
            result = runner.run(
                lambda: fk.rope_forward(Q, K, cos, sin),
                name=f"CUDA_table_N{N}_D{D}"
            )
            result.extra = {"backend": "cuda_table", "B": B, "H": H, "N": N, "D": D}
            result.to_csv(csv_path, append=True)
            results.append(result)

            # CUDA fused
            result = runner.run(
                lambda: fk.rope_forward_fused(Q, K),
                name=f"CUDA_fused_N{N}_D{D}"
            )
            result.extra = {"backend": "cuda_fused", "B": B, "H": H, "N": N, "D": D}
            result.to_csv(csv_path, append=True)
            results.append(result)

            # Triton table-lookup
            cos_t, sin_t = triton_rope_precompute_freqs(N, D)
            result = runner.run(
                lambda: triton_rope_forward(Q.clone(), K.clone(), cos_t, sin_t),
                name=f"Triton_table_N{N}_D{D}"
            )
            result.extra = {"backend": "triton_table", "B": B, "H": H, "N": N, "D": D}
            result.to_csv(csv_path, append=True)
            results.append(result)

            # Triton fused
            result = runner.run(
                lambda: triton_rope_forward_fused(Q.clone(), K.clone()),
                name=f"Triton_fused_N{N}_D{D}"
            )
            result.extra = {"backend": "triton_fused", "B": B, "H": H, "N": N, "D": D}
            result.to_csv(csv_path, append=True)
            results.append(result)

            compare_results(results, baseline_name=f"PyTorch_N{N}_D{D}")


def bench_batch_sweep(runner, csv_path):
    """Benchmark across batch sizes."""
    import flashkernel as fk

    print("\n" + "=" * 80)
    print("  BENCHMARK 2: Batch Size Sweep (N=1024)")
    print("=" * 80)

    for D in [64, 128]:
        print(f"\n--- head_dim = {D} ---")

        for B in [1, 4, 8]:
            H, N = 8, 1024
            Q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
            K = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")

            results = []

            result = runner.run(
                lambda: reference_rope(Q, K),
                name=f"PyTorch_B{B}_D{D}"
            )
            results.append(result)

            cos, sin = fk.rope_precompute_freqs(N, D)
            result = runner.run(
                lambda: fk.rope_forward(Q, K, cos, sin),
                name=f"CUDA_table_B{B}_D{D}"
            )
            results.append(result)

            result = runner.run(
                lambda: fk.rope_forward_fused(Q, K),
                name=f"CUDA_fused_B{B}_D{D}"
            )
            results.append(result)

            compare_results(results, baseline_name=f"PyTorch_B{B}_D{D}")


def bench_table_vs_fused(runner, csv_path):
    """Direct comparison: table-lookup vs fused for CUDA and Triton."""
    import flashkernel as fk
    from src.triton.rope import (
        triton_rope_precompute_freqs, triton_rope_forward,
        triton_rope_forward_fused,
    )

    print("\n" + "=" * 80)
    print("  BENCHMARK 3: Table vs Fused Comparison")
    print("=" * 80)

    configs = [
        (1, 8, 512, 64),
        (4, 8, 1024, 64),
        (4, 8, 2048, 64),
        (4, 12, 1024, 128),
        (4, 12, 2048, 128),
    ]

    for B, H, N, D in configs:
        print(f"\n--- B={B}, H={H}, N={N}, D={D} ---")
        Q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        K = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
        results = []

        # CUDA table
        cos, sin = fk.rope_precompute_freqs(N, D)
        result = runner.run(
            lambda: fk.rope_forward(Q, K, cos, sin),
            name="CUDA_table"
        )
        results.append(result)

        # CUDA fused
        result = runner.run(
            lambda: fk.rope_forward_fused(Q, K),
            name="CUDA_fused"
        )
        results.append(result)

        # Triton table
        cos_t, sin_t = triton_rope_precompute_freqs(N, D)
        result = runner.run(
            lambda: triton_rope_forward(Q.clone(), K.clone(), cos_t, sin_t),
            name="Triton_table"
        )
        results.append(result)

        # Triton fused
        result = runner.run(
            lambda: triton_rope_forward_fused(Q.clone(), K.clone()),
            name="Triton_fused"
        )
        results.append(result)

        compare_results(results, baseline_name="CUDA_table")

        # Print bandwidth
        for r in results:
            bw = effective_bandwidth_gbps(B, H, N, D, r.mean_ms)
            print(f"  {r.name:<20s} effective bandwidth: {bw:.1f} GB/s")


def bench_summary(runner):
    """Print ROADMAP-format summary table."""
    import flashkernel as fk

    print("\n" + "=" * 80)
    print("  SUMMARY TABLE (ROADMAP format)")
    print("=" * 80)

    header = f"  {'Backend':<25s}"
    for N in [512, 1024, 2048, 4096]:
        header += f" {'N='+str(N):>10s}"
    print(header)
    print("  " + "─" * 75)

    B, H, D = 4, 8, 64
    backends = {
        "PyTorch eager": lambda Q, K: reference_rope(Q, K),
        "CUDA table": None,  # set below per N
        "CUDA fused": lambda Q, K: fk.rope_forward_fused(Q, K),
    }

    for name in backends:
        row = f"  {name:<25s}"
        for N in [512, 1024, 2048, 4096]:
            Q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
            K = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
            if name == "CUDA table":
                cos, sin = fk.rope_precompute_freqs(N, D)
                fn = lambda: fk.rope_forward(Q, K, cos, sin)
            elif name == "PyTorch eager":
                fn = lambda: reference_rope(Q, K)
            else:
                fn = lambda: fk.rope_forward_fused(Q, K)
            result = runner.run(fn, name=f"{name}_N{N}", track_memory=False)
            row += f" {result.mean_ms:>9.3f}ms"
        print(row)

    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FlashKernel RoPE Benchmarks")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer iterations")
    args = parser.parse_args()

    if args.quick:
        args.warmup = 10
        args.timed = 50

    check_cuda()

    runner = BenchmarkRunner(warmup=args.warmup, timed=args.timed)
    csv_path = os.path.join(ROOT, "benchmarks", "results", "rope.csv")

    # Clear previous results
    if os.path.exists(csv_path):
        os.remove(csv_path)

    bench_seq_sweep(runner, csv_path)
    bench_batch_sweep(runner, csv_path)
    bench_table_vs_fused(runner, csv_path)
    bench_summary(runner)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
