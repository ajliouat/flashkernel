"""
FlashKernel — Head-to-Head Attention Benchmark (v1.0.3)

Four-way comparison:
  1. PyTorch eager  — F.scaled_dot_product_attention
  2. torch.compile  — compiled SDPA
  3. Triton (ours)  — Triton FlashAttention from src/triton/flash_attention.py
  4. CUDA (ours)    — CUDA FlashAttention from flashkernel C++ extension

Sweeps:
  - Sequence length: N = [512, 1024, 2048, 4096]
  - Head dimensions: D = [64, 128]
  - Causal vs non-causal
  - Batch sizes: B = [1, 4, 8]

Output:
  - Console comparison table (ROADMAP format)
  - benchmarks/results/attention_comparison.csv
"""

import os
import sys
import csv
import math
import argparse

import torch
import torch.nn.functional as F

# Add project root for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks.harness import BenchmarkRunner


# ─── Backend Helpers ─────────────────────────────────────────────────────────

def make_qkv(B, H, N, D, device="cuda", dtype=torch.float16):
    Q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    K = torch.randn(B, H, N, D, device=device, dtype=dtype)
    V = torch.randn(B, H, N, D, device=device, dtype=dtype)
    return Q, K, V


def get_peak_memory_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def run_pytorch_eager(Q, K, V, scale, is_causal=False):
    """PyTorch eager SDPA."""
    return F.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(),
        attn_mask=None,
        is_causal=is_causal,
        scale=scale,
    )


def get_compiled_sdpa():
    """Return compiled SDPA function."""
    @torch.compile
    def compiled_sdpa(Q, K, V, scale, is_causal):
        return F.scaled_dot_product_attention(
            Q.float(), K.float(), V.float(),
            attn_mask=None,
            is_causal=is_causal,
            scale=scale,
        )
    return compiled_sdpa


def get_triton_attn():
    """Import Triton FlashAttention."""
    try:
        from src.triton.flash_attention import triton_flash_attention_forward
        return triton_flash_attention_forward
    except ImportError:
        from flashkernel._triton.flash_attention import triton_flash_attention_forward
        return triton_flash_attention_forward


def get_cuda_attn():
    """Import CUDA FlashAttention."""
    try:
        import flashkernel
        return flashkernel.flash_attention_forward
    except (ImportError, RuntimeError):
        return None


# ─── CSV Export ──────────────────────────────────────────────────────────────

def write_csv(path, rows):
    """Write list of dicts to CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Collect all keys
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ═════════════════════════════════════════════════════════════════════════════
# SEQUENCE LENGTH SWEEP
# ═════════════════════════════════════════════════════════════════════════════

def bench_seq_sweep(runner, warmup, iters):
    """
    Sweep sequence lengths: 4-way comparison.
    Format from ROADMAP:
      | Backend       | seq=512 | seq=1024 | seq=2048 | seq=4096 | Peak Mem |
    """
    triton_attn = get_triton_attn()
    cuda_attn = get_cuda_attn()
    compiled_sdpa = get_compiled_sdpa()

    configs = []
    for D in [64, 128]:
        for N in [512, 1024, 2048, 4096]:
            configs.append((1, 8, N, D))

    results = []

    for B, H, N, D in configs:
        scale = 1.0 / math.sqrt(D)
        Q, K, V = make_qkv(B, H, N, D)

        row_base = {"B": B, "H": H, "N": N, "D": D, "causal": False}

        # 1. PyTorch eager
        torch.cuda.reset_peak_memory_stats()
        try:
            r = runner.run(
                lambda: run_pytorch_eager(Q, K, V, scale),
                name=f"PyTorch eager N={N} D={D}",
            )
            results.append({
                **row_base,
                "backend": "pytorch_eager",
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "p95_ms": r.p95_ms,
                "peak_mem_mb": get_peak_memory_mb(),
            })
        except RuntimeError as e:
            results.append({
                **row_base,
                "backend": "pytorch_eager",
                "mean_ms": -1, "std_ms": 0, "p95_ms": 0,
                "peak_mem_mb": 0, "error": str(e)[:50],
            })

        # 2. torch.compile
        torch.cuda.reset_peak_memory_stats()
        try:
            r = runner.run(
                lambda: compiled_sdpa(Q, K, V, scale, False),
                name=f"torch.compile N={N} D={D}",
            )
            results.append({
                **row_base,
                "backend": "torch_compile",
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "p95_ms": r.p95_ms,
                "peak_mem_mb": get_peak_memory_mb(),
            })
        except Exception as e:
            results.append({
                **row_base,
                "backend": "torch_compile",
                "mean_ms": -1, "std_ms": 0, "p95_ms": 0,
                "peak_mem_mb": 0, "error": str(e)[:50],
            })

        # 3. Triton (ours)
        torch.cuda.reset_peak_memory_stats()
        try:
            r = runner.run(
                lambda: triton_attn(Q, K, V, scale=scale),
                name=f"Triton (ours) N={N} D={D}",
            )
            results.append({
                **row_base,
                "backend": "triton_ours",
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "p95_ms": r.p95_ms,
                "peak_mem_mb": get_peak_memory_mb(),
            })
        except Exception as e:
            results.append({
                **row_base,
                "backend": "triton_ours",
                "mean_ms": -1, "std_ms": 0, "p95_ms": 0,
                "peak_mem_mb": 0, "error": str(e)[:50],
            })

        # 4. CUDA (ours)
        if cuda_attn is not None:
            torch.cuda.reset_peak_memory_stats()
            try:
                r = runner.run(
                    lambda: cuda_attn(Q, K, V, scale=scale),
                    name=f"CUDA (ours) N={N} D={D}",
                )
                results.append({
                    **row_base,
                    "backend": "cuda_ours",
                    "mean_ms": r.mean_ms,
                    "std_ms": r.std_ms,
                    "p95_ms": r.p95_ms,
                    "peak_mem_mb": get_peak_memory_mb(),
                })
            except Exception as e:
                results.append({
                    **row_base,
                    "backend": "cuda_ours",
                    "mean_ms": -1, "std_ms": 0, "p95_ms": 0,
                    "peak_mem_mb": 0, "error": str(e)[:50],
                })

        # Print this config
        config_results = [r for r in results if r["N"] == N and r["D"] == D]
        print(f"\n--- B={B}, H={H}, N={N}, D={D} ---")
        for cr in config_results:
            ms = cr["mean_ms"]
            ms_str = f"{ms:.4f} ms" if ms > 0 else "ERROR"
            print(f"  {cr['backend']:<20} {ms_str:>12}   peak={cr['peak_mem_mb']:.0f} MB")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# BATCH SIZE SWEEP
# ═════════════════════════════════════════════════════════════════════════════

def bench_batch_sweep(runner, warmup, iters):
    """Sweep batch sizes: Triton vs CUDA."""
    triton_attn = get_triton_attn()
    cuda_attn = get_cuda_attn()

    configs = [
        (1, 8, 1024, 64),
        (4, 8, 1024, 64),
        (8, 8, 1024, 64),
        (1, 12, 1024, 128),
        (4, 12, 1024, 128),
        (8, 12, 1024, 128),
    ]

    results = []

    for B, H, N, D in configs:
        scale = 1.0 / math.sqrt(D)
        Q, K, V = make_qkv(B, H, N, D)
        row_base = {"B": B, "H": H, "N": N, "D": D, "causal": False}

        # Triton
        r = runner.run(
            lambda: triton_attn(Q, K, V, scale=scale),
            name=f"Triton B={B} D={D}",
        )
        results.append({
            **row_base,
            "backend": "triton_ours",
            "mean_ms": r.mean_ms,
            "std_ms": r.std_ms,
            "p95_ms": r.p95_ms,
        })

        # CUDA
        if cuda_attn is not None:
            r = runner.run(
                lambda: cuda_attn(Q, K, V, scale=scale),
                name=f"CUDA B={B} D={D}",
            )
            results.append({
                **row_base,
                "backend": "cuda_ours",
                "mean_ms": r.mean_ms,
                "std_ms": r.std_ms,
                "p95_ms": r.p95_ms,
            })

        print(f"\n--- B={B}, H={H}, N={N}, D={D} ---")
        batch_results = [r for r in results[-2:] if r.get("mean_ms")]
        for cr in batch_results:
            print(f"  {cr['backend']:<20} {cr['mean_ms']:.4f} ms")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# CAUSAL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def bench_causal_comparison(runner, warmup, iters):
    """Compare causal: Triton vs CUDA vs SDPA."""
    triton_attn = get_triton_attn()
    cuda_attn = get_cuda_attn()

    configs = [
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (1, 8, 4096, 64),
    ]

    results = []

    for B, H, N, D in configs:
        scale = 1.0 / math.sqrt(D)
        Q, K, V = make_qkv(B, H, N, D)
        row_base = {"B": B, "H": H, "N": N, "D": D, "causal": True}

        # Triton causal
        r_triton = runner.run(
            lambda: triton_attn(Q, K, V, scale=scale, is_causal=True),
            name=f"Triton causal N={N}",
        )
        results.append({**row_base, "backend": "triton_ours", "mean_ms": r_triton.mean_ms})

        # Triton non-causal (for speedup reference)
        r_noncausal = runner.run(
            lambda: triton_attn(Q, K, V, scale=scale, is_causal=False),
            name=f"Triton non-causal N={N}",
        )

        # CUDA causal
        cuda_ms = None
        if cuda_attn is not None:
            r_cuda = runner.run(
                lambda: cuda_attn(Q, K, V, scale=scale, is_causal=True),
                name=f"CUDA causal N={N}",
            )
            results.append({**row_base, "backend": "cuda_ours", "mean_ms": r_cuda.mean_ms})
            cuda_ms = r_cuda.mean_ms

        print(f"\n--- N={N} causal comparison ---")
        print(f"  Triton non-causal: {r_noncausal.mean_ms:.4f} ms")
        print(f"  Triton causal:     {r_triton.mean_ms:.4f} ms")
        if r_triton.mean_ms > 0:
            causal_speedup = r_noncausal.mean_ms / r_triton.mean_ms
            print(f"  Causal speedup:    {causal_speedup:.2f}x")
        if cuda_ms is not None:
            print(f"  CUDA causal:       {cuda_ms:.4f} ms")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE (ROADMAP FORMAT)
# ═════════════════════════════════════════════════════════════════════════════

def print_summary_table(results, D=64):
    """Print ROADMAP-style comparison table."""
    seq_lens = [512, 1024, 2048, 4096]
    backends = ["pytorch_eager", "torch_compile", "triton_ours", "cuda_ours"]
    backend_names = {
        "pytorch_eager": "PyTorch eager",
        "torch_compile": "torch.compile",
        "triton_ours": "Triton (ours)",
        "cuda_ours": "CUDA (ours)",
    }

    # Build lookup: (backend, N) → mean_ms
    lookup = {}
    for r in results:
        if r.get("D") == D and not r.get("causal", False):
            key = (r["backend"], r["N"])
            lookup[key] = r.get("mean_ms", -1)

    print(f"\n{'=' * 80}")
    print(f"  COMPARISON TABLE — head_dim={D}")
    print(f"{'=' * 80}")

    header = f"  {'Backend':<20}"
    for N in seq_lens:
        header += f" {'seq=' + str(N):>10}"
    print(header)
    print(f"  {'-' * 60}")

    for backend in backends:
        name = backend_names.get(backend, backend)
        row = f"  {name:<20}"
        for N in seq_lens:
            ms = lookup.get((backend, N))
            if ms is None:
                row += f" {'N/A':>10}"
            elif ms < 0:
                row += f" {'ERROR':>10}"
            else:
                row += f" {ms:>8.2f}ms"
        print(row)

    print(f"{'=' * 80}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FlashKernel: CUDA vs Triton vs PyTorch attention comparison"
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(ROOT, "benchmarks", "results", "attention_comparison.csv"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"VRAM: {mem_gb:.1f} GB")
    print()

    runner = BenchmarkRunner(warmup=args.warmup, timed=args.iters)

    all_results = []

    print("=" * 70)
    print("  SEQUENCE LENGTH SWEEP — 4 backends")
    print("=" * 70)
    all_results.extend(bench_seq_sweep(runner, args.warmup, args.iters))

    print("\n" + "=" * 70)
    print("  BATCH SIZE SWEEP — Triton vs CUDA")
    print("=" * 70)
    all_results.extend(bench_batch_sweep(runner, args.warmup, args.iters))

    print("\n" + "=" * 70)
    print("  CAUSAL vs NON-CAUSAL")
    print("=" * 70)
    all_results.extend(bench_causal_comparison(runner, args.warmup, args.iters))

    # Summary tables
    print_summary_table(all_results, D=64)
    print_summary_table(all_results, D=128)

    # Export CSV
    if all_results:
        write_csv(args.output, all_results)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
