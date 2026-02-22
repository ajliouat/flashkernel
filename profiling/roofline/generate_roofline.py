#!/usr/bin/env python3
"""
FlashKernel — Roofline Plot Generator (v1.0.8)

Generates a log-log roofline diagram for NVIDIA T4 (Turing, SM 7.5) with
all FlashKernel CUDA kernels plotted. Reads kernel performance metrics
from kernel_metrics.json and produces an SVG.

T4 hardware ceilings:
  - fp16 Tensor Core peak: 65 TFLOPS
  - fp32 CUDA Core peak:   8.1 TFLOPS
  - HBM bandwidth:         300 GB/s
  - Ridge point (fp16):    ~217 FLOP/byte
  - Ridge point (fp32):    ~27 FLOP/byte

Usage:
    python profiling/roofline/generate_roofline.py
    python profiling/roofline/generate_roofline.py --metrics path/to/metrics.json
    python profiling/roofline/generate_roofline.py --output custom_name.svg
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for SVG generation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─── T4 Hardware Constants ──────────────────────────────────────────────────

T4_FP16_PEAK_TFLOPS = 65.0       # Tensor Core fp16 peak
T4_FP32_PEAK_TFLOPS = 8.1        # CUDA Core fp32 peak
T4_HBM_BW_GBS = 300.0            # HBM2 bandwidth (GB/s)

# Derived
T4_FP16_PEAK_FLOPS = T4_FP16_PEAK_TFLOPS * 1e12   # FLOP/s
T4_FP32_PEAK_FLOPS = T4_FP32_PEAK_TFLOPS * 1e12
T4_HBM_BW = T4_HBM_BW_GBS * 1e9                    # bytes/s

RIDGE_FP16 = T4_FP16_PEAK_FLOPS / T4_HBM_BW  # ~217 FLOP/byte
RIDGE_FP32 = T4_FP32_PEAK_FLOPS / T4_HBM_BW  # ~27 FLOP/byte


# ─── Default Metrics Path ───────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METRICS = os.path.join(SCRIPT_DIR, "kernel_metrics.json")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "roofline_all.svg")


# ─── Roofline Computation ───────────────────────────────────────────────────

def roofline_attainable(ai, peak_flops, peak_bw):
    """
    Compute attainable performance (FLOP/s) at a given arithmetic intensity.

    attainable = min(peak_flops, ai × peak_bw)

    Args:
        ai: Arithmetic intensity (FLOP/byte), scalar or array
        peak_flops: Peak compute throughput (FLOP/s)
        peak_bw: Peak memory bandwidth (bytes/s)

    Returns:
        Attainable performance in FLOP/s
    """
    return np.minimum(peak_flops, np.array(ai) * peak_bw)


# ─── Kernel Classification ──────────────────────────────────────────────────

CATEGORY_COLORS = {
    "elementwise":  "#4CAF50",   # Green — vector_add, rope
    "reduction":    "#2196F3",   # Blue — reduce_sum, reduce_max
    "attention":    "#F44336",   # Red — flash_attention
    "gemm":         "#FF9800",   # Orange — fused_gelu_linear
    "data_move":    "#9C27B0",   # Purple — kv_cache
}

CATEGORY_MARKERS = {
    "elementwise":  "o",
    "reduction":    "s",
    "attention":    "D",
    "gemm":         "^",
    "data_move":    "v",
}


# ─── Plot Generation ────────────────────────────────────────────────────────

def generate_roofline(metrics: list[dict], output_path: str):
    """
    Generate the roofline SVG plot.

    Args:
        metrics: List of kernel metric dicts, each with:
            - name (str)
            - arithmetic_intensity (float, FLOP/byte)
            - achieved_tflops (float, TFLOPS)
            - category (str)
            - precision (str, "fp16" or "fp32")
        output_path: Path for output SVG
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))

    # ── Arithmetic intensity range (log scale) ──────────────────────────
    ai_range = np.logspace(-1, 4, 1000)  # 0.1 to 10000 FLOP/byte

    # ── Roofline ceilings ────────────────────────────────────────────────
    # FP16 roofline
    fp16_roof = roofline_attainable(ai_range, T4_FP16_PEAK_FLOPS, T4_HBM_BW) / 1e12
    ax.plot(ai_range, fp16_roof, "-", color="#D32F2F", linewidth=2.5,
            label=f"fp16 Tensor Core ({T4_FP16_PEAK_TFLOPS} TFLOPS)", zorder=2)

    # FP32 roofline
    fp32_roof = roofline_attainable(ai_range, T4_FP32_PEAK_FLOPS, T4_HBM_BW) / 1e12
    ax.plot(ai_range, fp32_roof, "--", color="#1976D2", linewidth=2.0,
            label=f"fp32 CUDA Core ({T4_FP32_PEAK_TFLOPS} TFLOPS)", zorder=2)

    # Memory bandwidth ceiling (shared line, slope=1 on log-log)
    bw_roof = ai_range * T4_HBM_BW / 1e12
    # Only draw up to the lower of the two peaks for visual clarity
    bw_mask = bw_roof <= T4_FP32_PEAK_TFLOPS
    ax.plot(ai_range[bw_mask], bw_roof[bw_mask], ":", color="#666666",
            linewidth=1.5, alpha=0.6, zorder=1)

    # ── Ridge point markers ──────────────────────────────────────────────
    ax.axvline(x=RIDGE_FP16, color="#D32F2F", linestyle=":", alpha=0.3,
               linewidth=1)
    ax.axvline(x=RIDGE_FP32, color="#1976D2", linestyle=":", alpha=0.3,
               linewidth=1)

    ax.annotate(f"Ridge fp16\n({RIDGE_FP16:.0f} F/B)",
                xy=(RIDGE_FP16, T4_FP16_PEAK_TFLOPS),
                xytext=(RIDGE_FP16 * 1.5, T4_FP16_PEAK_TFLOPS * 0.6),
                fontsize=8, color="#D32F2F", alpha=0.7,
                arrowprops=dict(arrowstyle="->", color="#D32F2F", alpha=0.4))

    ax.annotate(f"Ridge fp32\n({RIDGE_FP32:.0f} F/B)",
                xy=(RIDGE_FP32, T4_FP32_PEAK_TFLOPS),
                xytext=(RIDGE_FP32 * 2.5, T4_FP32_PEAK_TFLOPS * 0.5),
                fontsize=8, color="#1976D2", alpha=0.7,
                arrowprops=dict(arrowstyle="->", color="#1976D2", alpha=0.4))

    # ── Plot kernel data points ──────────────────────────────────────────
    legend_categories = set()

    for k in metrics:
        ai = k["arithmetic_intensity"]
        perf = k["achieved_tflops"]
        cat = k.get("category", "elementwise")
        precision = k.get("precision", "fp16")
        name = k["name"]

        color = CATEGORY_COLORS.get(cat, "#333333")
        marker = CATEGORY_MARKERS.get(cat, "o")

        # Compute % of roofline
        if precision == "fp16":
            roof_at_ai = roofline_attainable(ai, T4_FP16_PEAK_FLOPS, T4_HBM_BW) / 1e12
        else:
            roof_at_ai = roofline_attainable(ai, T4_FP32_PEAK_FLOPS, T4_HBM_BW) / 1e12

        pct = (perf / roof_at_ai) * 100 if roof_at_ai > 0 else 0

        # Determine bound type
        if ai < (RIDGE_FP16 if precision == "fp16" else RIDGE_FP32):
            bound = "MEM"
        else:
            bound = "CMP"

        ax.scatter(ai, perf, c=color, marker=marker, s=180, zorder=5,
                   edgecolors="white", linewidths=0.8)

        # Label with name and % of roofline
        label_text = f"{name}\n({pct:.0f}% | {bound})"
        ax.annotate(label_text,
                    xy=(ai, perf),
                    xytext=(12, 10),
                    textcoords="offset points",
                    fontsize=7.5, fontweight="medium",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor=color,
                              alpha=0.85),
                    zorder=6)

        legend_categories.add(cat)

    # ── Shaded regions ───────────────────────────────────────────────────
    ax.fill_between(ai_range, 0, fp16_roof, alpha=0.04, color="#D32F2F")

    # ── Axes formatting ──────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12, fontweight="medium")
    ax.set_ylabel("Performance (TFLOPS)", fontsize=12, fontweight="medium")
    ax.set_title("FlashKernel — Roofline Analysis on NVIDIA T4 (Turing, SM 7.5)",
                 fontsize=14, fontweight="bold", pad=15)

    ax.set_xlim(0.08, 2000)
    ax.set_ylim(0.01, 100)

    ax.grid(True, which="both", ls="-", alpha=0.12)
    ax.grid(True, which="major", ls="-", alpha=0.25)

    # ── Legend ────────────────────────────────────────────────────────────
    # Hardware ceilings
    handles, labels = ax.get_legend_handles_labels()

    # Category patches
    for cat in sorted(legend_categories):
        patch = mpatches.Patch(color=CATEGORY_COLORS[cat],
                               label=cat.replace("_", " ").title())
        handles.append(patch)
        labels.append(cat.replace("_", " ").title())

    ax.legend(handles, labels, loc="lower right", fontsize=9,
              framealpha=0.9, edgecolor="#cccccc")

    # ── Bandwidth annotation ─────────────────────────────────────────────
    ax.text(0.12, 0.025, f"HBM2: {T4_HBM_BW_GBS:.0f} GB/s",
            fontsize=8, color="#666666", style="italic")

    # ── Save ─────────────────────────────────────────────────────────────
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=150, bbox_inches="tight")
    fig.savefig(output_path.replace(".svg", ".png"), format="png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Roofline plot saved to {output_path}")
    print(f"Roofline PNG saved to {output_path.replace('.svg', '.png')}")


# ─── Print Analysis ─────────────────────────────────────────────────────────

def print_analysis(metrics: list[dict]):
    """Print per-kernel analysis table to stdout."""
    print(f"\n{'═' * 100}")
    print(f"  FlashKernel — Roofline Analysis Summary (NVIDIA T4)")
    print(f"{'═' * 100}")
    print(f"  {'Kernel':<28} {'AI (F/B)':>10} {'TFLOPS':>10} {'BW (GB/s)':>10} "
          f"{'%Roof':>8} {'Bound':>10} {'Occupancy':>10}")
    print(f"{'─' * 100}")

    for k in metrics:
        ai = k["arithmetic_intensity"]
        perf = k["achieved_tflops"]
        bw = k.get("achieved_bw_gbs", 0)
        occ = k.get("sm_occupancy_pct", 0)
        precision = k.get("precision", "fp16")

        if precision == "fp16":
            roof = roofline_attainable(ai, T4_FP16_PEAK_FLOPS, T4_HBM_BW) / 1e12
        else:
            roof = roofline_attainable(ai, T4_FP32_PEAK_FLOPS, T4_HBM_BW) / 1e12

        pct = (perf / roof) * 100 if roof > 0 else 0
        bound = "MEM-bound" if ai < (RIDGE_FP16 if precision == "fp16" else RIDGE_FP32) else "CMP-bound"

        print(f"  {k['name']:<28} {ai:>10.1f} {perf:>10.2f} {bw:>10.1f} "
              f"{pct:>7.1f}% {bound:>10} {occ:>9.1f}%")

    print(f"{'═' * 100}")

    # Summary
    mem_bound = [k for k in metrics
                 if k["arithmetic_intensity"] < RIDGE_FP16]
    cmp_bound = [k for k in metrics
                 if k["arithmetic_intensity"] >= RIDGE_FP16]

    print(f"\n  Memory-bound kernels ({len(mem_bound)}): "
          f"{', '.join(k['name'] for k in mem_bound)}")
    print(f"  Compute-bound kernels ({len(cmp_bound)}): "
          f"{', '.join(k['name'] for k in cmp_bound)}")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate FlashKernel roofline plot for T4")
    parser.add_argument("--metrics", type=str, default=DEFAULT_METRICS,
                        help="Path to kernel_metrics.json")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output SVG path")
    args = parser.parse_args()

    # Load metrics
    with open(args.metrics) as f:
        data = json.load(f)

    metrics = data["kernels"]
    print(f"Loaded {len(metrics)} kernel metrics from {args.metrics}")

    # Print analysis table
    print_analysis(metrics)

    # Generate plot
    generate_roofline(metrics, args.output)


if __name__ == "__main__":
    main()
