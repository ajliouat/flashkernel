#!/usr/bin/env python3
"""
FlashKernel — Extract Roofline Metrics from Nsight Compute Reports (v1.0.8)

Parses ncu CSV exports to extract per-kernel metrics needed for
roofline analysis: FLOP/s, bandwidth, occupancy, stall reasons.

Usage:
    python profiling/scripts/extract_metrics.py
    python profiling/scripts/extract_metrics.py --report-dir profiling/ncu_reports
"""

import argparse
import csv
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_REPORT_DIR = os.path.join(PROJECT_DIR, "profiling", "ncu_reports")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "profiling", "roofline", "kernel_metrics.json")


# ─── T4 constants for deriving absolute values from percentages ──────────────

T4_FP16_PEAK_TFLOPS = 65.0
T4_FP32_PEAK_TFLOPS = 8.1
T4_HBM_BW_GBS = 300.0


# ─── Metric extraction ──────────────────────────────────────────────────────

def parse_ncu_csv(csv_path: str) -> dict:
    """
    Parse an ncu CSV export and extract roofline-relevant metrics.

    Returns dict with:
        - kernel_name
        - duration_us
        - dram_read_bytes, dram_write_bytes
        - flop_count_fp16, flop_count_fp32
        - sm_throughput_pct, dram_throughput_pct
        - occupancy_pct
        - stall_long_scoreboard_pct, stall_math_throttle_pct
    """
    metrics = {}

    if not os.path.exists(csv_path):
        return metrics

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # ncu CSV format: columns include "Metric Name", "Metric Value"
            # or sometimes flat format depending on export method
            name = row.get("Metric Name", row.get("metric_name", ""))
            value = row.get("Metric Value", row.get("metric_value", ""))

            if not name or not value:
                continue

            try:
                val = float(value.replace(",", ""))
            except (ValueError, TypeError):
                continue

            metrics[name] = val

    return metrics


def compute_roofline_point(metrics: dict, precision: str = "fp16") -> dict:
    """
    Compute roofline coordinates from raw ncu metrics.

    Returns dict with arithmetic_intensity, achieved_tflops, achieved_bw_gbs.
    """
    # FLOP count (sum of all FP operations)
    if precision == "fp16":
        flops = (
            metrics.get("sm__sass_thread_inst_executed_op_hadd_pred_on.sum", 0)
            + metrics.get("sm__sass_thread_inst_executed_op_hmul_pred_on.sum", 0)
            + 2 * metrics.get("sm__sass_thread_inst_executed_op_hfma_pred_on.sum", 0)
        )
    else:
        flops = (
            metrics.get("sm__sass_thread_inst_executed_op_fadd_pred_on.sum", 0)
            + metrics.get("sm__sass_thread_inst_executed_op_fmul_pred_on.sum", 0)
            + 2 * metrics.get("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", 0)
        )

    # Bytes
    bytes_read = metrics.get("dram__bytes_read.sum", 0)
    bytes_write = metrics.get("dram__bytes_write.sum", 0)
    total_bytes = bytes_read + bytes_write

    # Duration
    duration_ns = metrics.get("gpu__time_duration.sum", 1)
    duration_s = duration_ns / 1e9

    # Derived
    ai = flops / total_bytes if total_bytes > 0 else 0
    tflops = (flops / duration_s) / 1e12 if duration_s > 0 else 0
    bw_gbs = (total_bytes / duration_s) / 1e9 if duration_s > 0 else 0

    return {
        "arithmetic_intensity": round(ai, 2),
        "achieved_tflops": round(tflops, 3),
        "achieved_bw_gbs": round(bw_gbs, 1),
        "total_flops": flops,
        "total_bytes": total_bytes,
        "duration_us": round(duration_ns / 1e3, 2),
    }


def extract_all(report_dir: str) -> list[dict]:
    """Extract metrics from all CSV reports in a directory."""
    results = []

    for fname in sorted(os.listdir(report_dir)):
        if not fname.endswith(".csv"):
            continue

        csv_path = os.path.join(report_dir, fname)
        metrics = parse_ncu_csv(csv_path)

        if not metrics:
            print(f"  Skipping {fname} (no metrics found)")
            continue

        kernel_name = fname.replace(".csv", "")
        point = compute_roofline_point(metrics)

        occupancy = metrics.get("launch__occupancy", 0)
        stall_mem = metrics.get(
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", 0)
        stall_math = metrics.get(
            "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct", 0)

        results.append({
            "name": kernel_name,
            "arithmetic_intensity": point["arithmetic_intensity"],
            "achieved_tflops": point["achieved_tflops"],
            "achieved_bw_gbs": point["achieved_bw_gbs"],
            "sm_occupancy_pct": round(occupancy, 1),
            "stall_long_scoreboard_pct": round(stall_mem, 1),
            "stall_math_throttle_pct": round(stall_math, 1),
        })

        print(f"  {kernel_name}: AI={point['arithmetic_intensity']:.1f} "
              f"TFLOPS={point['achieved_tflops']:.3f} "
              f"BW={point['achieved_bw_gbs']:.1f} GB/s")

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract roofline metrics from ncu reports")
    parser.add_argument("--report-dir", type=str, default=DEFAULT_REPORT_DIR,
                        help="Directory with ncu CSV exports")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output JSON path")
    args = parser.parse_args()

    print(f"Extracting metrics from {args.report_dir}...")
    results = extract_all(args.report_dir)

    if results:
        output = {"kernels": results}
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nWrote {len(results)} kernel metrics to {args.output}")
    else:
        print("\nNo metrics extracted. Run profile_all.sh first.")


if __name__ == "__main__":
    main()
