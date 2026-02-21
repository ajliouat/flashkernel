#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FlashKernel — Run all benchmarks
#
# Usage:
#   bash benchmarks/run_all.sh
#
# Outputs:
#   benchmarks/results/*.csv    — raw timing data
#   stdout                      — formatted comparison tables
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  FlashKernel — Benchmark Suite"
echo "════════════════════════════════════════════════════════════"
echo ""

# Ensure results directory exists
mkdir -p benchmarks/results

# v1.0.0: Stub benchmark (vector add + bandwidth baseline)
echo "▸ Running stub benchmark (vector_add)..."
python benchmarks/bench_stub.py

# v1.0.1: Reduction benchmark
echo "▸ Running reduction benchmark (sum, max, row-wise)..."
python benchmarks/bench_reduce.py

# v1.0.2: Attention benchmark
echo "▸ Running attention benchmark (FlashAttention forward)..."
python benchmarks/bench_attention.py

# v1.0.3: CUDA vs Triton attention comparison
echo "▸ Running CUDA vs Triton attention comparison..."
python benchmarks/bench_attention_comparison.py

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All benchmarks complete. Results in benchmarks/results/"
echo "════════════════════════════════════════════════════════════"
