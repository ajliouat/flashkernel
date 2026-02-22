#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# FlashKernel — Profile All CUDA Kernels with Nsight Compute (v1.0.8)
#
# Runs ncu (NVIDIA Nsight Compute CLI) on each kernel to extract:
#   - Achieved FLOP/s and bandwidth
#   - SM occupancy
#   - Memory/compute throughput
#   - Warp stall reasons
#
# Requirements:
#   - NVIDIA Nsight Compute (ncu) installed
#   - FlashKernel built: pip install -e ".[dev]"
#   - CUDA-capable GPU (T4 for production numbers)
#
# Usage:
#   bash profiling/scripts/profile_all.sh
#
# Outputs:
#   profiling/ncu_reports/*.ncu-rep  — raw Nsight Compute reports
#   profiling/ncu_reports/*.csv      — extracted metrics
#   profiling/roofline/roofline_all.svg — roofline plot (auto-generated)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
REPORT_DIR="$PROJECT_DIR/profiling/ncu_reports"
ROOFLINE_DIR="$PROJECT_DIR/profiling/roofline"

cd "$PROJECT_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  FlashKernel — Nsight Compute Profiling Suite"
echo "════════════════════════════════════════════════════════════"
echo ""

# ─── Check prerequisites ────────────────────────────────────────────────────

if ! command -v ncu &>/dev/null; then
    echo "ERROR: ncu (NVIDIA Nsight Compute CLI) not found."
    echo "Install: https://developer.nvidia.com/nsight-compute"
    exit 1
fi

if ! python -c "import flashkernel" &>/dev/null; then
    echo "ERROR: flashkernel not importable. Run: pip install -e '.[dev]'"
    exit 1
fi

mkdir -p "$REPORT_DIR"

# ─── Common ncu metrics ─────────────────────────────────────────────────────

# Metrics we collect for every kernel (roofline-relevant)
NCU_METRICS=(
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum"
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
    "sm__sass_thread_inst_executed_op_hadd_pred_on.sum"
    "sm__sass_thread_inst_executed_op_hmul_pred_on.sum"
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum"
    "dram__bytes_read.sum"
    "dram__bytes_write.sum"
    "launch__occupancy"
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct"
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"
    "smsp__warp_issue_stalled_wait_per_warp_active.pct"
    "gpu__time_duration.sum"
)

METRICS_STR=$(IFS=,; echo "${NCU_METRICS[*]}")

# ─── Profile helper ─────────────────────────────────────────────────────────

profile_kernel() {
    local name="$1"
    local script="$2"
    local kernel_filter="${3:-}"  # optional kernel name filter

    echo "▸ Profiling: $name"

    local report="$REPORT_DIR/${name}.ncu-rep"
    local csv="$REPORT_DIR/${name}.csv"

    local filter_args=""
    if [[ -n "$kernel_filter" ]]; then
        filter_args="--kernel-name $kernel_filter"
    fi

    # Run ncu profiling
    ncu --set full \
        --metrics "$METRICS_STR" \
        $filter_args \
        --export "$report" \
        --force-overwrite \
        python "$script" 2>&1 | tail -5

    # Export to CSV
    ncu --import "$report" --csv > "$csv" 2>/dev/null || true

    echo "  → Report: $report"
    echo "  → CSV:    $csv"
    echo ""
}

# ─── Profiling scripts (minimal, single-invocation) ─────────────────────────
# Each writes a small Python script that invokes one kernel once for profiling.

# v1.0.0: vector_add
cat > /tmp/fk_profile_vector_add.py << 'PYEOF'
import torch
import flashkernel
N = 1_048_576
a = torch.randn(N, dtype=torch.float16, device="cuda")
b = torch.randn(N, dtype=torch.float16, device="cuda")
for _ in range(3):
    c = flashkernel.vector_add(a, b)
torch.cuda.synchronize()
PYEOF

# v1.0.1: reduce_sum
cat > /tmp/fk_profile_reduce.py << 'PYEOF'
import torch
import flashkernel
x = torch.randn(1_048_576, dtype=torch.float16, device="cuda")
for _ in range(3):
    s = flashkernel.reduce_sum(x)
torch.cuda.synchronize()
PYEOF

# v1.0.2: flash_attention
cat > /tmp/fk_profile_flash_attn.py << 'PYEOF'
import torch, math
import flashkernel
B, H, N, D = 4, 12, 1024, 64
Q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
scale = 1.0 / math.sqrt(D)
for _ in range(3):
    O, L = flashkernel.flash_attention_forward(Q, K, V, scale=scale, is_causal=True)
torch.cuda.synchronize()
PYEOF

# v1.0.4: fused_gelu_linear
cat > /tmp/fk_profile_fused_gelu.py << 'PYEOF'
import torch
import flashkernel
M, K, N_out = 1024, 768, 3072
X = torch.randn(M, K, dtype=torch.float16, device="cuda")
W = torch.randn(N_out, K, dtype=torch.float16, device="cuda")
bias = torch.randn(N_out, dtype=torch.float16, device="cuda")
for _ in range(3):
    Y = flashkernel.fused_gelu_linear(X, W, bias=bias, use_tanh_approx=True)
torch.cuda.synchronize()
PYEOF

# v1.0.5: rope_forward_fused
cat > /tmp/fk_profile_rope_fused.py << 'PYEOF'
import torch
import flashkernel
B, H, N, D = 4, 12, 1024, 64
Q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
for _ in range(3):
    Q_rot, K_rot = flashkernel.rope_forward_fused(Q, K, base=10000.0)
torch.cuda.synchronize()
PYEOF

# v1.0.5: rope_forward (table)
cat > /tmp/fk_profile_rope_table.py << 'PYEOF'
import torch
import flashkernel
B, H, N, D = 4, 12, 1024, 64
Q = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, H, N, D, dtype=torch.float16, device="cuda")
cos_table, sin_table = flashkernel.rope_precompute_freqs(N, D, base=10000.0)
for _ in range(3):
    Q_rot, K_rot = flashkernel.rope_forward(Q, K, cos_table, sin_table)
torch.cuda.synchronize()
PYEOF

# v1.0.6: paged_kv_cache_append
cat > /tmp/fk_profile_kv_append.py << 'PYEOF'
import torch
import flashkernel
H, D, page_size = 12, 64, 256
num_pages = 64
T = 128  # tokens to append
pool = torch.zeros(num_pages, 2, H, page_size, D, dtype=torch.float16, device="cuda")
slot_mapping = torch.arange(T, dtype=torch.int32, device="cuda")
new_keys = torch.randn(T, H, D, dtype=torch.float16, device="cuda")
new_values = torch.randn(T, H, D, dtype=torch.float16, device="cuda")
for _ in range(3):
    flashkernel.paged_kv_cache_append(pool, slot_mapping, new_keys, new_values)
torch.cuda.synchronize()
PYEOF

# v1.0.6: paged_kv_cache_read
cat > /tmp/fk_profile_kv_read.py << 'PYEOF'
import torch
import flashkernel
B, H, D, page_size = 4, 12, 64, 256
max_seq = 512
num_pages = 64
pool = torch.randn(num_pages, 2, H, page_size, D, dtype=torch.float16, device="cuda")
pages_per_seq = (max_seq + page_size - 1) // page_size
block_table = torch.zeros(B, pages_per_seq, dtype=torch.int32, device="cuda")
for b in range(B):
    for p in range(pages_per_seq):
        block_table[b, p] = b * pages_per_seq + p
seq_lens = torch.full((B,), max_seq, dtype=torch.int32, device="cuda")
for _ in range(3):
    K, V = flashkernel.paged_kv_cache_read(pool, block_table, seq_lens, max_seq)
torch.cuda.synchronize()
PYEOF

# ─── Run all profiles ───────────────────────────────────────────────────────

echo "Starting ncu profiling (this may take several minutes)..."
echo ""

profile_kernel "vector_add_f16"     "/tmp/fk_profile_vector_add.py"   "vector_add"
profile_kernel "reduce_sum_f16"     "/tmp/fk_profile_reduce.py"       "reduce"
profile_kernel "flash_attention"    "/tmp/fk_profile_flash_attn.py"   "flash_attention"
profile_kernel "fused_gelu_linear"  "/tmp/fk_profile_fused_gelu.py"   "fused_gelu_linear"
profile_kernel "rope_fwd_fused"     "/tmp/fk_profile_rope_fused.py"   "rope_forward_fused"
profile_kernel "rope_fwd_table"     "/tmp/fk_profile_rope_table.py"   "rope_forward"
profile_kernel "kv_cache_append"    "/tmp/fk_profile_kv_append.py"    "paged_kv_cache_append"
profile_kernel "kv_cache_read"      "/tmp/fk_profile_kv_read.py"      "paged_kv_cache_read"

# ─── Extract metrics and generate roofline ───────────────────────────────────

echo "────────────────────────────────────────────────────────────"
echo "  Generating roofline plot..."
echo "────────────────────────────────────────────────────────────"

python profiling/roofline/generate_roofline.py

# ─── Cleanup ─────────────────────────────────────────────────────────────────

rm -f /tmp/fk_profile_*.py

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Profiling complete!"
echo "  Reports:  $REPORT_DIR/"
echo "  Roofline: $ROOFLINE_DIR/roofline_all.svg"
echo "════════════════════════════════════════════════════════════"
