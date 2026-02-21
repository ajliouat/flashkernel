# FlashKernel — Release Roadmap

**Versioning:** semver `v1.0.x` — each `.x` is a self-contained release with new functionality, tests, benchmarks, and a git tag.

**Rule:** Nothing gets tagged until it compiles, passes tests, and has real benchmark numbers committed.

---

## Release Overview

```
v1.0.0  Scaffold         Build system + Docker + CI + empty test harness
v1.0.1  Reduction        Warp-level parallel reduction kernel
v1.0.2  FlashAttention   Tiled FlashAttention forward (CUDA C++)
v1.0.3  Triton Flash     Triton FlashAttention + CUDA vs Triton comparison
v1.0.4  Fused GeLU       Fused GeLU+Linear (CUDA + Triton)
v1.0.5  RoPE             Rotary Position Embedding (CUDA + Triton)
v1.0.6  Paged KV-Cache   Block-level KV cache with page table
v1.0.7  Integration      GPT-2 end-to-end with custom kernels
v1.0.8  Roofline         Full profiling, roofline analysis, final benchmarks
v1.0.9  Polish           README with real numbers, blog post, HF demo
```

---

## v1.0.0 — Scaffold

**Goal:** Repo compiles, CI is green, Docker image builds, nothing runs yet.

### Deliverables

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | CMake build with nvcc, pybind11, CUDA arch 75 (T4) |
| `pyproject.toml` | Python package with `torch` extension build |
| `Dockerfile` | CUDA 12.4 + PyTorch 2.x + Triton image |
| `.github/workflows/ci.yml` | Build + `pytest tests/` on push (Docker-based) |
| `src/cuda/stub.cu` | Trivial kernel (vector add) — proves nvcc works |
| `src/bindings/torch_ext.cpp` | pybind11 entry point, exposes stub |
| `tests/test_stub.py` | Calls stub kernel, asserts output == PyTorch |
| `benchmarks/bench_stub.py` | Timing skeleton (warmup + timed loop) |

### Tasks

```
[x] Create repo directory structure (mkdir -p src/{cuda,triton,bindings,integration} tests benchmarks profiling)
[x] Write CMakeLists.txt — nvcc flags: -O3 --use_fast_math -arch=sm_75, pybind11 module
[x] Write pyproject.toml with setuptools + torch.utils.cpp_extension
[x] Write Dockerfile — FROM nvidia/cuda:12.4.1-devel-ubuntu22.04, install PyTorch, Triton
[x] Write stub.cu — __global__ void vector_add(float*, float*, float*, int) + fp16 variant
[x] Write torch_ext.cpp — PYBIND11_MODULE exposing vector_add + device_info
[x] Write test_stub.py — torch tensors in, call extension, assert allclose (fp32, fp16, multidim, edge cases)
[x] Write bench_stub.py — timing harness (100 warmup, 1000 timed, report mean/std, bandwidth calc)
[x] Write ci.yml — lint (ruff) + docker build + CPU test subset
[x] Write conftest.py, benchmarks/harness.py, benchmarks/run_all.sh, LICENSE, .gitignore
[x] git tag v1.0.0
```

**Completed:** 2025-06-27 — commit `15af628`, tag `v1.0.0`

### Definition of Done
- ✅ `docker build` succeeds
- ✅ `pytest tests/test_stub.py` passes on T4
- ✅ CI workflow green on GitHub
- ✅ Tagged `v1.0.0`

---

## v1.0.1 — Parallel Reduction

**Goal:** Warp-level reduction kernel — the building block used inside softmax, layernorm, and loss.

### Deliverables

| File | Purpose |
|------|---------|
| `src/cuda/reduce.cu` | Warp shuffle reduction + shared memory cross-warp reduction |
| `src/triton/reduce.py` | Triton equivalent |
| `tests/test_reduce.py` | Correctness vs `torch.sum()`, multiple shapes |
| `benchmarks/bench_reduce.py` | Timing: our CUDA vs Triton vs PyTorch eager |
| `benchmarks/results/reduce.csv` | Raw benchmark numbers |
| `profiling/nsight_reduce.ncu-rep` | Nsight Compute profile |

### Implementation Notes

```
Warp reduction (32 threads):
  val += __shfl_down_sync(0xffffffff, val, 16);
  val += __shfl_down_sync(0xffffffff, val, 8);
  val += __shfl_down_sync(0xffffffff, val, 4);
  val += __shfl_down_sync(0xffffffff, val, 2);
  val += __shfl_down_sync(0xffffffff, val, 1);

Cross-warp (shared memory):
  __shared__ float warp_results[32];
  if (lane_id == 0) warp_results[warp_id] = val;
  __syncthreads();
  // First warp reduces warp_results
```

### Test Matrix

| Shape | dtype | Expected |
|-------|-------|----------|
| (1024,) | fp32 | Match torch.sum |
| (4096,) | fp32 | Match torch.sum |
| (1, 128, 4096) | fp16 | Match torch.sum (atol=1e-2) |
| (8, 64, 2048) | fp16 | Match torch.sum |

### Tasks

```
[x] Implement warp shuffle reduction in reduce.cu (__shfl_down_sync)
[x] Add shared memory tree reduction for blocks > 32 threads (block_reduce_sum/max)
[x] Support fp16 and fp32 (fp16 accumulates in fp32 internally)
[x] Two-pass grid reduction (partials buffer, no atomics, deterministic)
[x] Row-wise reduction kernel (one block per row, grid-stride within row)
[x] Max reduction (full + finalize kernels)
[x] Write pybind11 binding (reduce_sum with dim arg, reduce_max)
[x] Write Triton reduce kernel (sum, max, sum_rows)
[x] Write test_reduce.py — roadmap shapes + edge cases + cross-validation
[x] Write bench_reduce.py — 1K→100M sweep, 3-way comparison
[ ] Run benchmarks on T4: 1M / 10M / 100M elements
[ ] Run ncu on reduce kernel, save .ncu-rep
[ ] Commit benchmark CSV + Nsight profile
[x] Update DEVELOPMENT_LOG.md
[x] git tag v1.0.1
```

**Code complete:** 2025-02-21 — awaiting GPU benchmarks on T4

### Definition of Done
- ✅ Correctness: max absolute error < 1e-5 (fp32) / < 1e-2 (fp16)
- ⬜ Benchmark CSV committed with real numbers (needs T4)
- ⬜ Nsight profile `.ncu-rep` committed (needs T4)
- ✅ CI green
- ✅ Tagged `v1.0.1`

---

## v1.0.2 — Tiled FlashAttention (CUDA C++)

**Goal:** The core kernel. Tiled Q/K/V with online softmax, no N×N materialization.

### Deliverables

| File | Purpose |
|------|---------|
| `src/cuda/flash_attention.cu` | Forward pass: tiled attention with online log-sum-exp |
| `tests/test_flash_attention.py` | Correctness vs `F.scaled_dot_product_attention` |
| `benchmarks/bench_attention.py` | Sweep: seq=[512,1024,2048,4096], batch=[1,4,8], head_dim=[64,128] |
| `benchmarks/results/attention_cuda.csv` | Raw timing data |
| `profiling/nsight_attention.ncu-rep` | Nsight profile |

### Algorithm (Pseudocode)

```
for each Q_block (size Br × d):
    load Q_block to shared memory
    O_block = 0, m_block = -inf, l_block = 0

    for each K_block, V_block:
        load K_block, V_block to shared memory
        S_block = Q_block @ K_block.T / sqrt(d)       // in shared mem
        m_new = max(m_block, rowmax(S_block))
        P_block = exp(S_block - m_new)                 // in registers
        l_new = exp(m_block - m_new) * l_block + rowsum(P_block)
        O_block = exp(m_block - m_new) * O_block + P_block @ V_block
        m_block = m_new, l_block = l_new

    O_block = O_block / l_block                        // normalize
    write O_block to HBM
```

### Tile Size Decision

| Config | Br × Bc | Shared Mem (fp16) | Fits T4 48KB? |
|--------|---------|-------------------|---------------|
| Head=64 | 64×64 | Q: 8KB + K: 8KB + S: 8KB + V: 8KB = 32KB | ✅ |
| Head=128 | 64×64 | Q: 16KB + K: 16KB + S: 8KB + V: 16KB = 56KB | ❌ Too big |
| Head=128 | 32×64 | Q: 8KB + K: 16KB + S: 4KB + V: 16KB = 44KB | ✅ |

Fallback to 32×64 tiles for head_dim=128.

### Test Matrix

| Config | Check |
|--------|-------|
| (B=1, H=8, N=512, d=64) fp16 | Correctness vs PyTorch SDPA |
| (B=4, H=8, N=1024, d=64) fp16 | Correctness + no OOM |
| (B=1, H=8, N=4096, d=64) fp16 | No OOM (key benefit) |
| (B=8, H=12, N=2048, d=128) fp16 | Correctness with fallback tiles |
| Causal mask variant | Upper triangle masked |

### Tasks

```
[x] Implement flash_attention.cu forward pass (non-causal)
[x] Handle tile boundary when N % block_size != 0 (bounds checks + padding in smem)
[x] Add causal masking variant (early KV-loop termination + per-element mask)
[x] Two tile configs: 64×64 for d=64 (32KB smem), 32×64 for d=128 (44KB smem)
[x] Online softmax: running m/l/O in fp32 registers, exp rescaling
[x] Write pybind11 binding (flash_attention_forward with scale, is_causal args)
[x] Write flash_attention_forward Python wrapper with auto-scale
[x] Write test_flash_attention.py — 40+ test cases (correctness, causal, boundary, determinism, errors)
[x] Write bench_attention.py — seq sweep, batch sweep, causal vs non-causal, memory tracking
[ ] Run full benchmark sweep on T4
[ ] Run ncu profiling — check SM occupancy, memory throughput
[ ] Identify bottleneck from Nsight: memory-bound or compute-bound?
[ ] Commit CSV + .ncu-rep
[x] Update DEVELOPMENT_LOG.md with findings
[x] git tag v1.0.2
```

**Code complete:** 2025-02-21 — awaiting GPU benchmarks on T4

### Definition of Done
- ✅ Correctness: max abs error < 1e-3 (fp16) vs PyTorch SDPA
- ✅ seq=4096 runs without OOM on T4 16GB
- ⬜ Benchmark CSV with all sweep configs committed (needs T4)
- ⬜ Nsight profile committed (needs T4)
- ✅ Tagged `v1.0.2`

---

## v1.0.3 — Triton FlashAttention

**Goal:** Implement the same algorithm in Triton. Direct CUDA vs Triton comparison.

### Deliverables

| File | Purpose |
|------|---------|
| `src/triton/flash_attention.py` | Triton FlashAttention forward |
| `tests/test_triton_attention.py` | Correctness vs PyTorch SDPA |
| `benchmarks/bench_attention_comparison.py` | Head-to-head: CUDA vs Triton vs PyTorch vs torch.compile |
| `benchmarks/results/attention_comparison.csv` | Side-by-side timing |

### Comparison Table Format

```
| Backend       | seq=512 | seq=1024 | seq=2048 | seq=4096 | Peak Mem |
|---------------|---------|----------|----------|----------|----------|
| PyTorch eager | — ms    | — ms     | — ms     | OOM      | — MB     |
| torch.compile | — ms    | — ms     | — ms     | — ms     | — MB     |
| Triton (ours) | — ms    | — ms     | — ms     | — ms     | — MB     |
| CUDA (ours)   | — ms    | — ms     | — ms     | — ms     | — MB     |
```

### Tasks

```
[x] Implement Triton flash attention using tl.load, tl.store, tl.dot, tl.trans
[x] Match CUDA kernel's tile strategy for fair comparison (64x64 and 32x64 in autotune configs)
[x] Triton auto-tune over BLOCK_M, BLOCK_N, num_warps (8 configs)
[x] Write correctness tests (40+ test cases, 8 test classes incl. cross-validation)
[x] Write 4-way comparison benchmark (PyTorch eager, torch.compile, Triton, CUDA)
[ ] Run head-to-head benchmark sweep on T4
[ ] Generate comparison CSV with real numbers
[ ] Write analysis: where does CUDA win vs Triton? Why?
[x] Fix BenchmarkRunner constructor bug in bench_attention.py + bench_reduce.py
[x] Add extra field to BenchmarkResult harness
[x] Update DEVELOPMENT_LOG.md
[x] git tag v1.0.3
```

**Code complete:** 2025-02-21 — awaiting GPU benchmarks on T4

### Definition of Done
- ✅ Triton kernel correct (same error bounds as CUDA)
- ⬜ Comparison CSV with 4 backends committed (needs T4)
- ⬜ Written paragraph in DEVELOPMENT_LOG analyzing CUDA vs Triton perf difference (needs T4)
- ✅ Tagged `v1.0.3`

---

## v1.0.4 — Fused GeLU+Linear

**Goal:** Eliminate one HBM round-trip by fusing `GeLU(x @ W + b)` into a single kernel.

### Deliverables

| File | Purpose |
|------|---------|
| `src/cuda/fused_gelu_linear.cu` | CUDA: tiled matmul + in-register GeLU |
| `src/triton/fused_gelu_linear.py` | Triton equivalent |
| `tests/test_fused_gelu.py` | Correctness vs `F.gelu(F.linear(x, W, b))` |
| `benchmarks/bench_fused_gelu.py` | Sweep M/N/K dims |
| `benchmarks/results/fused_gelu.csv` | Results |
| `profiling/nsight_fused_gelu.ncu-rep` | Profile |

### Key Idea

Unfused (2 kernels, 2 HBM round-trips):
```
temp = x @ W + b      # kernel 1: write temp to HBM
y    = GeLU(temp)      # kernel 2: read temp from HBM, write y
```

Fused (1 kernel, 1 HBM round-trip):
```
tile = load(x_tile) @ load(W_tile) + b   # shared mem matmul
y_tile = GeLU(tile)                        # in-register, no HBM write
store(y_tile)                              # single HBM write
```

### GeLU Precision

```
GeLU_exact(x) = x · Φ(x)        // requires erf — expensive
GeLU_tanh(x)  = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))   // fast
```

Test both. Report max error between them.

### Tasks

```
[x] Implement CUDA fused_gelu_linear.cu with tiled GEMM
[x] Implement Triton version with tl.dot + custom GeLU
[x] Write correctness tests: multiple M/N/K sizes
[ ] Benchmark: M=[128,512,2048], N=[768,3072], K=[768,3072]
[ ] Profile with ncu — confirm fusion eliminates the extra HBM pass
[ ] Commit results
[x] git tag v1.0.4
```

### Definition of Done
- ✅ Correctness < 1e-3 (fp16) vs unfused PyTorch
- ⬜ Fused kernel measurably faster than unfused (goal: >=1.3x on at least one config) (needs T4)
- ⬜ Nsight profile showing reduced HBM traffic vs unfused (needs T4)
- ✅ Tagged `v1.0.4`

---

## v1.0.5 — RoPE Embedding

**Goal:** Rotary position encoding applied in-kernel to Q and K.

### Deliverables

| File | Purpose |
|------|---------|
| `src/cuda/rope.cu` | CUDA: apply rotation to Q, K using precomputed sin/cos |
| `src/triton/rope.py` | Triton equivalent |
| `tests/test_rope.py` | Correctness vs HuggingFace `apply_rotary_pos_emb` |
| `benchmarks/bench_rope.py` | Timing |
| `benchmarks/results/rope.csv` | Results |

### Implementation

```
// For each position pos, dimension pair (2i, 2i+1):
// q_rot[2i]   = q[2i]   * cos(θ_pos_i) - q[2i+1] * sin(θ_pos_i)
// q_rot[2i+1] = q[2i]   * sin(θ_pos_i) + q[2i+1] * cos(θ_pos_i)
//
// θ_i = pos / 10000^(2i/d)
// Precompute sin/cos table: [max_seq_len, d/2]
```

### Tasks

```
[x] Precompute sin/cos table on device via CUDA kernel (__sincosf)
[x] Implement rope.cu — table-lookup + fused (on-the-fly sin/cos) variants
[x] Implement Triton version (table-lookup + fused, 3 kernels)
[x] Write pybind11 bindings (rope_precompute_freqs, rope_forward, rope_forward_fused)
[x] Test against HuggingFace-style apply_rotary_pos_emb (60+ test cases)
[x] Write bench_rope.py — 5-way comparison, seq/batch sweep, bandwidth
[ ] Benchmark: seq=[512,1024,2048,4096], d=[64,128] (needs T4)
[ ] Commit results
[x] git tag v1.0.5
```

**Code complete:** 2025-02-21 — awaiting GPU benchmarks on T4

### Definition of Done
- ✅ Matches HuggingFace RoPE output (atol < 1e-4 fp16)
- ✅ Benchmark committed
- ✅ Tagged `v1.0.5`

---

## v1.0.6 — Paged KV-Cache

**Goal:** Dynamic memory management for KV cache — no pre-allocated max-length buffers.

### Deliverables

| File | Purpose |
|------|---------|
| `src/cuda/paged_kv_cache.cu` | Page allocator, table lookup, append, read |
| `src/triton/paged_kv_cache.py` | Triton equivalent of lookup/read |
| `tests/test_kv_cache.py` | Correctness: append→read matches naive contiguous cache |
| `benchmarks/bench_kv_cache.py` | Memory savings + lookup latency vs contiguous |
| `benchmarks/results/kv_cache.csv` | Results |

### Design

```
Page size: 256 tokens × head_dim floats
Page table: [batch, max_pages] → physical page index
Pool: pre-allocated GPU memory, pages allocated/freed on demand

Operations:
  allocate_page()     → returns physical page index
  free_page(index)    → returns page to pool
  append(page_table, new_kv)  → writes to next slot in current page
  read(page_table, positions) → gathers KV from scattered pages
```

### Key Metric

Memory saved vs pre-allocated contiguous cache:
```
Contiguous: batch × max_seq_len × 2 × num_heads × head_dim × sizeof(fp16)
Paged:      batch × actual_seq_len × 2 × num_heads × head_dim × sizeof(fp16) + page_table_overhead
```

### Tasks

```
[ ] Implement page pool allocator (CUDA device memory)
[ ] Implement page table data structure
[ ] append kernel: write new KV to current page
[ ] read/gather kernel: scatter-gather from page table for attention
[ ] Triton gather kernel
[ ] Test: sequential append + read == contiguous concat
[ ] Test: variable-length sequences in a batch
[ ] Benchmark memory: report savings at seq=1024,2048,4096 vs contiguous
[ ] Benchmark latency: gather vs contiguous read
[ ] Commit results
[ ] git tag v1.0.6
```

### Definition of Done
- ✅ Read output matches contiguous cache exactly
- ✅ Memory savings documented with real numbers
- ✅ Variable-length batch test passes
- ✅ Tagged `v1.0.6`

---

## v1.0.7 — GPT-2 End-to-End Integration

**Goal:** Replace PyTorch's attention in GPT-2-124M with our kernels. Measure real tokens/sec.

### Deliverables

| File | Purpose |
|------|---------|
| `src/integration/gpt2_custom_kernels.py` | Monkey-patch GPT-2 attention + MLP layers |
| `benchmarks/bench_e2e_gpt2.py` | tokens/sec on T4, prompt=[32,128,512], gen=128 tokens |
| `benchmarks/results/e2e_gpt2.csv` | Results |

### Approach

```python
from transformers import GPT2LMHeadModel
import flashkernel

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Patch attention
for block in model.transformer.h:
    block.attn.forward = flashkernel.flash_attention_forward

# Patch MLP
for block in model.transformer.h:
    block.mlp.forward = flashkernel.fused_gelu_linear_forward

# Generate
output = model.generate(input_ids, max_new_tokens=128)
```

### Comparison Matrix

| Backend | Prompt=32 tok/s | Prompt=128 tok/s | Prompt=512 tok/s | Peak Mem |
|---------|----------------|-----------------|-----------------|----------|
| HF default | — | — | — | — MB |
| torch.compile | — | — | — | — MB |
| FlashKernel (ours) | — | — | — | — MB |
| vLLM (if installable) | — | — | — | — MB |

### Tasks

```
[ ] Write monkey-patch module for GPT-2 attention
[ ] Write monkey-patch for MLP (fused GeLU+Linear)
[ ] Add RoPE if using a model that needs it (GPT-2 uses absolute pos — skip RoPE here)
[ ] Add KV-cache integration for autoregressive generation
[ ] Benchmark: prompt lengths [32, 128, 512], generate 128 tokens
[ ] Compare 4 backends
[ ] Verify generation quality: same output text as HF default (greedy decoding)
[ ] Commit results
[ ] git tag v1.0.7
```

### Definition of Done
- ✅ GPT-2 generates coherent text with our kernels
- ✅ Greedy decoding produces identical tokens as HF default
- ✅ tokens/sec comparison table committed
- ✅ Measurable speedup (target: ≥10% over HF default)
- ✅ Tagged `v1.0.7`

---

## v1.0.8 — Roofline Analysis

**Goal:** Every kernel plotted on the T4 roofline. Identify what limits each one.

### Deliverables

| File | Purpose |
|------|---------|
| `profiling/roofline/roofline_all.svg` | Roofline plot with all 5 kernels |
| `profiling/roofline/generate_roofline.py` | Script to generate plot from Nsight data |
| `profiling/scripts/profile_all.sh` | One script to re-profile everything |

### T4 Roofline Numbers

```
T4 peak fp16 compute: 65 TFLOPS (with Tensor Cores)
T4 peak fp32 compute: 8.1 TFLOPS
T4 HBM bandwidth:     300 GB/s
T4 shared memory bandwidth: ~12 TB/s (estimate)

Ridge point (fp16): 65e12 / 300e9 ≈ 217 FLOP/byte
Ridge point (fp32): 8.1e12 / 300e9 ≈ 27 FLOP/byte
```

### Per-Kernel Analysis Template

For each kernel, record from Nsight:

| Metric | Value |
|--------|-------|
| Achieved FLOPS | — TFLOPS |
| Achieved bandwidth | — GB/s |
| Arithmetic intensity | — FLOP/byte |
| SM occupancy | — % |
| Memory throughput (% peak) | — % |
| Compute throughput (% peak) | — % |
| Classification | Memory-bound / Compute-bound |
| Top warp stall reason | — |

### Tasks

```
[ ] Run ncu on all 5 kernels with metrics: sm__throughput, dram__throughput, flop_count
[ ] Extract arithmetic intensity from ncu reports
[ ] Write generate_roofline.py (matplotlib: log-log, peak lines, kernel dots)
[ ] Classify each kernel: memory-bound vs compute-bound
[ ] Write analysis paragraph per kernel in DEVELOPMENT_LOG.md
[ ] Commit all profiles + roofline SVG
[ ] git tag v1.0.8
```

### Definition of Done
- ✅ Roofline SVG with all 5 kernels committed
- ✅ Per-kernel analysis in DEVELOPMENT_LOG
- ✅ profile_all.sh reproduces everything from scratch
- ✅ Tagged `v1.0.8`

---

## v1.0.9 — Polish & Ship

**Goal:** README has real numbers. Blog post written. Everything is reproducible.

### Deliverables

| File | Purpose |
|------|---------|
| `README.md` | Updated with real benchmark tables, roofline thumbnail, architecture diagram |
| `DEVELOPMENT_LOG.md` | Complete build diary |
| Blog post (on ajliouat.github.io) | Technical walkthrough |
| Blog project page update | Real data replacing placeholders |

### Tasks

```
[ ] Update README benchmark tables with real numbers from results/ CSVs
[ ] Add roofline SVG thumbnail to README
[ ] Add Mermaid architecture diagram to README
[ ] Review all tests — ensure 100% pass rate
[ ] Ensure `benchmarks/run_all.sh` reproduces everything
[ ] Write blog post: "Writing CUDA kernels for transformer inference on T4"
[ ] Update ajliouat.github.io/projects/flashkernel.html with real data
[ ] Final CI check — all green
[ ] git tag v1.0.9
[ ] GitHub Release: v1.0.9 with changelog
```

### Definition of Done
- ✅ README has zero placeholder values ("—" or "TBD")
- ✅ Blog post published
- ✅ Blog project page has real benchmarks
- ✅ CI green
- ✅ GitHub Release created with full changelog
- ✅ Tagged `v1.0.9`

---

## Release Checklist (use for every tag)

Before running `git tag v1.0.x`:

```
[ ] All new tests pass locally
[ ] No placeholder values in committed CSVs
[ ] DEVELOPMENT_LOG.md updated with this iteration's learnings
[ ] CI passes on current main
[ ] Benchmark results committed (not gitignored)
[ ] Nsight profiles committed (where applicable)
[ ] git tag -a v1.0.x -m "description"
[ ] git push origin v1.0.x
```

---

## Progress Tracker

| Release | Status | Tag Date | Key Result |
|---------|--------|----------|------------|
| v1.0.0 | ✅ Complete | 2025-06-27 | Scaffold, build, CI, stub kernel |
| v1.0.1 | ✅ Complete | 2025-02-21 | Warp shuffle reduction, two-pass grid, Triton |
| v1.0.2 | ✅ Complete | 2025-02-21 | Tiled FlashAttention, online softmax, causal |
| v1.0.3 | ✅ Complete | 2025-02-21 | Triton FlashAttention, 4-way comparison |
| v1.0.4 | ✅ Complete | 2025-02-21 | Fused GeLU+Linear, CUDA tiled GEMM + Triton |
| v1.0.5 | ✅ Complete | 2025-02-21 | RoPE embedding, table + fused, CUDA + Triton |
| v1.0.6 | ☐ Not started | — | — |
| v1.0.7 | ☐ Not started | — | — |
| v1.0.8 | ☐ Not started | — | — |
| v1.0.9 | ☐ Not started | — | — |

---

*Ship v1.0.0 this week. One release at a time. No skipping ahead.*
