# FlashKernel — Development Log

> This file tracks real progress, problems encountered, and solutions found.
> Updated as work happens — not retroactively.

---

## Status: v1.0.0 COMPLETE — Scaffold shipped

### Pre-Development Research (Week 0)
- [ ] Read FlashAttention paper + blog post
- [ ] Read FlashAttention-2 paper (parallelism improvements)
- [ ] Study NVIDIA T4 Turing architecture (SM 7.5 specifics)
- [ ] Review Triton tutorial: fused attention
- [ ] Set up AWS g4dn.xlarge spot instance
- [ ] Verify CUDA toolkit version and driver on instance
- [ ] Run baseline: `torch.nn.functional.scaled_dot_product_attention` latency on T4
- [ ] Run baseline: simple PyTorch matmul benchmark to establish HBM bandwidth

---

## 2025-06-27 — v1.0.0: Project Scaffold

### What was built
- **Build system:** CMakeLists.txt (nvcc -O3 --use_fast_math -arch=sm_75) + pyproject.toml/setup.py (torch CUDAExtension)
- **CUDA kernel:** `stub.cu` — `vector_add_f32`, `vector_add_f16` (__hadd), `get_device_info` (cudaDeviceProp query)
- **Bindings:** `torch_ext.cpp` — pybind11 module `_flashkernel_C`, dtype dispatch (fp32/fp16), input validation (CUDA, contiguous, shape match), CUDA stream from PyTorch
- **Python package:** `flashkernel/__init__.py` with lazy extension loading
- **Docker:** nvidia/cuda:12.4.1-devel-ubuntu22.04, PyTorch 2.5.1+cu124, Triton >=2.1
- **Tests:** 5 test classes — fp32 (4 shapes + zeros/negative/large), fp16 (5 shapes + small values), multidim (2D/3D), edge cases (mismatch/CPU/non-contiguous), device_info (keys/values)
- **Benchmark harness:** CUDA events timing (100 warmup + 1000 timed), BenchmarkResult stats (mean/std/p50/p95/p99), CSV export, PyTorch vs FlashKernel comparison
- **CI:** GitHub Actions — lint (ruff), docker build (buildx + GHA cache), CPU test subset
- **Misc:** Apache 2.0 license, .gitignore, conftest.py (pytest fixtures), run_all.sh

### Files added (23 total, 1341 lines)
```
.github/workflows/ci.yml    CMakeLists.txt       Dockerfile
LICENSE                      pyproject.toml       setup.py
flashkernel/__init__.py      src/cuda/stub.cu     src/cuda/stub.cuh
src/bindings/torch_ext.cpp   tests/__init__.py    tests/conftest.py
tests/test_stub.py           benchmarks/harness.py
benchmarks/bench_stub.py     benchmarks/run_all.sh
benchmarks/results/.gitkeep  notebooks/.gitkeep
profiling/roofline/.gitkeep  profiling/scripts/.gitkeep
src/triton/.gitkeep          src/integration/.gitkeep
```

### Design decisions
1. **pybind11 via FetchContent fallback** — CMake tries system install first, falls back to FetchContent v2.12.0. Setup.py uses PyTorch-bundled pybind11.
2. **fp16 from day one** — Added half-precision variant using `__hadd` to exercise fp16 pipeline early. Tests use wider tolerance (atol=1e-2).
3. **Lazy extension loading** — `__init__.py` defers `torch.utils.cpp_extension.load()` until first function call, so import works even without CUDA.
4. **Benchmark harness as reusable framework** — `BenchmarkRunner` and `BenchmarkResult` classes will be used for all future kernels.

### What worked
- Clean separation: CUDA kernels → pybind11 bindings → Python package → tests
- Docker image covers full dependency chain (CUDA 12.4, PyTorch 2.5.1, Triton)

### Next steps
- v1.0.1: Parallel reduction kernel (warp shuffle `__shfl_down_sync` + shared memory tree reduction)

---

## 2025-02-21 — v1.0.1: Parallel Reduction Kernels

### What was built

**CUDA kernels** (`src/cuda/reduce.cu`, 420+ lines):
- **Warp-level reduction** via `__shfl_down_sync(0xffffffff, val, offset)` — unrolled 5-step shuffle for 32-thread warps
- **Block-level reduction** via shared memory: each warp writes its partial to `__shared__ float[32]`, first warp reduces those
- **Two-pass grid reduction** for full array sum/max:
  - Pass 1: grid-stride loop, each block produces one partial (capped at 1024 blocks)
  - Pass 2: single block reduces partials → scalar
  - No atomics: completely deterministic
- **Row-wise reduction** (one block per row, grid-stride within cols)
- **All variants**: `reduce_sum_f32`, `reduce_sum_f16`, `reduce_max_f32`, `reduce_max_f16`, `reduce_sum_rows_f32`, `reduce_sum_rows_f16`
- **fp16 stability**: all fp16 paths promote to fp32 for accumulation, cast back at output

**pybind11 bindings** (`torch_ext.cpp`):
- `reduce_sum(input, dim=-1)` — full reduction or dim-specified (auto-permutes non-last dims)
- `reduce_max(input)` — full max reduction
- Both dispatch fp32/fp16, validate CUDA + contiguous

**Triton equivalents** (`src/triton/reduce.py`):
- `triton_reduce_sum`, `triton_reduce_max`, `triton_reduce_sum_rows`
- Two-pass approach matching CUDA: program-per-chunk → finalize
- Row-wise kernel: one program per row with `tl.sum`

**Tests** (`tests/test_reduce.py`, 270+ lines, 50+ test cases):
- `TestReduceSumF32` — 6 sizes (1 to 1M), zeros, ones, negatives
- `TestReduceSumF16` — 5 sizes with wider tolerance
- `TestReduceSumRoadmapShapes` — exact shapes from ROADMAP.md test matrix
- `TestReduceMaxF32` / `TestReduceMaxF16` — parametric + known-max test
- `TestReduceSumRows` — 2D/3D, first/last dim, fp16
- `TestReduceEdgeCases` — CPU raises, non-contiguous raises, single-element
- `TestTritonReduceSum / Max / Rows` — Triton correctness
- `TestCrossValidation` — CUDA vs Triton agree on same inputs

**Benchmark** (`benchmarks/bench_reduce.py`):
- Full sum sweep: 1K → 100M elements, 3-way (PyTorch vs CUDA vs Triton)
- Full max sweep: 1K → 10M, 3-way
- Row-wise sweep: 5 shapes (128×4096 → 8192×512)
- Computes effective bandwidth (GB/s) for each measurement
- CSV export to `benchmarks/results/reduce.csv`

### Design decisions
1. **Two-pass over atomics** — atomic add is non-deterministic across runs. Two-pass with intermediate buffer gives bit-exact results across launches.
2. **Grid-stride loop** — each thread processes multiple elements before the block reduce, maximizing occupancy for large arrays.
3. **Block size = 256 (8 warps)** — sweet spot for T4: 4 blocks per SM at full occupancy, fits shared memory budget.
4. **fp16 accumulation in fp32** — half-precision accumulation overflows for n > ~2K. Always promote, cast back at finalize.
5. **Row-wise = one block per row** — simple mapping, good for cols ≤ blockDim.x. Falls back to grid-stride for large cols.
6. **`dim` argument with auto-permute** — non-last-dim reduces permute the tensor to put target dim last, then call the row-wise kernel. Matches PyTorch API.

### What worked
- Warp shuffle is elegant: 5 lines replace what would be 5 `__syncthreads()` barriers with shared memory
- Block-level shared memory reduction is clean: `shared[32]` is tiny, no bank conflicts
- Triton implementation was ~3x less code than CUDA for equivalent functionality

### Files added/modified
```
NEW:  src/cuda/reduce.cu          (420 lines — kernels + host wrappers)
NEW:  src/cuda/reduce.cuh         (header with API declarations)
NEW:  src/triton/reduce.py        (Triton equivalents)
NEW:  tests/test_reduce.py        (50+ test cases)
NEW:  benchmarks/bench_reduce.py  (3-way comparison benchmark)
MOD:  src/bindings/torch_ext.cpp  (added reduce_sum, reduce_max bindings)
MOD:  flashkernel/__init__.py     (expose reduce_sum, reduce_max, triton variants)
MOD:  pyproject.toml              (version → 1.0.1)
MOD:  setup.py                    (version → 1.0.1)
MOD:  benchmarks/run_all.sh       (added bench_reduce.py)
MOD:  .github/workflows/ci.yml    (lint src/triton/)
DEL:  src/triton/.gitkeep         (replaced by reduce.py)
```

### Next steps
- Run full benchmark suite + Nsight Compute profiling on T4 instance
- v1.0.2: Tiled FlashAttention forward (CUDA C++) — the core kernel

---

## 2025-02-21 — v1.0.2: Tiled FlashAttention (CUDA C++)

**The core kernel.** This is the reason the project exists.

### What was built

**CUDA kernel** (`src/cuda/flash_attention.cu`, 280+ lines):
- **Tiled FlashAttention forward pass** — Dao et al., 2022 algorithm
- **Online softmax**: running max `m`, sum `l`, and output `O` in fp32 registers
  - No N×N attention matrix ever materialized in HBM
  - Memory: O(N) instead of O(N²)
- **Two tile configurations** based on shared memory budget (T4 = 48 KB):
  - `head_dim=64`: Br=64 × Bc=64 → Q:8K + K:8K + V:8K = 24 KB smem ✓
  - `head_dim=128`: Br=32 × Bc=64 → Q:8K + K:16K + V:16K = 40 KB smem ✓
- **Causal masking**: early termination in KV-loop (`max_kv_block = q_row/Bc + 1`) + per-element `-inf` mask
- **Boundary handling**: padding zeros in shared memory for N not divisible by tile size
- **Grid mapping**: `(batch*heads, ceil(N/Br))` — one thread per query row in the tile
- **Log-sum-exp output**: `L[b,h,n] = m + log(l)` stored for potential backward pass

**Key algorithm steps per thread (one query row):**
```
1. Load my Q row to shared memory
2. For each KV block:
   a. Cooperatively load K, V to shared memory
   b. Compute dot(Q_row, K_col) * scale for all Bc columns → s_vals[]
   c. Apply causal mask if needed (s = -inf for future tokens)
   d. row_max = max(s_vals)
   e. Rescale: alpha = exp(m_old - m_new), O *= alpha, l *= alpha
   f. P = exp(s - m_new), accumulate l += P, O += P * V
3. Normalize: O /= l, write O (fp16) and L = m + log(l) to HBM
```

**pybind11 bindings** (`torch_ext.cpp`):
- `flash_attention_forward(Q, K, V, scale=-1, is_causal=false)` → `(O, L)`
- Full input validation: CUDA, contiguous, fp16, 4-D, shape match, head_dim ∈ {64, 128}
- Auto-scale: if `scale < 0`, computes `1/sqrt(D)` in Python wrapper

**Python API** (`flashkernel/__init__.py`):
- `flash_attention_forward(Q, K, V, scale=None, is_causal=False)` → `(O, L)`
- Auto-computes scale = 1/√d if not specified
- Clean docstring with supported configs

**Tests** (`tests/test_flash_attention.py`, 300+ lines, 40+ test cases):
- `TestFlashAttentionD64`: basic (B=1,H=8,N=512), batched (B=4,N=1024), long seq (N=4096), sweep [128→2048]
- `TestFlashAttentionD128`: B=8,H=12,N=2048, small config, sweep [128→1024]
- `TestFlashAttentionCausal`: d=64, d=128, long seq N=2048, sweep [64→1024]
- `TestFlashAttentionBoundary`: non-divisible N: [65, 100, 127, 129, 200, 513, 1000] for d=64; [33, 63, 97, 255] for d=128
- `TestFlashAttentionOutputs`: shapes, dtypes, LSE finite, output finite, custom scale
- `TestFlashAttentionDeterminism`: same input → bit-exact same output
- `TestFlashAttentionErrors`: CPU raises, fp32 raises, wrong head_dim, shape mismatch, 3-D input
- All correctness tests compare against `F.scaled_dot_product_attention` reference

**Benchmark** (`benchmarks/bench_attention.py`):
- Sequence sweep: N=[128,256,512,1024,2048,4096] × d=[64,128], FlashKernel vs SDPA vs naive
- Batch sweep: B=[1,4,8] × d=[64,128] at N=1024
- Causal vs non-causal: N=[512,1024,2048], measures speedup from early termination
- Peak memory tracking per config
- CSV export to `benchmarks/results/attention_cuda.csv`

### Design decisions
1. **One thread per query row** — simple mapping, each thread does full dot products and accumulation. Not the most parallel approach but clean and correct. Optimization target for v1.0.8.
2. **fp32 accumulators in registers** — all intermediate values (m, l, O, S) in fp32. Only final output cast to fp16.
3. **Dynamic shared memory** — `extern __shared__` with size computed by host. Allows same kernel template for different tile sizes.
4. **Template parameters for tile dims** — `<Br, Bc, D>` compile-time constants enable loop unrolling and register allocation.
5. **Causal early termination** — `max_kv_block = (q_row / Bc) + 1` skips all future KV blocks entirely. Expected ~2x speedup for causal at large N.
6. **Log-sum-exp stored** — `L[b,h,n] = m + log(l)` saved for potential backward pass implementation.

### Files added/modified
```
NEW:  src/cuda/flash_attention.cu       (280 lines — the core kernel)
NEW:  src/cuda/flash_attention.cuh      (header with API)
NEW:  tests/test_flash_attention.py     (40+ test cases, 300+ lines)
NEW:  benchmarks/bench_attention.py     (seq/batch/causal sweeps + memory)
MOD:  src/bindings/torch_ext.cpp        (added flash_attention_forward binding)
MOD:  flashkernel/__init__.py           (expose flash_attention_forward with auto-scale)
MOD:  pyproject.toml                    (version → 1.0.2)
MOD:  setup.py                          (version → 1.0.2)
MOD:  benchmarks/run_all.sh             (added bench_attention.py)
```

### Next steps
- Run full benchmark sweep + Nsight Compute on T4 instance
- Identify: memory-bound or compute-bound? Check SM occupancy, HBM throughput
- v1.0.3: Triton FlashAttention — same algorithm in Triton, head-to-head comparison

---
