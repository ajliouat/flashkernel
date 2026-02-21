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
