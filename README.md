# FlashKernel — Custom CUDA Kernels for Transformer Inference

**Hardware/GPU Compute × LLM**

> Real CUDA C++ and Triton kernels for transformer inference, benchmarked with Nsight Compute profiling on NVIDIA T4.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

FlashKernel is a collection of hand-written CUDA C++ kernels targeting the critical path of transformer inference. Each kernel has a Triton equivalent for comparison, and all are benchmarked against PyTorch eager, `torch.compile`, and cuBLAS baselines.

The goal is not to outperform production libraries (FlashAttention, cuDNN) on every axis — it's to demonstrate deep understanding of GPU memory hierarchy, warp-level primitives, and kernel fusion through verified, profiled implementations.

## Kernels

| Kernel | CUDA C++ | Triton | Description |
|--------|----------|--------|-------------|
| Tiled FlashAttention | `src/cuda/flash_attention.cu` | `src/triton/flash_attention.py` | Tiled softmax-attention with online softmax, shared memory |
| Fused GeLU + Linear | `src/cuda/fused_gelu_linear.cu` | `src/triton/fused_gelu_linear.py` | Single-kernel GeLU activation + linear projection |
| RoPE Embedding | `src/cuda/rope.cu` | `src/triton/rope.py` | Rotary positional encoding |
| Paged KV-Cache | `src/cuda/paged_kv_cache.cu` | `src/triton/paged_kv_cache.py` | Block-level KV cache for long sequences |
| Parallel Reduction | `src/cuda/reduce.cu` | `src/triton/reduce.py` | Warp-level reduction with shared memory |

## Project Structure

```
flashkernel/
├── README.md
├── PROJECT_SPEC.md              # Detailed technical specification
├── DEVELOPMENT_LOG.md           # Build diary — what worked, what didn't
├── LICENSE
├── CMakeLists.txt
├── pyproject.toml
├── Dockerfile                   # Reproducible CUDA build environment
├── src/
│   ├── cuda/                    # CUDA C++ kernels
│   │   ├── flash_attention.cu
│   │   ├── fused_gelu_linear.cu
│   │   ├── rope.cu
│   │   ├── paged_kv_cache.cu
│   │   └── reduce.cu
│   ├── triton/                  # Triton equivalents
│   │   ├── flash_attention.py
│   │   ├── fused_gelu_linear.py
│   │   ├── rope.py
│   │   ├── paged_kv_cache.py
│   │   └── reduce.py
│   ├── bindings/                # PyTorch C++ extension bindings
│   │   └── torch_ext.cpp
│   └── integration/             # End-to-end LLM integration
│       └── gpt2_custom_kernels.py
├── benchmarks/
│   ├── bench_attention.py
│   ├── bench_fused_gelu.py
│   ├── bench_rope.py
│   ├── bench_kv_cache.py
│   ├── run_all.sh               # Single script to reproduce all benchmarks
│   └── results/                 # Raw benchmark outputs (committed)
│       └── .gitkeep
├── profiling/
│   ├── nsight_attention.ncu-rep  # Nsight Compute profiles (committed)
│   ├── roofline/                 # Roofline analysis outputs
│   └── scripts/
│       └── profile_all.sh
├── tests/
│   ├── test_flash_attention.py
│   ├── test_fused_gelu.py
│   ├── test_rope.py
│   ├── test_kv_cache.py
│   └── test_correctness.py      # Numerical correctness vs PyTorch reference
├── notebooks/
│   └── analysis.ipynb           # Visualization of benchmark results
└── .github/
    └── workflows/
        └── ci.yml               # Build + unit tests on push
```

## Hardware Target

- **Primary:** NVIDIA T4 (Turing, 16 GB, compute capability 7.5)
- **Development:** Apple Silicon Mac (Metal for prototyping, CUDA on AWS)
- **Instance:** AWS g4dn.xlarge ($0.526/hr on-demand, ~$0.16/hr spot)

## Build & Run

```bash
# Docker (recommended — reproducible CUDA environment)
docker build -t flashkernel .
docker run --gpus all flashkernel pytest tests/

# Native CUDA build
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
make -j$(nproc)

# Python interface
pip install -e .
python benchmarks/run_all.sh
```

## Benchmarks

_To be populated with real results after training runs. Example format:_

| Kernel | PyTorch Eager | torch.compile | Triton (ours) | CUDA C++ (ours) | Speedup |
|--------|--------------|---------------|---------------|-----------------|---------|
| FlashAttention (seq=2048) | — ms | — ms | — ms | — ms | —× |
| Fused GeLU+Linear | — ms | — ms | — ms | — ms | —× |
| RoPE | — ms | — ms | — ms | — ms | —× |

_All benchmarks measured on T4 16GB, CUDA 12.x, PyTorch 2.x, averaged over 100 warmup + 1000 timed iterations._

## Why This Project Exists

CUDA kernel engineering is the scarcest skill in ML infrastructure. This project demonstrates:

1. **Memory hierarchy mastery** — shared memory tiling, bank conflict avoidance, register pressure management
2. **Warp-level programming** — shuffle reductions, cooperative groups
3. **Kernel fusion** — eliminating memory round-trips between operations
4. **Profiling-driven optimization** — using Nsight Compute to identify bottlenecks, not guessing
5. **Real verification** — correctness tests against PyTorch reference, not fabricated benchmark tables

## License

Apache 2.0
