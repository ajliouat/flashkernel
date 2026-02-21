# FlashKernel — Technical Specification

## 1. Problem Statement

Transformer inference latency is dominated by attention computation and memory bandwidth.
Production systems use FlashAttention (Dao, 2022) and kernel fusion to reduce this bottleneck.
This project implements the core ideas from scratch in CUDA C++ and Triton, targeting NVIDIA T4 (Turing architecture, SM 7.5).

**Non-goal:** Beat FlashAttention-2 or cuDNN. The goal is to demonstrate understanding through working, profiled implementations.

## 2. Kernel Specifications

### 2.1 Tiled FlashAttention

**Paper reference:** [FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)

**Approach:**
- Tile Q, K, V into blocks that fit in shared memory (48 KB on T4)
- Online softmax using the log-sum-exp trick (Milakov & Gimelshein, 2018)
- No materialization of the full N×N attention matrix in HBM
- Two-pass: forward pass writes O and logsumexp; backward pass recomputes attention weights

**Tile sizes (T4 tuning):**
- Block size: 64×64 (fits in shared memory with fp16)
- Thread block: 128 threads (4 warps)
- Register budget: ~64 registers per thread

**Expected memory behavior:**
- HBM reads: O(N × d) instead of O(N²)
- Shared memory: 48 KB per block (Q tile + K tile + O accumulator)

### 2.2 Fused GeLU + Linear

Fuse `y = GeLU(x @ W + b)` into a single kernel:
- Load tile of x into shared memory
- Compute matrix multiply using wmma (Tensor Core) instructions on T4
- Apply GeLU activation in-register before writing to HBM
- Eliminates one full HBM read/write round-trip

**GeLU approximation:** Use the tanh approximation:
```
GeLU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

### 2.3 RoPE Embedding

Rotary Position Embedding applied directly in-kernel:
- Precompute sin/cos tables for positions
- Apply rotation to Q and K before attention
- Fuse with attention kernel when possible

### 2.4 Paged KV-Cache

Block-level virtual memory for KV cache:
- Fixed-size pages (256 tokens each)
- Page table to map logical → physical positions
- Support for dynamic sequence lengths without pre-allocation
- Memory pool with allocation/deallocation

### 2.5 Parallel Reduction

Warp-level reduction primitives used across other kernels:
- `__shfl_down_sync` based warp reduction
- Shared memory tree reduction for cross-warp results
- Used for: softmax normalization, loss computation, layer norm

## 3. Benchmark Design

### Methodology
- **Warmup:** 100 iterations (discard)
- **Timed:** 1000 iterations
- **Reporting:** Mean, std, min, max, p50, p95, p99
- **Memory:** Peak GPU memory via `torch.cuda.max_memory_allocated()`
- **Baselines:** PyTorch eager, `torch.compile(mode="max-autotune")`, cuBLAS (where applicable)

### Configurations
- Sequence lengths: 512, 1024, 2048, 4096
- Head dimensions: 64, 128
- Batch sizes: 1, 4, 8
- Data types: fp16, fp32
- Device: NVIDIA T4 16GB

### Profiling
- **Tool:** NVIDIA Nsight Compute (`ncu`)
- **Metrics:** SM occupancy, memory throughput (% of peak), compute throughput, warp stall reasons
- **Roofline:** Plot each kernel on arithmetic intensity vs performance roofline

## 4. End-to-End Integration

Replace PyTorch's native attention in GPT-2 (124M) or Llama-3.2-1B:
1. Load model with HuggingFace Transformers
2. Monkey-patch attention layers with custom CUDA/Triton kernels
3. Measure end-to-end generation latency (tokens/sec) on T4
4. Compare with: HF default, `torch.compile`, vLLM baseline

## 5. Success Criteria

| Metric | Threshold |
|--------|-----------|
| All kernels produce correct output (vs PyTorch) | Max absolute error < 1e-3 (fp16) |
| FlashAttention tiled kernel runs without OOM at seq=4096 | ✓ |
| At least one kernel faster than PyTorch eager | > 1.5× speedup |
| Nsight Compute profiles committed for every kernel | ✓ |
| End-to-end LLM inference shows measurable speedup | > 10% tokens/sec improvement |
| CI passes on every push | ✓ |

## 6. References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [NVIDIA T4 Architecture Whitepaper](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf)
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)

## 7. Timeline

| Week | Milestone |
|------|-----------|
| 1-2 | Set up CUDA build system, Docker, CI. Implement parallel reduction kernel. |
| 3-4 | Implement tiled FlashAttention (forward pass only). Benchmark and profile. |
| 5 | Implement Triton FlashAttention equivalent. Compare CUDA vs Triton. |
| 6 | Fused GeLU+Linear kernel (CUDA + Triton). RoPE kernel. |
| 7 | Paged KV-Cache implementation. |
| 8 | End-to-end GPT-2 integration. Full benchmark suite. |
| 9 | Roofline analysis. Polish README with real results. Write blog post. |
