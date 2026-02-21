/**
 * FlashKernel — Stub kernel (v1.0.0)
 *
 * Trivial vector-add kernel that proves:
 * 1. nvcc compiles correctly for SM 7.5
 * 2. pybind11 binding works
 * 3. Host↔Device memory transfer is correct
 * 4. Benchmark harness can time it
 *
 * This gets replaced by real kernels in v1.0.1+.
 */

#include "stub.cuh"
#include <cuda_runtime.h>
#include <cstdio>

namespace flashkernel {

// ─── Device Kernel ──────────────────────────────────────────────────────────

__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ c,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_add_f16_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half*       __restrict__ c,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

// ─── Host Wrappers ──────────────────────────────────────────────────────────

void vector_add_f32(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads, 0, stream>>>(a, b, c, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "vector_add_f32 launch failed: %s\n", cudaGetErrorString(err));
    }
}

void vector_add_f16(const half* a, const half* b, half* c, int n, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add_f16_kernel<<<blocks, threads, 0, stream>>>(a, b, c, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "vector_add_f16 launch failed: %s\n", cudaGetErrorString(err));
    }
}

// ─── Device Query ───────────────────────────────────────────────────────────

DeviceInfo get_device_info(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    DeviceInfo info;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.global_mem_mb = static_cast<int>(prop.totalGlobalMem / (1024 * 1024));
    info.shared_mem_per_block_kb = static_cast<int>(prop.sharedMemPerBlock / 1024);
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.warp_size = prop.warpSize;
    info.clock_rate_mhz = prop.clockRate / 1000;
    info.memory_clock_mhz = prop.memoryClockRate / 1000;
    info.memory_bus_width = prop.memoryBusWidth;
    return info;
}

}  // namespace flashkernel
