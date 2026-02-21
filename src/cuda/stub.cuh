/**
 * FlashKernel — Stub kernel header
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>

namespace flashkernel {

// ─── Device Info ────────────────────────────────────────────────────────────

struct DeviceInfo {
    std::string name;
    int compute_major;
    int compute_minor;
    int sm_count;
    int global_mem_mb;
    int shared_mem_per_block_kb;
    int max_threads_per_block;
    int warp_size;
    int clock_rate_mhz;
    int memory_clock_mhz;
    int memory_bus_width;
};

DeviceInfo get_device_info(int device_id = 0);

// ─── Kernel Wrappers ────────────────────────────────────────────────────────

void vector_add_f32(const float* a, const float* b, float* c, int n, cudaStream_t stream = 0);
void vector_add_f16(const half* a, const half* b, half* c, int n, cudaStream_t stream = 0);

}  // namespace flashkernel
