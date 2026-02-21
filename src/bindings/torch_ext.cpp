/**
 * FlashKernel — PyTorch C++ Extension / pybind11 bindings
 *
 * Exposes CUDA kernels to Python via pybind11.
 * Each kernel version (v1.0.x) adds new bindings here.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "stub.cuh"

namespace py = pybind11;

// ─── Tensor-level wrappers ──────────────────────────────────────────────────
// These accept PyTorch tensors, extract data pointers, and call CUDA kernels.

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Input 'a' must be on CUDA device");
    TORCH_CHECK(b.device().is_cuda(), "Input 'b' must be on CUDA device");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have same shape");
    TORCH_CHECK(a.is_contiguous(), "Input 'a' must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input 'b' must be contiguous");

    auto c = torch::empty_like(a);
    const int n = a.numel();

    // Get current CUDA stream from PyTorch
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (a.scalar_type() == torch::kFloat32) {
        flashkernel::vector_add_f32(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            n, stream
        );
    } else if (a.scalar_type() == torch::kFloat16) {
        flashkernel::vector_add_f16(
            reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(b.data_ptr<at::Half>()),
            reinterpret_cast<half*>(c.data_ptr<at::Half>()),
            n, stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: expected float32 or float16");
    }

    return c;
}

py::dict device_info(int device_id = 0) {
    auto info = flashkernel::get_device_info(device_id);
    py::dict d;
    d["name"] = info.name;
    d["compute_capability"] = std::to_string(info.compute_major) + "." + std::to_string(info.compute_minor);
    d["sm_count"] = info.sm_count;
    d["global_mem_mb"] = info.global_mem_mb;
    d["shared_mem_per_block_kb"] = info.shared_mem_per_block_kb;
    d["max_threads_per_block"] = info.max_threads_per_block;
    d["warp_size"] = info.warp_size;
    d["clock_rate_mhz"] = info.clock_rate_mhz;
    d["memory_clock_mhz"] = info.memory_clock_mhz;
    d["memory_bus_width"] = info.memory_bus_width;
    return d;
}

// ─── Module Definition ──────────────────────────────────────────────────────

PYBIND11_MODULE(_flashkernel_C, m) {
    m.doc() = "FlashKernel — Custom CUDA kernels for transformer inference";

    // v1.0.0: stub
    m.def("vector_add", &vector_add,
          "Element-wise addition of two CUDA tensors (fp32 or fp16)",
          py::arg("a"), py::arg("b"));

    m.def("device_info", &device_info,
          "Query CUDA device properties",
          py::arg("device_id") = 0);

    // Version info
    m.attr("__version__") = "1.0.0";

    // Future kernel bindings will be added here:
    // v1.0.1: m.def("parallel_reduce", ...)
    // v1.0.2: m.def("flash_attention_forward", ...)
    // v1.0.3: (Triton — pure Python, no C++ binding needed)
    // v1.0.4: m.def("fused_gelu_linear", ...)
    // v1.0.5: m.def("rope_forward", ...)
    // v1.0.6: m.def("paged_kv_cache_append", ...) m.def("paged_kv_cache_read", ...)
}
