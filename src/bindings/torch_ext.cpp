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
#include "reduce.cuh"

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

// ─── v1.0.1: Reduction wrappers ─────────────────────────────────────────────

torch::Tensor reduce_sum(torch::Tensor input, int dim) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (dim == -1) {
        // Full reduction → scalar
        auto output = torch::empty({1}, input.options());
        const int n = input.numel();

        if (input.scalar_type() == torch::kFloat32) {
            flashkernel::reduce_sum_f32(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                n, stream);
        } else if (input.scalar_type() == torch::kFloat16) {
            flashkernel::reduce_sum_f16(
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                n, stream);
        } else {
            TORCH_CHECK(false, "reduce_sum: unsupported dtype (expected fp32 or fp16)");
        }
        return output.squeeze(0);  // return scalar tensor
    }

    // dim-specified reduction: currently supports last-dim only
    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim,
                "reduce_sum: dim out of range [0, ", ndim, ")");

    // For last-dim reduction, reshape to (rows, cols) and use row-wise kernel
    if (dim == ndim - 1) {
        int cols = input.size(dim);
        int rows = input.numel() / cols;

        // Output shape: input shape with last dim removed
        auto out_sizes = input.sizes().vec();
        out_sizes.pop_back();
        if (out_sizes.empty()) out_sizes.push_back(1);
        auto output = torch::empty(out_sizes, input.options());

        if (input.scalar_type() == torch::kFloat32) {
            flashkernel::reduce_sum_rows_f32(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                rows, cols, stream);
        } else if (input.scalar_type() == torch::kFloat16) {
            flashkernel::reduce_sum_rows_f16(
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
                reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                rows, cols, stream);
        } else {
            TORCH_CHECK(false, "reduce_sum: unsupported dtype");
        }
        return output;
    }

    // For non-last dims: permute so target dim is last, then reduce
    std::vector<int64_t> perm;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) perm.push_back(i);
    }
    perm.push_back(dim);
    auto permuted = input.permute(perm).contiguous();
    return reduce_sum(permuted, ndim - 1);
}

torch::Tensor reduce_max(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto output = torch::empty({1}, input.options());
    const int n = input.numel();

    if (input.scalar_type() == torch::kFloat32) {
        flashkernel::reduce_max_f32(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            n, stream);
    } else if (input.scalar_type() == torch::kFloat16) {
        flashkernel::reduce_max_f16(
            reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            n, stream);
    } else {
        TORCH_CHECK(false, "reduce_max: unsupported dtype (expected fp32 or fp16)");
    }
    return output.squeeze(0);  // return scalar tensor
}

// ─── Device Info ────────────────────────────────────────────────────────────

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

    // v1.0.1: parallel reduction
    m.def("reduce_sum", &reduce_sum,
          "Sum reduction of a CUDA tensor (fp32 or fp16).\n"
          "If dim is specified, reduces along that dimension.\n"
          "Otherwise performs a full reduction to a scalar.",
          py::arg("input"), py::arg("dim") = -1);

    m.def("reduce_max", &reduce_max,
          "Max reduction of a CUDA tensor (fp32 or fp16).\n"
          "Full reduction to a scalar.",
          py::arg("input"));

    // Version info
    m.attr("__version__") = "1.0.1";

    // Future kernel bindings:
    // v1.0.2: m.def("flash_attention_forward", ...)
    // v1.0.3: (Triton — pure Python, no C++ binding needed)
    // v1.0.4: m.def("fused_gelu_linear", ...)
    // v1.0.5: m.def("rope_forward", ...)
    // v1.0.6: m.def("paged_kv_cache_append", ...) m.def("paged_kv_cache_read", ...)
}
