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
#include "flash_attention.cuh"
#include "fused_gelu_linear.cuh"
#include "rope.cuh"
#include "paged_kv_cache.cuh"

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

// ─── v1.0.2: FlashAttention wrappers ────────────────────────────────────────

std::vector<torch::Tensor> flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
    bool is_causal
) {
    // Validate inputs
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA device");
    TORCH_CHECK(V.device().is_cuda(), "V must be on CUDA device");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16,
                "Q must be float16 (FlashAttention operates in fp16)");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(V.scalar_type() == torch::kFloat16, "V must be float16");

    // Shape: [batch, num_heads, seq_len, head_dim]
    TORCH_CHECK(Q.dim() == 4, "Q must be 4-D [B, H, N, D]");
    TORCH_CHECK(K.dim() == 4, "K must be 4-D [B, H, N, D]");
    TORCH_CHECK(V.dim() == 4, "V must be 4-D [B, H, N, D]");

    int batch    = Q.size(0);
    int heads    = Q.size(1);
    int seq_len  = Q.size(2);
    int head_dim = Q.size(3);

    TORCH_CHECK(K.size(0) == batch && K.size(1) == heads,
                "K batch/heads must match Q");
    TORCH_CHECK(V.size(0) == batch && V.size(1) == heads,
                "V batch/heads must match Q");
    TORCH_CHECK(K.size(2) == seq_len, "K seq_len must match Q");
    TORCH_CHECK(V.size(2) == seq_len, "V seq_len must match Q");
    TORCH_CHECK(K.size(3) == head_dim, "K head_dim must match Q");
    TORCH_CHECK(V.size(3) == head_dim, "V head_dim must match Q");
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);

    // Allocate output
    auto O = torch::empty_like(Q);  // [B, H, N, D] fp16
    auto L = torch::empty({batch, heads, seq_len},
                          torch::TensorOptions().device(Q.device()).dtype(torch::kFloat32));

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    flashkernel::flash_attention_forward(
        reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
        reinterpret_cast<half*>(O.data_ptr<at::Half>()),
        L.data_ptr<float>(),
        batch, heads, seq_len, head_dim,
        scale, is_causal, stream
    );

    return {O, L};
}

// ─── Device Info ────────────────────────────────────────────────────────────

// ─── v1.0.4: Fused GeLU+Linear wrapper ─────────────────────────────────────

torch::Tensor fused_gelu_linear(
    torch::Tensor X,
    torch::Tensor W,
    c10::optional<torch::Tensor> bias,
    bool use_tanh_approx
) {
    TORCH_CHECK(X.device().is_cuda(), "X must be on CUDA device");
    TORCH_CHECK(W.device().is_cuda(), "W must be on CUDA device");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(X.scalar_type() == torch::kFloat16, "X must be float16");
    TORCH_CHECK(W.scalar_type() == torch::kFloat16, "W must be float16");
    TORCH_CHECK(X.dim() == 2, "X must be 2-D [M, K]");
    TORCH_CHECK(W.dim() == 2, "W must be 2-D [N, K]");
    TORCH_CHECK(X.size(1) == W.size(1),
                "X columns (", X.size(1), ") must match W columns (", W.size(1), ")");

    int M = X.size(0);
    int K = X.size(1);
    int N = W.size(0);

    const half* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto& b = bias.value();
        TORCH_CHECK(b.device().is_cuda(), "bias must be on CUDA device");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.scalar_type() == torch::kFloat16, "bias must be float16");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == N,
                    "bias must be 1-D of size N=", N, ", got shape [", b.size(0), "]");
        bias_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    }

    auto Y = torch::empty({M, N}, X.options());  // [M, N] fp16
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    flashkernel::fused_gelu_linear(
        reinterpret_cast<const half*>(X.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(W.data_ptr<at::Half>()),
        bias_ptr,
        reinterpret_cast<half*>(Y.data_ptr<at::Half>()),
        M, N, K,
        use_tanh_approx,
        stream
    );

    return Y;
}

// ─── v1.0.5: RoPE wrappers ──────────────────────────────────────────────────

std::vector<torch::Tensor> rope_precompute_freqs(
    int max_seq_len,
    int head_dim,
    float base
) {
    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even, got ", head_dim);

    int half_dim = head_dim / 2;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto cos_table = torch::empty({max_seq_len, half_dim}, options);
    auto sin_table = torch::empty({max_seq_len, half_dim}, options);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    flashkernel::rope_precompute_freqs(
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        max_seq_len, head_dim, base, stream
    );

    return {cos_table, sin_table};
}

std::vector<torch::Tensor> rope_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor cos_table,
    torch::Tensor sin_table
) {
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA device");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4-D [B, H, N, D]");
    TORCH_CHECK(K.dim() == 4, "K must be 4-D [B, H, N, D]");

    int batch    = Q.size(0);
    int heads    = Q.size(1);
    int seq_len  = Q.size(2);
    int head_dim = Q.size(3);

    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even, got ", head_dim);
    TORCH_CHECK(K.sizes() == Q.sizes(), "K shape must match Q shape");
    TORCH_CHECK(cos_table.size(0) >= seq_len,
                "cos_table max_seq_len (", cos_table.size(0),
                ") must be >= seq_len (", seq_len, ")");
    TORCH_CHECK(cos_table.size(1) == head_dim / 2,
                "cos_table half_dim mismatch");
    TORCH_CHECK(sin_table.sizes() == cos_table.sizes(),
                "sin_table shape must match cos_table");
    TORCH_CHECK(cos_table.scalar_type() == torch::kFloat32, "cos_table must be float32");
    TORCH_CHECK(sin_table.scalar_type() == torch::kFloat32, "sin_table must be float32");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Apply in-place to clones (don't modify originals unexpectedly)
    auto Q_out = Q.clone();
    auto K_out = K.clone();

    flashkernel::rope_forward(
        reinterpret_cast<half*>(Q_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(K_out.data_ptr<at::Half>()),
        cos_table.data_ptr<float>(),
        sin_table.data_ptr<float>(),
        batch, heads, seq_len, head_dim, stream
    );

    return {Q_out, K_out};
}

std::vector<torch::Tensor> rope_forward_fused(
    torch::Tensor Q,
    torch::Tensor K,
    float base
) {
    TORCH_CHECK(Q.device().is_cuda(), "Q must be on CUDA device");
    TORCH_CHECK(K.device().is_cuda(), "K must be on CUDA device");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Q must be float16");
    TORCH_CHECK(K.scalar_type() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4-D [B, H, N, D]");
    TORCH_CHECK(K.dim() == 4, "K must be 4-D [B, H, N, D]");

    int batch    = Q.size(0);
    int heads    = Q.size(1);
    int seq_len  = Q.size(2);
    int head_dim = Q.size(3);

    TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even, got ", head_dim);
    TORCH_CHECK(K.sizes() == Q.sizes(), "K shape must match Q shape");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto Q_out = Q.clone();
    auto K_out = K.clone();

    flashkernel::rope_forward_fused(
        reinterpret_cast<half*>(Q_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(K_out.data_ptr<at::Half>()),
        batch, heads, seq_len, head_dim, base, stream
    );

    return {Q_out, K_out};
}

// ─── v1.0.6: Paged KV-Cache wrappers ────────────────────────────────────────

void paged_kv_cache_append(
    torch::Tensor pool,
    torch::Tensor slot_mapping,
    torch::Tensor new_keys,
    torch::Tensor new_values
) {
    TORCH_CHECK(pool.device().is_cuda(), "pool must be on CUDA device");
    TORCH_CHECK(slot_mapping.device().is_cuda(), "slot_mapping must be on CUDA device");
    TORCH_CHECK(new_keys.device().is_cuda(), "new_keys must be on CUDA device");
    TORCH_CHECK(new_values.device().is_cuda(), "new_values must be on CUDA device");
    TORCH_CHECK(pool.is_contiguous(), "pool must be contiguous");
    TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
    TORCH_CHECK(new_keys.is_contiguous(), "new_keys must be contiguous");
    TORCH_CHECK(new_values.is_contiguous(), "new_values must be contiguous");

    TORCH_CHECK(pool.scalar_type() == torch::kFloat16, "pool must be float16");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt32, "slot_mapping must be int32");
    TORCH_CHECK(new_keys.scalar_type() == torch::kFloat16, "new_keys must be float16");
    TORCH_CHECK(new_values.scalar_type() == torch::kFloat16, "new_values must be float16");

    // Pool: [num_pages, 2, num_heads, page_size, head_dim]
    TORCH_CHECK(pool.dim() == 5, "pool must be 5-D [P, 2, H, S, D]");
    TORCH_CHECK(pool.size(1) == 2, "pool dim 1 must be 2 (K/V)");

    int num_pages = pool.size(0);
    int num_heads = pool.size(2);
    int page_size = pool.size(3);
    int head_dim  = pool.size(4);

    // slot_mapping: [total_tokens]
    TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1-D");
    int total_tokens = slot_mapping.size(0);

    // new_keys/values: [total_tokens, num_heads, head_dim]
    TORCH_CHECK(new_keys.dim() == 3, "new_keys must be 3-D [T, H, D]");
    TORCH_CHECK(new_values.dim() == 3, "new_values must be 3-D [T, H, D]");
    TORCH_CHECK(new_keys.size(0) == total_tokens,
                "new_keys tokens (", new_keys.size(0), ") must match slot_mapping (", total_tokens, ")");
    TORCH_CHECK(new_values.size(0) == total_tokens,
                "new_values tokens must match slot_mapping");
    TORCH_CHECK(new_keys.size(1) == num_heads, "new_keys heads must match pool");
    TORCH_CHECK(new_keys.size(2) == head_dim, "new_keys head_dim must match pool");
    TORCH_CHECK(new_values.size(1) == num_heads, "new_values heads must match pool");
    TORCH_CHECK(new_values.size(2) == head_dim, "new_values head_dim must match pool");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    flashkernel::paged_kv_cache_append(
        reinterpret_cast<half*>(pool.data_ptr<at::Half>()),
        slot_mapping.data_ptr<int>(),
        reinterpret_cast<const half*>(new_keys.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(new_values.data_ptr<at::Half>()),
        total_tokens, num_heads, page_size, head_dim, num_pages, stream
    );
}

std::vector<torch::Tensor> paged_kv_cache_read(
    torch::Tensor pool,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int max_seq_len
) {
    TORCH_CHECK(pool.device().is_cuda(), "pool must be on CUDA device");
    TORCH_CHECK(block_table.device().is_cuda(), "block_table must be on CUDA device");
    TORCH_CHECK(seq_lens.device().is_cuda(), "seq_lens must be on CUDA device");
    TORCH_CHECK(pool.is_contiguous(), "pool must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(seq_lens.is_contiguous(), "seq_lens must be contiguous");

    TORCH_CHECK(pool.scalar_type() == torch::kFloat16, "pool must be float16");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");

    TORCH_CHECK(pool.dim() == 5, "pool must be 5-D [P, 2, H, S, D]");
    TORCH_CHECK(pool.size(1) == 2, "pool dim 1 must be 2 (K/V)");
    TORCH_CHECK(block_table.dim() == 2, "block_table must be 2-D [B, max_blocks]");
    TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1-D [B]");

    int batch = block_table.size(0);
    int max_blocks_per_seq = block_table.size(1);
    int num_heads = pool.size(2);
    int page_size = pool.size(3);
    int head_dim  = pool.size(4);

    TORCH_CHECK(seq_lens.size(0) == batch,
                "seq_lens (", seq_lens.size(0), ") must match block_table batch (", batch, ")");
    TORCH_CHECK(max_seq_len > 0, "max_seq_len must be positive");

    // Allocate output (zero-filled for padding positions)
    auto K_out = torch::zeros({batch, num_heads, max_seq_len, head_dim},
                              pool.options());
    auto V_out = torch::zeros({batch, num_heads, max_seq_len, head_dim},
                              pool.options());

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    flashkernel::paged_kv_cache_read(
        reinterpret_cast<const half*>(pool.data_ptr<at::Half>()),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        reinterpret_cast<half*>(K_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(V_out.data_ptr<at::Half>()),
        batch, num_heads, max_seq_len, page_size, head_dim,
        max_blocks_per_seq, stream
    );

    return {K_out, V_out};
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

    // v1.0.2: FlashAttention
    m.def("flash_attention_forward", &flash_attention_forward,
          "FlashAttention forward pass (tiled, online softmax, no N×N materialization).\n"
          "Input: Q, K, V [B, H, N, D] fp16. Returns (O [B,H,N,D] fp16, L [B,H,N] fp32).\n"
          "Supports head_dim=64 (tiles 64×64) and head_dim=128 (tiles 32×64).",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("scale") = -1.0f,
          py::arg("is_causal") = false);

    // v1.0.4: Fused GeLU+Linear
    m.def("fused_gelu_linear", &fused_gelu_linear,
          "Fused linear projection + GeLU activation.\n"
          "Computes Y = GeLU(X @ W^T + bias) in a single kernel,\n"
          "eliminating one HBM round-trip vs unfused.\n"
          "Supports exact (erf) and tanh GeLU approximation.",
          py::arg("X"), py::arg("W"),
          py::arg("bias") = c10::nullopt,
          py::arg("use_tanh_approx") = false);

    // v1.0.5: RoPE Embedding
    m.def("rope_precompute_freqs", &rope_precompute_freqs,
          "Precompute RoPE cos/sin frequency tables on device.\n"
          "Returns (cos_table, sin_table) each [max_seq_len, head_dim/2] fp32.",
          py::arg("max_seq_len"), py::arg("head_dim"),
          py::arg("base") = 10000.0f);

    m.def("rope_forward", &rope_forward,
          "Apply Rotary Position Embedding to Q and K using precomputed tables.\n"
          "Input: Q, K [B, H, N, D] fp16. Returns (Q_rot, K_rot) fp16.\n"
          "cos_table, sin_table: [max_seq_len, D/2] fp32.",
          py::arg("Q"), py::arg("K"),
          py::arg("cos_table"), py::arg("sin_table"));

    m.def("rope_forward_fused", &rope_forward_fused,
          "Apply RoPE to Q and K with on-the-fly sin/cos computation.\n"
          "No precomputed table needed — saves HBM bandwidth.\n"
          "Input: Q, K [B, H, N, D] fp16. Returns (Q_rot, K_rot) fp16.",
          py::arg("Q"), py::arg("K"),
          py::arg("base") = 10000.0f);

    // v1.0.6: Paged KV-Cache
    m.def("paged_kv_cache_append", &paged_kv_cache_append,
          "Append new KV tokens to the paged cache pool.\n"
          "pool: [P, 2, H, S, D] fp16. slot_mapping: [T] int32.\n"
          "new_keys, new_values: [T, H, D] fp16. Modifies pool in-place.",
          py::arg("pool"), py::arg("slot_mapping"),
          py::arg("new_keys"), py::arg("new_values"));

    m.def("paged_kv_cache_read", &paged_kv_cache_read,
          "Scatter-gather read from paged KV cache.\n"
          "pool: [P, 2, H, S, D] fp16. block_table: [B, max_blocks] int32.\n"
          "seq_lens: [B] int32. Returns (K, V) each [B, H, max_seq_len, D] fp16.",
          py::arg("pool"), py::arg("block_table"),
          py::arg("seq_lens"), py::arg("max_seq_len"));

    // Version info
    m.attr("__version__") = "1.0.10";

    // v1.0.10: Showcase Polish — project page 10/10, blog fix, status complete
}
