"""
FlashKernel — GPT-2 Custom Kernel Integration (v1.0.7)

Replaces PyTorch's default attention and MLP computations in GPT-2-124M
with FlashKernel CUDA/Triton kernels, then benchmarks real tokens/sec.

Strategy:
  1. Attention: Replace the Q@K^T → softmax → @V step with our
     FlashAttention (tiled, online softmax, no N×N materialization).
  2. MLP: Replace the c_fc → GeLU step with our fused GeLU+Linear
     kernel (single HBM round-trip instead of two).
  3. KV-Cache: Optionally integrate PagedKVCache for autoregressive
     generation (reduces memory for variable-length batches).

GPT-2-124M architecture:
  - 12 layers, 12 heads, head_dim=64, hidden=768
  - Absolute positional embeddings (no RoPE needed)
  - MLP: c_fc (768→3072, GeLU_tanh_approx), c_proj (3072→768)
  - HF Conv1D weights: [in_features, out_features] (transposed vs nn.Linear)

Usage:
    from src.integration.gpt2_custom_kernels import (
        patch_gpt2_model,
        unpatch_gpt2_model,
    )
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().half()
    patch_gpt2_model(model)
    output = model.generate(input_ids, max_new_tokens=128)
    unpatch_gpt2_model(model)

Reference: Radford et al., "Language Models are Unsupervised Multitask
           Learners" (GPT-2, 2019)
"""

import math
from typing import Optional
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Attention Patch ─────────────────────────────────────────────────────────

def _make_custom_attention_forward(original_forward, attn_module, backend="cuda"):
    """
    Build a replacement forward for GPT2Attention that uses FlashKernel.

    HF GPT2Attention.forward signature:
        forward(hidden_states, layer_past=None, attention_mask=None,
                head_mask=None, use_cache=False, output_attentions=False)

    We intercept after Q/K/V projection and replace the attention
    computation with our FlashAttention kernel.
    """

    def custom_attention_forward(
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs,
    ):
        # ── Step 1: Project to Q, K, V using the existing c_attn ────────
        # c_attn: Conv1D(3*n_embd, n_embd)
        # hidden_states: [batch, seq_len, n_embd]
        qkv = attn_module.c_attn(hidden_states)  # [B, N, 3*H*D]

        # Split into Q, K, V
        query, key, value = qkv.split(attn_module.split_size, dim=2)

        # Reshape to multi-head: [B, N, H*D] → [B, H, N, D]
        B, N, _ = query.shape
        num_heads = attn_module.num_heads
        head_dim = attn_module.head_dim

        query = query.view(B, N, num_heads, head_dim).transpose(1, 2)
        key = key.view(B, N, num_heads, head_dim).transpose(1, 2)
        value = value.view(B, N, num_heads, head_dim).transpose(1, 2)

        # ── Step 2: Handle KV-cache (past) ──────────────────────────────
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key, key], dim=2)      # [B, H, past+N, D]
            value = torch.cat([past_value, value], dim=2)

        present = (key, value) if use_cache else None

        # ── Step 3: FlashAttention ──────────────────────────────────────
        # Ensure fp16 for our kernels
        q_fp16 = query.half() if query.dtype != torch.float16 else query
        k_fp16 = key.half() if key.dtype != torch.float16 else key
        v_fp16 = value.half() if value.dtype != torch.float16 else value

        # Ensure contiguous
        q_fp16 = q_fp16.contiguous()
        k_fp16 = k_fp16.contiguous()
        v_fp16 = v_fp16.contiguous()

        scale = 1.0 / math.sqrt(head_dim)

        if backend == "triton":
            import flashkernel
            attn_output, _ = flashkernel.triton_flash_attention_forward(
                q_fp16, k_fp16, v_fp16, scale=scale, is_causal=True
            )
        else:
            import flashkernel
            attn_output, _ = flashkernel.flash_attention_forward(
                q_fp16, k_fp16, v_fp16, scale=scale, is_causal=True
            )

        # Cast back if needed
        if query.dtype != torch.float16:
            attn_output = attn_output.to(query.dtype)

        # ── Step 4: Reshape and project output ──────────────────────────
        # [B, H, N, D] → [B, N, H*D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, num_heads * head_dim)

        # Apply output projection
        attn_output = attn_module.c_proj(attn_output)
        attn_output = attn_module.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            # We don't compute attention weights with FlashAttention
            # Return None for attention weights (common practice)
            outputs += (None,)

        return outputs

    return custom_attention_forward


# ─── MLP Patch ───────────────────────────────────────────────────────────────

def _make_custom_mlp_forward(original_forward, mlp_module, backend="cuda"):
    """
    Build a replacement forward for GPT2MLP that uses fused GeLU+Linear.

    HF GPT2MLP.forward:
        h = self.act(self.c_fc(hidden_states))   ← fuse this
        h = self.c_proj(h)
        h = self.dropout(h)
        return h

    Our fused_gelu_linear computes Y = GeLU(X @ W^T + bias), which
    replaces c_fc + GeLU in a single kernel.

    Conv1D stores weights as [in, out], so weight.T gives [out, in]
    which is the [N, K] layout our kernel expects.
    """

    def custom_mlp_forward(hidden_states, **kwargs):
        import flashkernel

        B, N, C = hidden_states.shape

        # Flatten to 2D for our kernel
        x_2d = hidden_states.view(-1, C)  # [B*N, C]

        # Ensure fp16
        x_fp16 = x_2d.half() if x_2d.dtype != torch.float16 else x_2d

        # Conv1D weight: [in_features, out_features] = [C, 4C]
        # Our kernel expects W: [N, K] = [4C, C] → use weight.T
        w = mlp_module.c_fc.weight.T.contiguous()  # [4C, C]
        w_fp16 = w.half() if w.dtype != torch.float16 else w

        bias = mlp_module.c_fc.bias
        b_fp16 = bias.half() if bias is not None and bias.dtype != torch.float16 else bias

        # Fused GeLU+Linear: Y = GeLU(X @ W^T + bias)
        # This is: GeLU(x @ c_fc.weight + c_fc.bias) in one kernel
        if backend == "triton":
            h = flashkernel.triton_fused_gelu_linear(
                x_fp16, w_fp16, bias=b_fp16, use_tanh_approx=True
            )
        else:
            h = flashkernel.fused_gelu_linear(
                x_fp16, w_fp16, bias=b_fp16, use_tanh_approx=True
            )

        # Cast back if needed
        if hidden_states.dtype != torch.float16:
            h = h.to(hidden_states.dtype)

        # Reshape back to 3D
        h = h.view(B, N, -1)

        # Output projection + dropout (keep as-is)
        h = mlp_module.c_proj(h)
        h = mlp_module.dropout(h)

        return h

    return custom_mlp_forward


# ─── Model Patching API ─────────────────────────────────────────────────────

_ORIGINAL_FORWARDS = {}  # module id → original forward (for unpatching)


def patch_gpt2_attention(model, backend="cuda"):
    """
    Replace attention in all GPT-2 transformer blocks with FlashAttention.

    Args:
        model: GPT2LMHeadModel instance
        backend: 'cuda' or 'triton'
    """
    for i, block in enumerate(model.transformer.h):
        attn = block.attn
        key = f"attn_{id(attn)}"
        if key not in _ORIGINAL_FORWARDS:
            _ORIGINAL_FORWARDS[key] = attn.forward
        attn.forward = _make_custom_attention_forward(
            attn.forward, attn, backend=backend
        )


def patch_gpt2_mlp(model, backend="cuda"):
    """
    Replace MLP in all GPT-2 transformer blocks with fused GeLU+Linear.

    Args:
        model: GPT2LMHeadModel instance
        backend: 'cuda' or 'triton'
    """
    for i, block in enumerate(model.transformer.h):
        mlp = block.mlp
        key = f"mlp_{id(mlp)}"
        if key not in _ORIGINAL_FORWARDS:
            _ORIGINAL_FORWARDS[key] = mlp.forward
        mlp.forward = _make_custom_mlp_forward(
            mlp.forward, mlp, backend=backend
        )


def patch_gpt2_model(model, backend="cuda", patch_attention=True, patch_mlp=True):
    """
    Patch GPT-2 model to use FlashKernel custom kernels.

    Args:
        model: GPT2LMHeadModel instance (must be on CUDA, fp16)
        backend: 'cuda' or 'triton'
        patch_attention: Whether to replace attention with FlashAttention
        patch_mlp: Whether to replace MLP with fused GeLU+Linear

    Example:
        model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().half()
        patch_gpt2_model(model)
        output = model.generate(input_ids, max_new_tokens=128)
    """
    if patch_attention:
        patch_gpt2_attention(model, backend=backend)
    if patch_mlp:
        patch_gpt2_mlp(model, backend=backend)


def unpatch_gpt2_model(model):
    """
    Restore original forward methods for all patched modules.
    """
    for block in model.transformer.h:
        attn = block.attn
        key = f"attn_{id(attn)}"
        if key in _ORIGINAL_FORWARDS:
            attn.forward = _ORIGINAL_FORWARDS.pop(key)

        mlp = block.mlp
        key = f"mlp_{id(mlp)}"
        if key in _ORIGINAL_FORWARDS:
            mlp.forward = _ORIGINAL_FORWARDS.pop(key)


# ─── Utilities ───────────────────────────────────────────────────────────────

def count_patched_modules(model) -> dict:
    """Count how many attention/mlp modules are patched."""
    patched_attn = 0
    patched_mlp = 0
    for block in model.transformer.h:
        if f"attn_{id(block.attn)}" in _ORIGINAL_FORWARDS:
            patched_attn += 1
        if f"mlp_{id(block.mlp)}" in _ORIGINAL_FORWARDS:
            patched_mlp += 1
    return {"attention": patched_attn, "mlp": patched_mlp}


def get_gpt2_config_info(model) -> dict:
    """Extract GPT-2 architecture info for logging."""
    config = model.config
    return {
        "model_name": getattr(config, "_name_or_path", "gpt2"),
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "head_dim": config.n_embd // config.n_head,
        "vocab_size": config.vocab_size,
        "max_position": config.n_positions,
        "n_params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }
