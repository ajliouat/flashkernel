#!/usr/bin/env python3
"""
FlashKernel — GPT-2 End-to-End Benchmark (v1.0.7)

Measures real tokens/sec for GPT-2-124M text generation with:
  1. HuggingFace default (vanilla PyTorch)
  2. torch.compile (PyTorch 2.x optimized)
  3. FlashKernel — our custom CUDA/Triton kernels

Prompt lengths: [32, 128, 512], generate 128 tokens each.
Also verifies greedy decoding produces identical output across backends.

Usage:
    python -m benchmarks.bench_e2e_gpt2 [--gen-tokens 128] [--warmup 3] [--timed 10]
"""

import argparse
import csv
import gc
import os
import time
from contextlib import contextmanager

import torch

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "gpt2"  # GPT-2-124M
PROMPT_LENGTHS = [32, 128, 512]
DEFAULT_GEN_TOKENS = 128
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CSV_PATH = os.path.join(RESULTS_DIR, "e2e_gpt2.csv")


# ─── Helpers ─────────────────────────────────────────────────────────────────

@contextmanager
def track_memory():
    """Context manager to track peak GPU memory."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()


def get_peak_memory_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def make_prompt(tokenizer, length: int) -> torch.Tensor:
    """Create a prompt of approximately `length` tokens."""
    # Use a repeated sentence to fill to desired length
    base = "The quick brown fox jumps over the lazy dog. "
    text = base * (length // 8 + 1)
    ids = tokenizer.encode(text, return_tensors="pt")
    return ids[:, :length].cuda()


def measure_generation(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    warmup: int = 3,
    timed: int = 10,
) -> dict:
    """
    Measure tokens/sec for model.generate().

    Returns:
        dict with keys: mean_tok_per_sec, std_tok_per_sec, mean_ms,
                        peak_mem_mb, output_ids
    """
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    torch.cuda.synchronize()

    # Timed runs
    times_ms = []
    last_output = None

    torch.cuda.reset_peak_memory_stats()

    for _ in range(timed):
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)
        last_output = output

    # Stats
    mean_ms = sum(times_ms) / len(times_ms)
    std_ms = (
        (sum((t - mean_ms) ** 2 for t in times_ms) / (len(times_ms) - 1)) ** 0.5
        if len(times_ms) > 1
        else 0.0
    )
    generated_tokens = max_new_tokens
    mean_tok_per_sec = generated_tokens / (mean_ms / 1000)
    std_tok_per_sec = (
        generated_tokens / ((mean_ms - std_ms) / 1000)
        - generated_tokens / ((mean_ms + std_ms) / 1000)
    ) / 2 if std_ms > 0 else 0.0

    return {
        "mean_tok_per_sec": mean_tok_per_sec,
        "std_tok_per_sec": std_tok_per_sec,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "peak_mem_mb": get_peak_memory_mb(),
        "output_ids": last_output,
    }


# ─── Backend Runners ─────────────────────────────────────────────────────────

def run_hf_default(model, tokenizer, prompt_len, gen_tokens, warmup, timed):
    """Baseline: HuggingFace default generation."""
    input_ids = make_prompt(tokenizer, prompt_len)
    result = measure_generation(model, input_ids, gen_tokens, warmup, timed)
    return {
        "backend": "hf_default",
        "prompt_len": prompt_len,
        **result,
    }


def run_torch_compile(model, tokenizer, prompt_len, gen_tokens, warmup, timed):
    """Backend: torch.compile() wrapped model."""
    compiled_model = torch.compile(model, mode="reduce-overhead")
    input_ids = make_prompt(tokenizer, prompt_len)
    result = measure_generation(compiled_model, input_ids, gen_tokens, warmup, timed)
    return {
        "backend": "torch_compile",
        "prompt_len": prompt_len,
        **result,
    }


def run_flashkernel(model, tokenizer, prompt_len, gen_tokens, warmup, timed,
                    backend="cuda"):
    """Backend: FlashKernel custom kernels via monkey-patching."""
    from src.integration.gpt2_custom_kernels import patch_gpt2_model, unpatch_gpt2_model

    patch_gpt2_model(model, backend=backend)
    input_ids = make_prompt(tokenizer, prompt_len)
    result = measure_generation(model, input_ids, gen_tokens, warmup, timed)
    unpatch_gpt2_model(model)

    return {
        "backend": f"flashkernel_{backend}",
        "prompt_len": prompt_len,
        **result,
    }


# ─── Greedy Decoding Verification ───────────────────────────────────────────

def verify_greedy_identity(model, tokenizer, prompt_len=32, gen_tokens=64):
    """
    Verify that FlashKernel-patched model produces the same greedy
    output as the default HF model (bit-exact token match).
    """
    from src.integration.gpt2_custom_kernels import patch_gpt2_model, unpatch_gpt2_model

    input_ids = make_prompt(tokenizer, prompt_len)

    # Default HF output
    with torch.no_grad():
        hf_output = model.generate(input_ids, max_new_tokens=gen_tokens, do_sample=False)

    # FlashKernel output
    patch_gpt2_model(model, backend="cuda")
    with torch.no_grad():
        fk_output = model.generate(input_ids, max_new_tokens=gen_tokens, do_sample=False)
    unpatch_gpt2_model(model)

    hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
    fk_text = tokenizer.decode(fk_output[0], skip_special_tokens=True)

    match = torch.equal(hf_output, fk_output)

    print(f"\n{'─' * 60}")
    print(f"  Greedy Decoding Verification")
    print(f"{'─' * 60}")
    print(f"  Prompt tokens: {prompt_len}")
    print(f"  Generated tokens: {gen_tokens}")
    print(f"  Token-exact match: {'✓ PASS' if match else '✗ FAIL'}")

    if not match:
        # Show where they diverge
        hf_ids = hf_output[0].tolist()
        fk_ids = fk_output[0].tolist()
        for i, (a, b) in enumerate(zip(hf_ids, fk_ids)):
            if a != b:
                print(f"  First divergence at position {i}: HF={a} FK={b}")
                break
        print(f"  HF text (first 200 chars): {hf_text[:200]}")
        print(f"  FK text (first 200 chars): {fk_text[:200]}")
    else:
        print(f"  Output: {hf_text[:200]}...")

    print(f"{'─' * 60}\n")
    return match


# ─── CSV Export ──────────────────────────────────────────────────────────────

def write_results_csv(results: list[dict]):
    """Write benchmark results to CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fieldnames = [
        "backend", "prompt_len", "gen_tokens",
        "mean_tok_per_sec", "std_tok_per_sec",
        "mean_ms", "std_ms", "peak_mem_mb",
    ]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: round(v, 2) if isinstance(v, float) else v
                             for k, v in r.items() if k in fieldnames})

    print(f"Results written to {CSV_PATH}")


# ─── Pretty-Print ────────────────────────────────────────────────────────────

def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'═' * 90}")
    print(f"  GPT-2-124M End-to-End Generation Benchmark")
    print(f"{'═' * 90}")
    print(f"  {'Backend':<25} {'Prompt':>7} {'Tok/s':>10} {'±':>8} "
          f"{'Latency':>10} {'Mem MB':>8}")
    print(f"{'─' * 90}")

    # Group by prompt length for comparison
    baseline_by_prompt = {}
    for r in results:
        if r["backend"] == "hf_default":
            baseline_by_prompt[r["prompt_len"]] = r["mean_tok_per_sec"]

    for r in results:
        speedup = ""
        bl = baseline_by_prompt.get(r["prompt_len"])
        if bl and bl > 0 and r["backend"] != "hf_default":
            speedup = f" ({r['mean_tok_per_sec']/bl:.2f}×)"

        print(f"  {r['backend']:<25} {r['prompt_len']:>7} "
              f"{r['mean_tok_per_sec']:>10.1f} {r.get('std_tok_per_sec', 0):>8.1f} "
              f"{r['mean_ms']:>8.1f}ms {r['peak_mem_mb']:>8.1f}"
              f"{speedup}")

    print(f"{'═' * 90}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPT-2 E2E Benchmark")
    parser.add_argument("--gen-tokens", type=int, default=DEFAULT_GEN_TOKENS,
                        help="Number of tokens to generate")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations")
    parser.add_argument("--timed", type=int, default=10,
                        help="Timed iterations")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip greedy decoding verification")
    parser.add_argument("--prompt-lengths", type=int, nargs="+",
                        default=PROMPT_LENGTHS,
                        help="Prompt lengths to benchmark")
    args = parser.parse_args()

    # ── Load model and tokenizer ────────────────────────────────────────
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).cuda().half()
    model.eval()

    config = model.config
    print(f"  Layers: {config.n_layer}, Heads: {config.n_head}, "
          f"Hidden: {config.n_embd}, Params: "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    print(f"  Generate: {args.gen_tokens} tokens, "
          f"Prompt lengths: {args.prompt_lengths}")

    # ── Verify greedy decoding identity ─────────────────────────────────
    if not args.skip_verify:
        print("\nVerifying greedy decoding identity...")
        match = verify_greedy_identity(model, tokenizer)
        if not match:
            print("⚠ WARNING: Greedy outputs differ. Continuing benchmark...")

    # ── Run benchmarks ──────────────────────────────────────────────────
    all_results = []

    for prompt_len in args.prompt_lengths:
        print(f"\n─── Prompt length: {prompt_len} tokens ───")

        # 1. HF default
        print("  [1/3] HF default...")
        r = run_hf_default(model, tokenizer, prompt_len,
                           args.gen_tokens, args.warmup, args.timed)
        r["gen_tokens"] = args.gen_tokens
        all_results.append(r)

        gc.collect()
        torch.cuda.empty_cache()

        # 2. torch.compile
        print("  [2/3] torch.compile...")
        try:
            r = run_torch_compile(model, tokenizer, prompt_len,
                                  args.gen_tokens, args.warmup, args.timed)
            r["gen_tokens"] = args.gen_tokens
            all_results.append(r)
        except Exception as e:
            print(f"    torch.compile failed: {e}")

        gc.collect()
        torch.cuda.empty_cache()

        # 3. FlashKernel (CUDA)
        print("  [3/3] FlashKernel...")
        try:
            r = run_flashkernel(model, tokenizer, prompt_len,
                                args.gen_tokens, args.warmup, args.timed,
                                backend="cuda")
            r["gen_tokens"] = args.gen_tokens
            all_results.append(r)
        except Exception as e:
            print(f"    FlashKernel failed: {e}")

        gc.collect()
        torch.cuda.empty_cache()

    # ── Results ─────────────────────────────────────────────────────────
    print_comparison_table(all_results)
    write_results_csv(all_results)


if __name__ == "__main__":
    main()
