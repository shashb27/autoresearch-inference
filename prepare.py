"""
Benchmark harness for autoresearch-inference experiments.
Downloads model, validates environment, runs standardized benchmarks.

This file is READ-ONLY. The agent must NOT modify it.
The agent modifies infer.py only.

Usage:
    uv run prepare.py              # Full setup: download model + run baseline
    uv run prepare.py --check      # Verify project structure only (no GPU needed)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B"
MAX_NEW_TOKENS = 256
NUM_PROMPTS = 20
NUM_WARMUP_RUNS = 3
TIME_BUDGET = 300  # 5 minutes wall clock for full benchmark
VRAM_LIMIT_GB = 90.0
DEVICE = "cuda"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-inference")
PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "prompts.json")

# Canonical output for model identity verification (first 10 tokens of greedy
# generation from "The capital of France is"). Set to None to skip verification
# on first run, then update after establishing baseline.
CANONICAL_OUTPUT_IDS = None

# ---------------------------------------------------------------------------
# Environment verification
# ---------------------------------------------------------------------------

def verify_cuda():
    """Verify CUDA is available and print GPU info."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This project requires an NVIDIA GPU.")
        print("If running --check on a non-GPU machine, this is expected.")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gpu_mem:.1f} GB")
    print(f"Compute capability: {compute_cap[0]}.{compute_cap[1]}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    return True


def check_project_structure():
    """Verify all required project files exist."""
    required = [
        "pyproject.toml",
        "prepare.py",
        "infer.py",
        "program.md",
        "prompts/prompts.json",
    ]
    project_dir = os.path.dirname(os.path.abspath(__file__))
    ok = True
    for f in required:
        path = os.path.join(project_dir, f)
        if os.path.exists(path):
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            ok = False
    return ok

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_model():
    """Download Qwen 2.5 7B to local cache if not present."""
    from huggingface_hub import snapshot_download

    model_cache = os.path.join(CACHE_DIR, "model")
    os.makedirs(model_cache, exist_ok=True)

    print(f"Downloading {MODEL_ID} to {model_cache}...")
    print("(This may take a while on first run)")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=model_cache,
    )
    print(f"Model downloaded to {model_cache}")
    print()
    return model_cache

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts():
    """Load fixed benchmark prompts from prompts.json."""
    with open(PROMPT_FILE, "r") as f:
        data = json.load(f)
    prompts = data["prompts"]
    assert len(prompts) == NUM_PROMPTS, (
        f"Expected {NUM_PROMPTS} prompts, got {len(prompts)}"
    )
    return prompts

# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_output(output_ids, input_length, tokenizer):
    """Validate generated output is real text, not garbage.

    Checks:
    1. Output has the expected number of new tokens
    2. Output is not all padding or EOS tokens
    3. Output does not contain excessive repetition
    4. Output decodes to readable text

    Returns (is_valid, error_message).
    """
    new_tokens = output_ids[input_length:]
    num_generated = len(new_tokens)

    # Check token count (allow tolerance for early EOS on some prompts)
    if num_generated < MAX_NEW_TOKENS * 0.05:
        return False, f"Too few tokens generated: {num_generated} (expected ~{MAX_NEW_TOKENS})"

    # Check for all-padding or all-EOS
    unique_tokens = set(new_tokens.tolist() if hasattr(new_tokens, 'tolist') else new_tokens)
    if len(unique_tokens) <= 2:
        return False, f"Output has only {len(unique_tokens)} unique tokens (likely garbage)"

    # Check for excessive repetition (same token >50% of output)
    from collections import Counter
    token_counts = Counter(new_tokens.tolist() if hasattr(new_tokens, 'tolist') else new_tokens)
    most_common_count = token_counts.most_common(1)[0][1]
    if most_common_count > num_generated * 0.5:
        return False, f"Excessive repetition: most common token appears {most_common_count}/{num_generated} times"

    # Check it decodes to real text
    try:
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if len(text.strip()) < 10:
            return False, f"Decoded text too short: '{text[:50]}'"
    except Exception as e:
        return False, f"Failed to decode output: {e}"

    return True, ""

# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def benchmark(generate_fn, tokenizer):
    """Run the standardized inference benchmark.

    Args:
        generate_fn: Callable that takes input_ids (tensor) and returns
                     output_ids (tensor) with MAX_NEW_TOKENS new tokens.
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        dict with keys: tok_s, ttft_ms, peak_vram_gb, total_prompts,
                        total_tokens, valid_outputs
    """
    prompts = load_prompts()

    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- Warmup ---
    print(f"Running {NUM_WARMUP_RUNS} warmup iterations...")
    warmup_prompt = prompts[0]
    warmup_ids = tokenizer.encode(warmup_prompt["text"], return_tensors="pt")
    for _ in range(NUM_WARMUP_RUNS):
        _ = generate_fn(warmup_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Warmup complete.")
    print()

    # --- Benchmark ---
    total_gen_time = 0.0
    total_tokens_generated = 0
    ttft_values = []
    valid_count = 0
    invalid_count = 0

    print(f"Benchmarking {NUM_PROMPTS} prompts (generating {MAX_NEW_TOKENS} tokens each)...")
    for i, prompt_data in enumerate(prompts):
        input_ids = tokenizer.encode(prompt_data["text"], return_tensors="pt")
        input_length = input_ids.shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        output_ids = generate_fn(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        gen_time = t_end - t_start
        total_gen_time += gen_time

        # Flatten output if batched
        if output_ids.dim() > 1:
            output_ids = output_ids[0]

        num_new = len(output_ids) - input_length
        total_tokens_generated += num_new

        # Time to first token approximation (total_time / num_tokens for first token)
        # In a more sophisticated setup, we'd hook into the generation loop
        # For now, estimate TTFT as gen_time * (1 / num_new) * some factor
        # A rough approximation: TTFT ~ total_time / num_tokens (prefill-dominated)
        ttft_est = gen_time / max(num_new, 1) * 1000  # ms per token as proxy
        ttft_values.append(ttft_est)

        # Validate output
        is_valid, error = validate_output(output_ids, input_length, tokenizer)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            print(f"  WARNING: Prompt {prompt_data['id']} invalid output: {error}")

        status = "OK" if is_valid else "INVALID"
        tok_s_this = num_new / gen_time if gen_time > 0 else 0
        print(f"  [{i+1:2d}/{NUM_PROMPTS}] {prompt_data['category']:6s} | "
              f"{num_new:3d} tokens | {gen_time:.2f}s | "
              f"{tok_s_this:.1f} tok/s | {status}")

    # --- Compute metrics ---
    tok_s = total_tokens_generated / total_gen_time if total_gen_time > 0 else 0
    ttft_ms = sorted(ttft_values)[len(ttft_values) // 2]  # median

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    results = {
        "tok_s": tok_s,
        "ttft_ms": ttft_ms,
        "peak_vram_gb": peak_vram_gb,
        "total_prompts": NUM_PROMPTS,
        "total_tokens": total_tokens_generated,
        "valid_outputs": valid_count,
        "invalid_outputs": invalid_count,
    }

    # --- Print results (parseable format) ---
    print()
    print("---")
    print(f"tok_s:            {results['tok_s']:.2f}")
    print(f"ttft_ms:          {results['ttft_ms']:.2f}")
    print(f"peak_vram_gb:     {results['peak_vram_gb']:.1f}")
    print(f"total_prompts:    {results['total_prompts']}")
    print(f"total_tokens:     {results['total_tokens']}")
    print(f"valid_outputs:    {results['valid_outputs']}")
    print(f"invalid_outputs:  {results['invalid_outputs']}")

    # --- Safety checks ---
    if peak_vram_gb > VRAM_LIMIT_GB:
        print(f"\nWARNING: Peak VRAM ({peak_vram_gb:.1f} GB) exceeds limit ({VRAM_LIMIT_GB} GB)")

    if invalid_count > NUM_PROMPTS * 0.2:
        print(f"\nWARNING: {invalid_count}/{NUM_PROMPTS} outputs were invalid")

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup and baseline for autoresearch-inference")
    parser.add_argument("--check", action="store_true",
                        help="Only verify project structure (no GPU needed)")
    args = parser.parse_args()

    print("=" * 60)
    print("autoresearch-inference: Setup & Baseline")
    print("=" * 60)
    print()

    # Always check project structure
    print("Checking project structure...")
    if not check_project_structure():
        print("\nERROR: Missing required files. Check the README.")
        sys.exit(1)
    print()

    if args.check:
        print("Structure check passed. Run without --check to download model and benchmark.")
        sys.exit(0)

    # Verify CUDA
    if not verify_cuda():
        print("Run with --check to verify project structure without a GPU.")
        sys.exit(1)

    # Download model
    download_model()

    # Run baseline benchmark using infer.py
    print("Running baseline benchmark...")
    print("Importing infer.py...")
    try:
        from infer import run_inference
        generate_fn, tokenizer = run_inference()
        results = benchmark(generate_fn, tokenizer)
    except Exception as e:
        print(f"\nERROR: Baseline benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 60)
    print("Setup complete! Ready to experiment.")
    print(f"Baseline tok/s: {results['tok_s']:.2f}")
    print("=" * 60)
