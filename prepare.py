"""
Benchmark harness for autoresearch-inference experiments.
Discovers hardware, downloads model, runs standardized benchmarks.

This file is READ-ONLY during experiments. The agent modifies infer.py only.

Usage:
    uv run prepare.py                            # Setup with default model (Qwen 2.5 7B)
    uv run prepare.py --model "meta-llama/..."    # Use a different model
    uv run prepare.py --check                     # Verify project structure (no GPU)
    uv run prepare.py --profile                   # Profile a single prompt
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Fixed constants (never change between experiments)
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 256
NUM_PROMPTS = 20
NUM_WARMUP_RUNS = 3
TIME_BUDGET = 300  # 5 minutes wall clock for full benchmark

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "prompts.json")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-inference")

# ---------------------------------------------------------------------------
# Dynamic config (set by main(), or loaded from config.json on import)
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B"
DEVICE = "cuda"
VRAM_LIMIT_GB = 90.0
CACHE_DIR = BASE_CACHE_DIR  # backwards compat


def _load_config():
    """Auto-load config.json on import so benchmark() uses correct settings."""
    global MODEL_ID, DEVICE, VRAM_LIMIT_GB, CACHE_DIR
    config_path = os.path.join(PROJECT_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        MODEL_ID = cfg.get("model_id", MODEL_ID)
        DEVICE = cfg.get("device", DEVICE)
        VRAM_LIMIT_GB = cfg.get("vram_limit_gb", VRAM_LIMIT_GB)
        if "model_path" in cfg:
            CACHE_DIR = os.path.dirname(cfg["model_path"])

_load_config()


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------

def select_device():
    """Pick the GPU with the most free VRAM.

    Uses torch.cuda.mem_get_info() which reads from the CUDA driver and
    accounts for ALL GPU processes (not just PyTorch), so it correctly
    avoids GPUs loaded with other work like RTL simulations.
    """
    if not torch.cuda.is_available():
        return "cpu"
    if torch.cuda.device_count() == 1:
        return "cuda:0"

    best_gpu, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {name} — {free / 1024**3:.1f} GB free / {total / 1024**3:.1f} GB total")
        if free > best_free:
            best_free = free
            best_gpu = i

    selected = f"cuda:{best_gpu}"
    print(f"  → Selected: {selected} ({torch.cuda.get_device_name(best_gpu)})")
    return selected


def _model_slug(model_id):
    """Convert HuggingFace model ID to safe directory name."""
    return model_id.lower().replace("/", "-").replace(".", "-")


# ---------------------------------------------------------------------------
# Hardware & config output
# ---------------------------------------------------------------------------

def write_hardware_json(device):
    """Write hardware.json so the agent knows GPU capabilities."""
    info = {"device": device, "num_gpus": torch.cuda.device_count()}

    if device != "cpu":
        idx = int(device.split(":")[-1]) if ":" in device else 0
        free, total = torch.cuda.mem_get_info(idx)
        cap = torch.cuda.get_device_capability(idx)
        info.update({
            "gpu_name": torch.cuda.get_device_name(idx),
            "gpu_index": idx,
            "vram_total_gb": round(total / 1024**3, 1),
            "vram_free_gb": round(free / 1024**3, 1),
            "compute_capability": f"{cap[0]}.{cap[1]}",
            "cuda_version": torch.version.cuda,
            "bf16_supported": cap[0] >= 8,   # Ampere+
            "fp8_supported": cap[0] >= 9,    # Hopper+
        })

    path = os.path.join(PROJECT_DIR, "hardware.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote {path}")
    return info


def write_config_json(model_id, model_path, device, vram_limit_gb):
    """Write config.json for infer.py to read instead of importing constants."""
    config = {
        "model_id": model_id,
        "model_path": model_path,
        "device": device,
        "max_new_tokens": MAX_NEW_TOKENS,
        "vram_limit_gb": vram_limit_gb,
    }
    path = os.path.join(PROJECT_DIR, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {path}")
    return config


# ---------------------------------------------------------------------------
# Environment verification
# ---------------------------------------------------------------------------

def verify_cuda(device=None):
    """Verify CUDA is available and print GPU info for selected device."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This project requires an NVIDIA GPU.")
        print("If running --check on a non-GPU machine, this is expected.")
        return False

    idx = 0
    if device and ":" in device:
        idx = int(device.split(":")[-1])

    gpu_name = torch.cuda.get_device_name(idx)
    gpu_mem = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
    compute_cap = torch.cuda.get_device_capability(idx)
    print(f"GPU: {gpu_name} (cuda:{idx})")
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
    ok = True
    for f in required:
        path = os.path.join(PROJECT_DIR, f)
        if os.path.exists(path):
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def _find_or_download_model(model_id):
    """Find cached model or download it. Returns model path."""
    from huggingface_hub import snapshot_download

    slug = _model_slug(model_id)
    new_path = os.path.join(BASE_CACHE_DIR, slug, "model")
    legacy_path = os.path.join(BASE_CACHE_DIR, "model")

    # Check slug-based location first
    if os.path.exists(new_path) and any(Path(new_path).iterdir()):
        print(f"Model found at {new_path}")
        return new_path

    # Check legacy v1 location (v1 used flat cache dir for default model)
    if model_id == "Qwen/Qwen2.5-7B" and os.path.exists(legacy_path) and any(Path(legacy_path).iterdir()):
        print(f"Model found at legacy location: {legacy_path}")
        return legacy_path

    # Download
    os.makedirs(new_path, exist_ok=True)
    print(f"Downloading {model_id} to {new_path}...")
    print("(This may take a while on first run)")
    snapshot_download(repo_id=model_id, local_dir=new_path)
    print(f"Model downloaded to {new_path}")
    return new_path


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
    1. Output has enough new tokens
    2. Output is not all padding or EOS tokens
    3. Output does not contain excessive repetition
    4. Output decodes to readable text

    Returns (is_valid, error_message).
    """
    new_tokens = output_ids[input_length:]
    num_generated = len(new_tokens)

    # Allow tolerance for prompts that naturally complete early
    if num_generated < MAX_NEW_TOKENS * 0.05:
        return False, f"Too few tokens generated: {num_generated} (expected ~{MAX_NEW_TOKENS})"

    unique_tokens = set(new_tokens.tolist() if hasattr(new_tokens, 'tolist') else new_tokens)
    if len(unique_tokens) <= 2:
        return False, f"Output has only {len(unique_tokens)} unique tokens (likely garbage)"

    from collections import Counter
    token_counts = Counter(new_tokens.tolist() if hasattr(new_tokens, 'tolist') else new_tokens)
    most_common_count = token_counts.most_common(1)[0][1]
    if most_common_count > num_generated * 0.5:
        return False, f"Excessive repetition: most common token appears {most_common_count}/{num_generated} times"

    try:
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if len(text.strip()) < 10:
            return False, f"Decoded text too short: '{text[:50]}'"
    except Exception as e:
        return False, f"Failed to decode output: {e}"

    return True, ""


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def run_profile(generate_fn, tokenizer):
    """Profile a single prompt and write profile.txt.

    Helps the agent understand where time is spent (attention vs linear vs
    layernorm) and make data-driven optimization decisions.
    """
    from torch.profiler import profile, ProfilerActivity

    prompts = load_prompts()
    input_ids = tokenizer.encode(prompts[0]["text"], return_tensors="pt")

    # Warmup
    print("Warming up for profiling...")
    for _ in range(2):
        generate_fn(input_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize(DEVICE)

    print("Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        generate_fn(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize(DEVICE)

    # Write profile
    path = os.path.join(PROJECT_DIR, "profile.txt")
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)

    with open(path, "w") as f:
        f.write("=== Profiler Output (sorted by CUDA time) ===\n\n")
        f.write(table)
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
            f.write(f"\nPeak GPU memory: {peak:.1f} GB\n")

    print(f"Wrote {path}")
    # Print truncated summary to stdout
    print(table[:2000])


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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    # --- Warmup ---
    print(f"Running {NUM_WARMUP_RUNS} warmup iterations...")
    warmup_prompt = prompts[0]
    warmup_ids = tokenizer.encode(warmup_prompt["text"], return_tensors="pt")
    for _ in range(NUM_WARMUP_RUNS):
        _ = generate_fn(warmup_ids)
    if torch.cuda.is_available():
        torch.cuda.synchronize(DEVICE)
    print("Warmup complete.")
    print()

    # --- Benchmark ---
    total_gen_time = 0.0
    total_tokens_generated = 0
    ttft_values = []
    valid_count = 0
    invalid_count = 0

    # Per-prompt records for category breakdown
    from collections import defaultdict
    cat_records = defaultdict(list)  # category -> list of (tok_s, is_valid)

    print(f"Benchmarking {NUM_PROMPTS} prompts (generating {MAX_NEW_TOKENS} tokens each)...")
    for i, prompt_data in enumerate(prompts):
        input_ids = tokenizer.encode(prompt_data["text"], return_tensors="pt")
        input_length = input_ids.shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize(DEVICE)

        t_start = time.perf_counter()
        output_ids = generate_fn(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize(DEVICE)
        t_end = time.perf_counter()

        gen_time = t_end - t_start
        total_gen_time += gen_time

        if output_ids.dim() > 1:
            output_ids = output_ids[0]

        num_new = len(output_ids) - input_length
        total_tokens_generated += num_new

        ttft_est = gen_time / max(num_new, 1) * 1000  # ms per token as proxy
        ttft_values.append(ttft_est)

        is_valid, error = validate_output(output_ids, input_length, tokenizer)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            print(f"  WARNING: Prompt {prompt_data['id']} invalid output: {error}")

        tok_s_this = num_new / gen_time if gen_time > 0 else 0
        cat = prompt_data.get("category", "unknown")
        cat_records[cat].append((tok_s_this, is_valid))

        status = "OK" if is_valid else "INVALID"
        print(f"  [{i+1:2d}/{NUM_PROMPTS}] {cat:8s} | "
              f"{num_new:3d} tokens | {gen_time:.2f}s | "
              f"{tok_s_this:.1f} tok/s | {status}")

    # --- Compute metrics ---
    tok_s = total_tokens_generated / total_gen_time if total_gen_time > 0 else 0
    ttft_ms = sorted(ttft_values)[len(ttft_values) // 2]  # median

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)

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

    # --- Per-category breakdown ---
    print()
    print("--- category breakdown ---")
    for cat in sorted(cat_records.keys()):
        records = cat_records[cat]
        cat_tok_s_vals = [r[0] for r in records]
        cat_valid_count = sum(1 for r in records if r[1])
        avg_tok_s = sum(cat_tok_s_vals) / len(cat_tok_s_vals) if cat_tok_s_vals else 0
        print(f"  {cat:10s}: avg {avg_tok_s:6.1f} tok/s | "
              f"{cat_valid_count}/{len(records)} valid")

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
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-7B)")
    parser.add_argument("--check", action="store_true",
                        help="Only verify project structure (no GPU needed)")
    parser.add_argument("--profile", action="store_true",
                        help="Profile a single prompt and write profile.txt")
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

    # --- GPU Discovery ---
    print("Discovering GPUs...")
    device = select_device()
    print()

    if not verify_cuda(device):
        print("Run with --check to verify project structure without a GPU.")
        sys.exit(1)

    # Set module globals for this run
    MODEL_ID = args.model
    DEVICE = device

    # VRAM limit = 90% of selected GPU's total memory
    if device != "cpu":
        idx = int(device.split(":")[-1]) if ":" in device else 0
        _, total = torch.cuda.mem_get_info(idx)
        VRAM_LIMIT_GB = round(total / 1024**3 * 0.9, 1)
        print(f"VRAM limit: {VRAM_LIMIT_GB} GB (90% of total)")

    # Write hardware context for agent
    write_hardware_json(device)

    # --- Model ---
    print()
    model_path = _find_or_download_model(args.model)

    # Write config for infer.py
    write_config_json(args.model, model_path, device, VRAM_LIMIT_GB)
    print()

    # --- Benchmark or Profile ---
    print("Importing infer.py...")
    try:
        from infer import run_inference
        generate_fn, tokenizer = run_inference()

        if args.profile:
            run_profile(generate_fn, tokenizer)
        else:
            print("Running baseline benchmark...")
            results = benchmark(generate_fn, tokenizer)
    except Exception as e:
        print(f"\nERROR: Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 60)
    print("Setup complete! Ready to experiment.")
    if not args.profile:
        print(f"Baseline tok/s: {results['tok_s']:.2f}")
    print("=" * 60)
