"""
Benchmark harness for autoresearch-inference experiments.
Discovers hardware, downloads model, runs standardized benchmarks.

This file is READ-ONLY during experiments. The agent modifies infer.py only.

Usage:
    uv run prepare.py                            # Setup with default model (Qwen 2.5 7B)
    uv run prepare.py --model "meta-llama/..."    # Use a different model
    uv run prepare.py --check                     # Verify project structure (no GPU)
    uv run prepare.py --profile                   # Profile a single prompt
    uv run prepare.py --validate                  # Validate infer.py without benchmarking
"""

import os
import sys
import gc
import json
import time
import signal
import threading
import argparse
import subprocess
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Fixed constants (never change between experiments)
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 256
NUM_PROMPTS = 20
NUM_WARMUP_RUNS = 3
TIME_BUDGET = 300          # 5 minutes wall clock for full benchmark
GENERATION_TIMEOUT = 120   # seconds before a single generate_fn call is killed
MIN_OUTPUT_RATIO = 0.25    # require at least 25% of MAX_NEW_TOKENS generated

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "prompts", "prompts.json")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-inference")

# ---------------------------------------------------------------------------
# Config / Hardware schema (dataclasses for validation)
# ---------------------------------------------------------------------------

@dataclass
class InferConfig:
    model_id: str
    model_path: str
    device: str
    max_new_tokens: int
    vram_limit_gb: float

    @classmethod
    def from_json(cls, path: str) -> "InferConfig":
        if not os.path.exists(path):
            raise FileNotFoundError(f"config.json not found at {path}. Run prepare.py first.")
        with open(path) as f:
            raw = json.load(f)
        required = {"model_id", "model_path", "device", "max_new_tokens", "vram_limit_gb"}
        missing = required - set(raw.keys())
        if missing:
            raise ValueError(f"config.json missing required keys: {missing}")
        return cls(
            model_id=str(raw["model_id"]),
            model_path=str(raw["model_path"]),
            device=str(raw["device"]),
            max_new_tokens=int(raw["max_new_tokens"]),
            vram_limit_gb=float(raw["vram_limit_gb"]),
        )


@dataclass
class HardwareInfo:
    device: str
    num_gpus: int
    gpu_name: str = ""
    gpu_index: int = 0
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    compute_capability: str = ""
    cuda_version: str = ""
    bf16_supported: bool = False
    fp8_supported: bool = False

    @classmethod
    def from_json(cls, path: str) -> "HardwareInfo":
        if not os.path.exists(path):
            raise FileNotFoundError(f"hardware.json not found at {path}. Run prepare.py first.")
        with open(path) as f:
            raw = json.load(f)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


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
        try:
            cfg = InferConfig.from_json(config_path)
            MODEL_ID = cfg.model_id
            DEVICE = cfg.device
            VRAM_LIMIT_GB = cfg.vram_limit_gb
            CACHE_DIR = os.path.dirname(cfg.model_path)
        except (ValueError, FileNotFoundError) as e:
            print(f"WARNING: Could not load config.json: {e}")

_load_config()


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------

def select_device() -> str:
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


def _model_slug(model_id: str) -> str:
    """Convert HuggingFace model ID to safe directory name."""
    return model_id.lower().replace("/", "-").replace(".", "-")


def _gpu_index(device: str) -> int:
    """Extract integer GPU index from device string like 'cuda:0'."""
    return int(device.split(":")[-1]) if ":" in device else 0


# ---------------------------------------------------------------------------
# Hardware & config output
# ---------------------------------------------------------------------------

def write_hardware_json(device: str) -> dict:
    """Write hardware.json so the agent knows GPU capabilities."""
    info: dict = {"device": device, "num_gpus": torch.cuda.device_count()}

    if device != "cpu":
        idx = _gpu_index(device)
        free, total = torch.cuda.mem_get_info(idx)
        cap = torch.cuda.get_device_capability(idx)
        major, minor = cap[0], cap[1]

        # BF16: Ampere (sm_8.0) and later
        bf16_supported = major >= 8
        # FP8: Hopper (sm_9.0+) natively fast; Ada (sm_8.9) also has FP8 via e4m3/e5m2
        fp8_supported = major > 9 or (major == 9) or (major == 8 and minor >= 9)

        info.update({
            "gpu_name": torch.cuda.get_device_name(idx),
            "gpu_index": idx,
            "vram_total_gb": round(total / 1024**3, 1),
            "vram_free_gb": round(free / 1024**3, 1),
            "compute_capability": f"{major}.{minor}",
            "cuda_version": torch.version.cuda,
            "bf16_supported": bf16_supported,
            "fp8_supported": fp8_supported,
        })

    path = os.path.join(PROJECT_DIR, "hardware.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote {path}")
    return info


def detect_model_metadata(model_path: str) -> dict:
    """Detect model architecture and size from model config files (no weights loaded).

    Returns a dict with:
      model_params_b    — estimated parameter count in billions
      model_type        — architecture family (e.g. "qwen2", "llama", "mistral")
      num_hidden_layers — transformer depth
      num_attention_heads — total attention heads
      num_key_value_heads — KV heads (< num_attention_heads means GQA)
      hidden_size       — model width
      intermediate_size — FFN intermediate width
      sliding_window    — sliding window size if used, else null
      mtp_supported     — true if model has multi-token prediction heads
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)

        hidden       = getattr(cfg, "hidden_size", 0)
        layers       = getattr(cfg, "num_hidden_layers", 0)
        vocab        = getattr(cfg, "vocab_size", 0)
        heads        = getattr(cfg, "num_attention_heads", 0)
        kv_heads     = getattr(cfg, "num_key_value_heads", heads)
        intermediate = getattr(cfg, "intermediate_size", 4 * hidden)

        # Heuristic param estimate (SwiGLU FFN: gate+up+down = 3 matrices)
        embedding_params = vocab * hidden
        attn_params = layers * (
            hidden * hidden                        # Q projection
            + (hidden // max(heads, 1)) * kv_heads * hidden * 2  # K+V projections
            + hidden * hidden                      # O projection
        )
        ffn_params = layers * (3 * hidden * intermediate)
        model_params_b = round((embedding_params + attn_params + ffn_params) / 1e9, 2)

        return {
            "model_params_b":       model_params_b,
            "model_type":           getattr(cfg, "model_type", "unknown"),
            "num_hidden_layers":    layers,
            "num_attention_heads":  heads,
            "num_key_value_heads":  kv_heads,
            "hidden_size":          hidden,
            "intermediate_size":    intermediate,
            "sliding_window":       getattr(cfg, "sliding_window", None),
            "mtp_supported":        getattr(cfg, "num_nextn_predict_layers", 0) > 0,
        }
    except Exception as e:
        print(f"  WARNING: Could not detect model metadata: {e}")
        return {
            "model_params_b": None, "model_type": "unknown",
            "num_hidden_layers": None, "num_attention_heads": None,
            "num_key_value_heads": None, "hidden_size": None,
            "intermediate_size": None, "sliding_window": None,
            "mtp_supported": False,
        }


def write_config_json(model_id: str, model_path: str, device: str, vram_limit_gb: float) -> dict:
    """Write config.json for infer.py to read instead of importing constants.

    Also detects and embeds model architecture metadata so the agent knows
    actual parameter count, GQA layout, and MTP support without guessing
    from the model name.
    """
    print("Detecting model architecture...")
    metadata = detect_model_metadata(model_path)
    params_b = metadata.get("model_params_b")
    if params_b is not None:
        print(f"  Model size: ~{params_b}B parameters")
    print(f"  Architecture: {metadata.get('model_type', 'unknown')}")
    if metadata.get("num_key_value_heads") and metadata.get("num_attention_heads"):
        if metadata["num_key_value_heads"] < metadata["num_attention_heads"]:
            print(f"  GQA: {metadata['num_key_value_heads']} KV heads / {metadata['num_attention_heads']} Q heads")
    if metadata.get("mtp_supported"):
        print("  MTP: supported (multi-token prediction heads detected)")
    if metadata.get("sliding_window"):
        print(f"  Sliding window: {metadata['sliding_window']} tokens")

    config = {
        "model_id":      model_id,
        "model_path":    model_path,
        "device":        device,
        "max_new_tokens": MAX_NEW_TOKENS,
        "vram_limit_gb": vram_limit_gb,
        # Model metadata — agent reads these for evidence-based decisions
        "model_params_b":      metadata["model_params_b"],
        "model_type":          metadata["model_type"],
        "num_hidden_layers":   metadata["num_hidden_layers"],
        "num_attention_heads": metadata["num_attention_heads"],
        "num_key_value_heads": metadata["num_key_value_heads"],
        "hidden_size":         metadata["hidden_size"],
        "intermediate_size":   metadata["intermediate_size"],
        "sliding_window":      metadata["sliding_window"],
        "mtp_supported":       metadata["mtp_supported"],
    }
    path = os.path.join(PROJECT_DIR, "config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {path}")
    return config


# ---------------------------------------------------------------------------
# Environment verification
# ---------------------------------------------------------------------------

def verify_cuda(device: Optional[str] = None) -> bool:
    """Verify CUDA is available and print GPU info for selected device."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This project requires an NVIDIA GPU.")
        print("If running --check on a non-GPU machine, this is expected.")
        return False

    idx = _gpu_index(device) if device else 0

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


def check_project_structure() -> bool:
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
# infer.py pre-flight validation
# ---------------------------------------------------------------------------

def validate_infer() -> Tuple[bool, str]:
    """Validate that infer.py is importable and exposes run_inference().

    Runs in a subprocess so import errors don't pollute the main process.
    Returns (is_valid, error_message).
    """
    check_script = (
        "import sys; sys.path.insert(0, '.'); "
        "from infer import run_inference; "
        "assert callable(run_inference), 'run_inference is not callable'; "
        "print('OK')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", check_script],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_DIR,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            return True, ""
        err = result.stderr.strip() or result.stdout.strip()
        return False, err
    except subprocess.TimeoutExpired:
        return False, "infer.py validation timed out after 30s (possible import hang)"
    except Exception as e:
        return False, f"Validation subprocess failed: {e}"


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def _find_or_download_model(model_id: str) -> str:
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

def load_prompts() -> list:
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

def validate_output(output_ids, input_length: int, tokenizer) -> Tuple[bool, str]:
    """Validate generated output is real text, not garbage.

    Checks:
    1. Output has enough new tokens (at least MIN_OUTPUT_RATIO of MAX_NEW_TOKENS)
    2. Output is not all padding or EOS tokens
    3. Output does not contain excessive repetition
    4. Output decodes to readable text

    Returns (is_valid, error_message).
    """
    new_tokens = output_ids[input_length:]
    num_generated = len(new_tokens)

    if num_generated < MAX_NEW_TOKENS * MIN_OUTPUT_RATIO:
        return False, (
            f"Too few tokens generated: {num_generated} "
            f"(minimum {int(MAX_NEW_TOKENS * MIN_OUTPUT_RATIO)} required, "
            f"got {num_generated / MAX_NEW_TOKENS * 100:.0f}% of max)"
        )

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
# Generation with timeout + OOM protection
# ---------------------------------------------------------------------------

class _GenerationResult:
    """Thread-safe container for generate_fn results."""
    def __init__(self):
        self.output = None
        self.error: Optional[Exception] = None
        self.ttft_ms: Optional[float] = None


def _call_generate_with_timeout(
    generate_fn: Callable,
    input_ids,
    timeout: float = GENERATION_TIMEOUT,
) -> Tuple[Optional[object], Optional[float], Optional[str]]:
    """Call generate_fn with a hard timeout and OOM protection.

    Returns (output_ids, ttft_ms, error_message).
    - output_ids: tensor if successful, None on failure
    - ttft_ms: first-token latency if returned by generate_fn, else None
    - error_message: None on success, string description on failure
    """
    result = _GenerationResult()

    def _target():
        try:
            ret = generate_fn(input_ids)
            # Support generate_fn returning (output_ids, metadata) or just output_ids
            if isinstance(ret, tuple) and len(ret) == 2:
                result.output, metadata = ret
                if isinstance(metadata, dict):
                    result.ttft_ms = metadata.get("ttft_ms")
            else:
                result.output = ret
        except torch.cuda.OutOfMemoryError as e:
            result.error = e
        except Exception as e:
            result.error = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Thread is still running — timed out
        return None, None, f"Generation timed out after {timeout}s"

    if result.error is not None:
        err = result.error
        if isinstance(err, torch.cuda.OutOfMemoryError):
            # Clean up GPU memory before returning
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None, f"CUDA out of memory: {err}"
        return None, None, f"Generation failed: {err}"

    return result.output, result.ttft_ms, None


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def run_profile(generate_fn: Callable, tokenizer) -> None:
    """Profile a single prompt and write profile.txt.

    Helps the agent understand where time is spent (attention vs linear vs
    layernorm) and make data-driven optimization decisions.

    Writes the FULL profiler table to profile.txt (no truncation).
    Prints top 15 ops to stdout for quick review.
    """
    from torch.profiler import profile, ProfilerActivity

    prompts = load_prompts()
    input_ids = tokenizer.encode(prompts[0]["text"], return_tensors="pt")

    # Warmup
    print("Warming up for profiling...")
    for _ in range(2):
        _call_generate_with_timeout(generate_fn, input_ids, timeout=GENERATION_TIMEOUT)
    if torch.cuda.is_available():
        torch.cuda.synchronize(DEVICE)

    print("Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        _call_generate_with_timeout(generate_fn, input_ids, timeout=GENERATION_TIMEOUT)
        if torch.cuda.is_available():
            torch.cuda.synchronize(DEVICE)

    # Write FULL profile to file (no truncation)
    path = os.path.join(PROJECT_DIR, "profile.txt")
    full_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=50)
    top15_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)

    with open(path, "w") as f:
        f.write("=== Profiler Output (sorted by CUDA time, top 50 ops) ===\n\n")
        f.write(full_table)
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
            f.write(f"\nPeak GPU memory: {peak:.2f} GB\n")

    print(f"Wrote {path} (full profile — {len(full_table.splitlines())} lines)")
    # Print top 15 to stdout for quick review
    print("\n--- Top 15 CUDA ops (stdout summary) ---")
    print(top15_table)


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def benchmark(generate_fn: Callable, tokenizer) -> dict:
    """Run the standardized inference benchmark.

    Args:
        generate_fn: Callable that takes input_ids (tensor) and returns either:
                     - output_ids (tensor), or
                     - (output_ids, metadata_dict) where metadata may include 'ttft_ms'
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        dict with keys: tok_s, ttft_ms, peak_vram_gb, total_prompts,
                        total_tokens, valid_outputs, invalid_outputs, oom_count
    """
    prompts = load_prompts()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(DEVICE)

    # --- Warmup ---
    print(f"Running {NUM_WARMUP_RUNS} warmup iterations...")
    warmup_prompt = prompts[0]
    warmup_ids = tokenizer.encode(warmup_prompt["text"], return_tensors="pt")
    for i in range(NUM_WARMUP_RUNS):
        out, _, err = _call_generate_with_timeout(generate_fn, warmup_ids)
        if err:
            print(f"  WARNING: Warmup {i+1} failed: {err}")
    if torch.cuda.is_available():
        torch.cuda.synchronize(DEVICE)
    print("Warmup complete.")
    print()

    # --- Benchmark ---
    total_gen_time = 0.0
    total_tokens_generated = 0
    ttft_values: list = []
    valid_count = 0
    invalid_count = 0
    oom_count = 0
    timeout_count = 0

    from collections import defaultdict
    cat_records = defaultdict(list)  # category -> list of (tok_s, is_valid)

    print(f"Benchmarking {NUM_PROMPTS} prompts (generating up to {MAX_NEW_TOKENS} tokens each)...")
    for i, prompt_data in enumerate(prompts):
        input_ids = tokenizer.encode(prompt_data["text"], return_tensors="pt")
        input_length = input_ids.shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize(DEVICE)

        t_start = time.perf_counter()
        output_ids, ttft_from_fn, gen_error = _call_generate_with_timeout(
            generate_fn, input_ids, timeout=GENERATION_TIMEOUT
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(DEVICE)
        t_end = time.perf_counter()

        cat = prompt_data.get("category", "unknown")

        if gen_error is not None:
            # Track failure type
            if "out of memory" in gen_error.lower():
                oom_count += 1
                status_str = "OOM"
            elif "timed out" in gen_error.lower():
                timeout_count += 1
                status_str = "TIMEOUT"
            else:
                status_str = "ERROR"
            invalid_count += 1
            cat_records[cat].append((0.0, False))
            print(f"  [{i+1:2d}/{NUM_PROMPTS}] {cat:8s} | ERROR: {gen_error[:60]}")
            continue

        gen_time = t_end - t_start
        total_gen_time += gen_time

        if output_ids.dim() > 1:
            output_ids = output_ids[0]

        num_new = len(output_ids) - input_length
        total_tokens_generated += num_new

        # Use TTFT from generate_fn if provided; otherwise use avg decode latency as proxy
        if ttft_from_fn is not None:
            ttft_values.append(ttft_from_fn)
        else:
            # Fallback: avg token latency (not true TTFT, but useful as a proxy)
            ttft_values.append(gen_time / max(num_new, 1) * 1000)

        is_valid, error = validate_output(output_ids, input_length, tokenizer)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            print(f"  WARNING: Prompt {prompt_data['id']} invalid output: {error}")

        tok_s_this = num_new / gen_time if gen_time > 0 else 0
        cat_records[cat].append((tok_s_this, is_valid))

        status = "OK" if is_valid else "INVALID"
        print(f"  [{i+1:2d}/{NUM_PROMPTS}] {cat:8s} | "
              f"{num_new:3d} tokens | {gen_time:.2f}s | "
              f"{tok_s_this:.1f} tok/s | {status}")

    # --- Compute metrics ---
    tok_s = total_tokens_generated / total_gen_time if total_gen_time > 0 else 0
    ttft_ms = sorted(ttft_values)[len(ttft_values) // 2] if ttft_values else 0.0

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
        "oom_count": oom_count,
        "timeout_count": timeout_count,
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
    if oom_count > 0:
        print(f"oom_count:        {results['oom_count']}")
    if timeout_count > 0:
        print(f"timeout_count:    {results['timeout_count']}")

    # --- Per-category breakdown ---
    print()
    print("--- category breakdown ---")
    for cat in sorted(cat_records.keys()):
        records = cat_records[cat]
        cat_tok_s_vals = [r[0] for r in records if r[0] > 0]
        cat_valid_count = sum(1 for r in records if r[1])
        avg_tok_s = sum(cat_tok_s_vals) / len(cat_tok_s_vals) if cat_tok_s_vals else 0
        print(f"  {cat:10s}: avg {avg_tok_s:6.1f} tok/s | "
              f"{cat_valid_count}/{len(records)} valid")

    # --- Safety checks ---
    if peak_vram_gb > VRAM_LIMIT_GB:
        print(f"\nWARNING: Peak VRAM ({peak_vram_gb:.1f} GB) exceeds limit ({VRAM_LIMIT_GB} GB)")

    if invalid_count > NUM_PROMPTS * 0.2:
        print(f"\nWARNING: {invalid_count}/{NUM_PROMPTS} outputs were invalid")

    if oom_count > 0:
        print(f"\nWARNING: {oom_count} prompt(s) hit OOM — experiment should be discarded")

    if timeout_count > 0:
        print(f"\nWARNING: {timeout_count} prompt(s) timed out after {GENERATION_TIMEOUT}s")

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
    parser.add_argument("--validate", action="store_true",
                        help="Validate infer.py interface without running benchmark")
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

    # --validate: just check infer.py is sane, no GPU needed
    if args.validate:
        print("Validating infer.py...")
        ok, err = validate_infer()
        if ok:
            print("  [OK] infer.py is valid — run_inference() is callable")
            sys.exit(0)
        else:
            print(f"  [FAIL] infer.py validation failed:\n{err}")
            sys.exit(1)

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
        idx = _gpu_index(device)
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

    # --- Pre-flight validation ---
    print("Validating infer.py before benchmark...")
    ok, err = validate_infer()
    if not ok:
        print(f"\nERROR: infer.py failed pre-flight validation:\n{err}")
        print("Fix infer.py before running the benchmark.")
        sys.exit(1)
    print("  [OK] infer.py is valid")
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
        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 60)
    print("Setup complete! Ready to experiment.")
    if not args.profile:
        print(f"Baseline tok/s: {results['tok_s']:.2f}")
    print("=" * 60)
