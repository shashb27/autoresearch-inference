"""
Inference optimization script for autoresearch-inference.
This is the ONLY file the agent modifies.

The agent can change anything in this file: model loading strategy,
generation loop, attention configuration, KV cache management,
quantization, torch.compile settings, memory management, etc.

Constraint: Must expose run_inference() -> (generate_fn, tokenizer)
  where generate_fn(input_ids: torch.Tensor) returns either:
    - output_ids: torch.Tensor                    (legacy, still accepted)
    - (output_ids, metadata): (torch.Tensor, dict) where metadata MAY include:
        "ttft_ms": float  — true time-to-first-token in milliseconds

Configuration is read from config.json (written by prepare.py).
Usage: uv run infer.py
"""

import os
import gc
import json
import time
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch._dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow more cache entries for varying input shapes
torch._dynamo.config.cache_size_limit = 64

from prepare import benchmark

# ============================================================
# SECTION 0: Runtime Config (from config.json)
# ============================================================

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def _load_config() -> dict:
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}

_CFG = _load_config()

DEVICE: str = _CFG.get("device", "cuda")
MODEL_PATH: str = _CFG.get("model_path", os.path.join(
    os.path.expanduser("~"), ".cache", "autoresearch-inference", "model"))
MAX_NEW_TOKENS: int = _CFG.get("max_new_tokens", 256)

# ============================================================
# SECTION 1: Configuration & Hyperparameters
# ============================================================

# Model loading
DTYPE = torch.bfloat16
ATTENTION_IMPLEMENTATION: str = "sdpa"   # "sdpa", "flash_attention_2", "eager"

# Compilation
USE_TORCH_COMPILE: bool = True
COMPILE_MODE: str = "default"            # "default", "reduce-overhead", "max-autotune"
COMPILE_BACKEND: str = "inductor"

# Quantization
QUANTIZATION_ENABLED: bool = False
QUANTIZATION_TYPE: Optional[str] = None  # "int8", "int4", "fp8", "nf4"

# Generation
USE_STATIC_CACHE: bool = False
KV_CACHE_DTYPE: Optional[str] = None    # None (same as model), "fp8", "int8"

# Memory
PREALLOCATE_MEMORY: bool = False
GC_COLLECT_BEFORE_BENCHMARK: bool = True
EMPTY_CACHE_BEFORE_BENCHMARK: bool = True


# ============================================================
# SECTION 2: Model Loading
# ============================================================

def load_model() -> torch.nn.Module:
    """Load and configure the model for inference."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    model.eval()
    return model


def load_tokenizer() -> AutoTokenizer:
    """Load the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ============================================================
# SECTION 3: Model Optimization (post-load)
# ============================================================

def optimize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Apply post-load optimizations to the model."""
    if USE_TORCH_COMPILE:
        model = torch.compile(
            model,
            mode=COMPILE_MODE,
            backend=COMPILE_BACKEND,
            dynamic=True,
        )

    return model


# ============================================================
# SECTION 4: Generation Function
# ============================================================

def make_generate_fn(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict]]:
    """Create the generation callable for benchmarking.

    Returns a function:
        generate_fn(input_ids: torch.Tensor) -> (output_ids: torch.Tensor, metadata: dict)

    where metadata contains:
        "ttft_ms": float — true time-to-first-token (prefill latency) in milliseconds

    Generates MAX_NEW_TOKENS tokens using greedy decoding with a custom
    decode loop to reduce HF generate() framework overhead.
    """
    eos_token_id = tokenizer.eos_token_id

    @torch.inference_mode()
    def generate_fn(
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        input_ids = input_ids.to(DEVICE)
        generated = input_ids
        past_key_values = None
        ttft_ms: Optional[float] = None

        t_start = time.perf_counter()

        for step in range(MAX_NEW_TOKENS):
            if past_key_values is None:
                outputs = model(generated, use_cache=True)
            else:
                outputs = model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Measure TTFT: time until first token is produced (end of prefill)
            if step == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft_ms = (time.perf_counter() - t_start) * 1000

            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == eos_token_id:
                break

        return generated, {"ttft_ms": ttft_ms}

    return generate_fn


# ============================================================
# SECTION 5: Memory Management
# ============================================================

def setup_memory() -> None:
    """Configure GPU memory allocation before benchmarking."""
    if GC_COLLECT_BEFORE_BENCHMARK:
        gc.collect()
    if EMPTY_CACHE_BEFORE_BENCHMARK and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# SECTION 6: Entry Point
# ============================================================

def run_inference() -> Tuple[Callable, AutoTokenizer]:
    """Main entry point. Returns (generate_fn, tokenizer).

    Called by prepare.benchmark() and by this script's __main__.

    generate_fn signature:
        (input_ids: torch.Tensor) -> (output_ids: torch.Tensor, metadata: dict)
    """
    setup_memory()

    print("Loading model...")
    t0 = time.time()
    model = load_model()
    t1 = time.time()
    print(f"  Model loaded in {t1 - t0:.1f}s")

    print("Optimizing model...")
    model = optimize_model(model)
    t2 = time.time()
    print(f"  Optimization applied in {t2 - t1:.1f}s")

    tokenizer = load_tokenizer()
    generate_fn = make_generate_fn(model, tokenizer)

    return generate_fn, tokenizer


if __name__ == "__main__":
    generate_fn, tokenizer = run_inference()
    results = benchmark(generate_fn, tokenizer)
