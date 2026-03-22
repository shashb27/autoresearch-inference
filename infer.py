"""
Inference optimization script for autoresearch-inference.
This is the ONLY file the agent modifies.

The agent can change anything in this file: model loading strategy,
generation loop, attention configuration, KV cache management,
quantization, torch.compile settings, memory management, etc.

Constraint: Must expose run_inference() that returns (generate_fn, tokenizer)
compatible with prepare.benchmark().

Configuration is read from config.json (written by prepare.py).
Usage: uv run infer.py
"""

import os
import gc
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare import benchmark

# ============================================================
# SECTION 0: Runtime Config (from config.json)
# ============================================================

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def _load_config():
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}

_CFG = _load_config()

DEVICE = _CFG.get("device", "cuda")
MODEL_PATH = _CFG.get("model_path", os.path.join(
    os.path.expanduser("~"), ".cache", "autoresearch-inference", "model"))
MAX_NEW_TOKENS = _CFG.get("max_new_tokens", 256)

# ============================================================
# SECTION 1: Configuration & Hyperparameters
# ============================================================

# Model loading
DTYPE = torch.float16
ATTENTION_IMPLEMENTATION = "sdpa"   # "sdpa", "flash_attention_2", "eager"

# Compilation
USE_TORCH_COMPILE = True
COMPILE_MODE = "default"            # "default", "reduce-overhead", "max-autotune"
COMPILE_BACKEND = "inductor"

# Quantization
QUANTIZATION_ENABLED = False
QUANTIZATION_TYPE = None            # "int8", "int4", "fp8", "nf4"

# Generation
USE_STATIC_CACHE = False
KV_CACHE_DTYPE = None               # None (same as model), "fp8", "int8"

# Memory
PREALLOCATE_MEMORY = False
GC_COLLECT_BEFORE_BENCHMARK = True
EMPTY_CACHE_BEFORE_BENCHMARK = True

# TF32 matmul precision (faster on Ampere+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# SECTION 2: Model Loading
# ============================================================

def load_model():
    """Load and configure the model for inference."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=DTYPE,
        device_map=DEVICE,
        attn_implementation=ATTENTION_IMPLEMENTATION,
    )
    model.eval()
    return model


def load_tokenizer():
    """Load the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ============================================================
# SECTION 3: Model Optimization (post-load)
# ============================================================

def optimize_model(model):
    """Apply post-load optimizations to the model."""
    if USE_TORCH_COMPILE:
        model = torch.compile(
            model,
            mode=COMPILE_MODE,
            backend=COMPILE_BACKEND,
        )

    return model


# ============================================================
# SECTION 4: Generation Function
# ============================================================

def make_generate_fn(model, tokenizer):
    """Create the generation callable for benchmarking.

    Returns a function: generate_fn(input_ids) -> output_ids
    that generates MAX_NEW_TOKENS tokens using greedy decoding.
    """
    @torch.inference_mode()
    def generate_fn(input_ids):
        input_ids = input_ids.to(DEVICE)
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        return output

    return generate_fn


# ============================================================
# SECTION 5: Memory Management
# ============================================================

def setup_memory():
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

def run_inference():
    """Main entry point. Returns (generate_fn, tokenizer).

    Called by prepare.benchmark() and by this script's __main__.
    """
    setup_memory()

    print("Loading model...")
    t0 = time.time()
    model = load_model()
    t1 = time.time()
    print(f"Model loaded in {t1 - t0:.1f}s")

    print("Optimizing model...")
    model = optimize_model(model)
    t2 = time.time()
    print(f"Optimization applied in {t2 - t1:.1f}s")

    tokenizer = load_tokenizer()
    generate_fn = make_generate_fn(model, tokenizer)

    return generate_fn, tokenizer


if __name__ == "__main__":
    generate_fn, tokenizer = run_inference()
    results = benchmark(generate_fn, tokenizer)
