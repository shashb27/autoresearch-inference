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
import torch._dynamo
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow more cache entries for varying input shapes
torch._dynamo.config.cache_size_limit = 64

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
DTYPE = torch.bfloat16
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
            dynamic=True,
        )

    return model


# ============================================================
# SECTION 4: Generation Function
# ============================================================

def make_generate_fn(model, tokenizer):
    """Create the generation callable for benchmarking.

    Returns a function: generate_fn(input_ids) -> output_ids
    that generates MAX_NEW_TOKENS tokens using greedy decoding.
    Custom decode loop to reduce HF generate() framework overhead.
    """
    eos_token_id = tokenizer.eos_token_id

    @torch.inference_mode()
    def generate_fn(input_ids):
        input_ids = input_ids.to(DEVICE)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Preallocate output buffer to avoid repeated torch.cat
        output_ids = torch.empty(
            batch_size, seq_len + MAX_NEW_TOKENS,
            dtype=input_ids.dtype, device=input_ids.device
        )
        output_ids[:, :seq_len] = input_ids
        past_key_values = None
        num_generated = 0

        for i in range(MAX_NEW_TOKENS):
            if past_key_values is None:
                outputs = model(input_ids, use_cache=True)
            else:
                outputs = model(output_ids[:, seq_len + i - 1:seq_len + i], past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
            output_ids[:, seq_len + i] = next_token_id
            num_generated += 1
            if next_token_id.item() == eos_token_id:
                break

        return output_ids[:, :seq_len + num_generated]

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
