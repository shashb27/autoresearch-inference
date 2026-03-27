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

# Enable TF32 for float32 matmuls (uses tensor cores)
torch.set_float32_matmul_precision('high')

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

DEVICE: str         = _CFG.get("device", "cuda")
MODEL_PATH: str     = _CFG.get("model_path", os.path.join(
    os.path.expanduser("~"), ".cache", "autoresearch-inference", "model"))
MAX_NEW_TOKENS: int = _CFG.get("max_new_tokens", 256)

# ============================================================
# SECTION 1: Configuration & Hyperparameters
# ============================================================
# All flags default to the original baseline behaviour.
# The agent changes these — one or a few per experiment.

# --- Dtype & attention ---
DTYPE = torch.bfloat16
ATTENTION_IMPLEMENTATION: str = "sdpa"   # "sdpa" | "flash_attention_2" | "eager"

# --- Compilation ---
USE_TORCH_COMPILE: bool  = True
COMPILE_MODE: str        = "default"     # "default" | "reduce-overhead" | "max-autotune"
COMPILE_BACKEND: str     = "inductor"
# Separate prefill vs decode compile modes (only effective when USE_SPLIT_COMPILE=True)
USE_SPLIT_COMPILE: bool  = False
COMPILE_PREFILL_MODE: str = "default"           # compute-bound: try "max-autotune"
COMPILE_DECODE_MODE: str  = "reduce-overhead"   # bandwidth-bound: launch overhead matters

# --- Inductor knobs (applied before torch.compile) ---
# INDUCTOR_FX_GRAPH_CACHE caches compiled graphs to disk — saves compile time on reruns.
# The rest are off by default; turn on one at a time to measure their effect.
INDUCTOR_FX_GRAPH_CACHE: bool          = True   # free win: no runtime cost
INDUCTOR_EPILOGUE_FUSION: bool         = True   # fuse pointwise ops into matmul epilogues
INDUCTOR_COORDINATE_DESCENT: bool      = True   # autotune Triton tile sizes (slow first run)
INDUCTOR_SHAPE_PADDING: bool           = True   # pad shapes for memory alignment

# --- Quantization ---
QUANTIZATION_ENABLED: bool      = False
QUANTIZATION_TYPE: Optional[str] = None  # "int8" | "int4" | "fp8" | "nf4" | "awq" | "gptq"

# --- Generation loop ---
RETURN_DICT: bool      = False  # False = return tuple, avoids dict construction overhead
SKIP_EARLY_STOP: bool  = True   # set min_new_tokens=max_new_tokens, skips EOS check each step

# --- KV cache ---
USE_STATIC_CACHE: bool           = False  # fixed-size cache, required for CUDA graphs
KV_CACHE_DTYPE: Optional[str]    = None   # None (same as model) | "fp8" | "int8"
USE_QUANTIZED_CACHE: bool        = False  # quantize KV cache (transformers.QuantizedCache)
KV_CACHE_QUANT_BACKEND: str      = "quanto"  # "quanto" | "HQQ"

# --- CUDA graphs ---
# Requires USE_STATIC_CACHE=True and fixed batch/sequence shapes.
# Captures the decode step as a CUDA graph for near-zero kernel-launch overhead.
# NOTE: incompatible with the per-prompt generation timeout in prepare.py.
# When enabled, set GENERATION_TIMEOUT high enough to cover full decode.
USE_CUDA_GRAPHS: bool = False

# --- Speculative decoding ---
# A small draft model proposes SPECULATIVE_K tokens; the main model verifies
# all K in one prefill-like forward pass. Typical gain: 2–3× on decode-bound workloads.
# DRAFT_MODEL_PATH must be a same-family smaller model (e.g. 0.5B for a 7B main model).
USE_SPECULATIVE_DECODING: bool = False
DRAFT_MODEL_PATH: str          = ""   # fill with path from config.json model family
SPECULATIVE_K: int             = 4    # tokens to draft per step (tune 3–8)

# --- Batch size ---
# Currently the benchmark harness sends prompts one-at-a-time (batch=1).
# Changing BATCH_SIZE here affects only the generate_fn; the harness still
# iterates over individual prompts. Useful for throughput experiments.
BATCH_SIZE: int = 1

# --- Memory ---
PREALLOCATE_MEMORY: bool         = False
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

def _apply_inductor_configs() -> None:
    """Apply torch._inductor config knobs before compilation.

    These affect how Inductor generates Triton kernels. Applied once at startup.
    INDUCTOR_FX_GRAPH_CACHE is on by default — it caches compiled graphs to disk
    so reruns skip recompilation entirely.
    """
    import torch._inductor.config as ind
    ind.fx_graph_cache            = INDUCTOR_FX_GRAPH_CACHE
    ind.epilogue_fusion           = INDUCTOR_EPILOGUE_FUSION
    ind.coordinate_descent_tuning = INDUCTOR_COORDINATE_DESCENT
    ind.shape_padding             = INDUCTOR_SHAPE_PADDING


def optimize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Apply post-load optimizations to the model."""
    if USE_TORCH_COMPILE:
        _apply_inductor_configs()

        if USE_SPLIT_COMPILE:
            # Separate compile modes for prefill (compute-bound) and decode
            # (bandwidth-bound). The agent uses two model references and dispatches
            # on step==0 inside generate_fn.
            # This is a template — implement dispatch logic in make_generate_fn.
            model = torch.compile(
                model,
                mode=COMPILE_DECODE_MODE,
                backend=COMPILE_BACKEND,
                dynamic=True,
            )
        else:
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

    Returns:
        generate_fn(input_ids) -> (output_ids, {"ttft_ms": float})

    The function uses a custom decode loop to eliminate HF generate() overhead.
    TTFT is measured at the end of the first forward pass (true prefill latency).

    CUDA graphs (USE_CUDA_GRAPHS=True):
      - Requires USE_STATIC_CACHE=True
      - Warmup runs capture the decode graph; subsequent calls replay it
      - Near-zero kernel-launch overhead per decode step
      - Template below shows the capture pattern; agent fills in StaticCache setup

    Speculative decoding (USE_SPECULATIVE_DECODING=True):
      - Draft model proposes SPECULATIVE_K tokens per step
      - Main model verifies all K in one forward pass
      - Accept/reject via probability ratio; fall back on rejection
      - Template below shows the verification loop structure
    """
    eos_token_id = tokenizer.eos_token_id
    min_new = MAX_NEW_TOKENS if SKIP_EARLY_STOP else 1

    # ----------------------------------------------------------------
    # CUDA GRAPH DECODE TEMPLATE
    # Uncomment and complete when USE_CUDA_GRAPHS=True.
    # Requires USE_STATIC_CACHE=True and fixed MAX_NEW_TOKENS.
    # ----------------------------------------------------------------
    # if USE_CUDA_GRAPHS:
    #     from transformers import StaticCache
    #     # Set up a static cache sized for the longest expected sequence
    #     MAX_SEQ_LEN = 1024  # adjust to prompts + MAX_NEW_TOKENS
    #     static_cache = StaticCache(
    #         config=model.config, max_batch_size=BATCH_SIZE,
    #         max_cache_len=MAX_SEQ_LEN, device=DEVICE, dtype=DTYPE,
    #     )
    #     # Warmup to compile and capture the decode graph
    #     # (do this during optimize_model or first call to generate_fn)
    #     ...

    # ----------------------------------------------------------------
    # SPECULATIVE DECODING TEMPLATE
    # Uncomment and complete when USE_SPECULATIVE_DECODING=True.
    # DRAFT_MODEL_PATH must be set to a smaller same-family model.
    # ----------------------------------------------------------------
    # if USE_SPECULATIVE_DECODING:
    #     draft_model = AutoModelForCausalLM.from_pretrained(
    #         DRAFT_MODEL_PATH, dtype=DTYPE, device_map=DEVICE,
    #     ).eval()
    #     if USE_TORCH_COMPILE:
    #         draft_model = torch.compile(draft_model, mode="reduce-overhead", dynamic=True)
    #
    # @torch.inference_mode()
    # def generate_fn_speculative(input_ids):
    #     # 1. Draft: generate SPECULATIVE_K tokens autoregressively with draft model
    #     # 2. Verify: run main model on draft tokens in one forward pass
    #     # 3. Accept: keep tokens up to first rejection; append one corrected token
    #     # 4. Repeat until MAX_NEW_TOKENS reached
    #     ...

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
                outputs = model(
                    generated,
                    use_cache=True,
                    return_dict=RETURN_DICT,
                )
            else:
                outputs = model(
                    generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=RETURN_DICT,
                )

            # Support both dict output (return_dict=True) and tuple (return_dict=False)
            if RETURN_DICT:
                logits          = outputs.logits
                past_key_values = outputs.past_key_values
            else:
                logits          = outputs[0]
                past_key_values = outputs[1]

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Measure TTFT: time until first token is produced (end of prefill)
            if step == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft_ms = (time.perf_counter() - t_start) * 1000

            generated = torch.cat([generated, next_token], dim=-1)

            # Skip EOS check when SKIP_EARLY_STOP=True (saves a Python conditional per step)
            if step >= min_new - 1 and next_token.item() == eos_token_id:
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
