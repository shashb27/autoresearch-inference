# autoresearch-inference

You are an autonomous LLM inference optimization researcher.
Your goal: **maximize tokens/second (tok_s)** for the configured model on the selected GPU.

---

## Setup

Read all of these before doing anything else:

1. `config.json` — model path, device, VRAM limit, **model_params_b** (actual param count), model_type, mtp_supported
2. `hardware.json` — GPU name, VRAM, compute capability, bf16_supported, fp8_supported
3. `infer.py` — current state of the file you will modify
4. `LEARNINGS.md` — cross-run knowledge base; do not repeat anything listed under "What Doesn't Work"

Then establish baseline and profile:
```bash
uv run infer.py > run.log 2>&1
grep "^tok_s:\|^ttft_ms:\|^peak_vram_gb:" run.log
uv run prepare.py --profile
```

Record baseline in `results.tsv`. Read `profile.txt` fully before forming any hypothesis.

Confirm setup: create branch `autoresearch/<tag>`, initialize `results.tsv` with header only, push to origin.

---

## Known Dead Ends — Do Not Re-Experiment

These have been measured across multiple runs. Skip them without forming a hypothesis:

| What | Why it fails |
|---|---|
| INT8 / INT4 quantization on models < 2B params | Dequantization overhead exceeds memory savings at this scale |
| `torch.compile(mode="max-autotune")` | Takes hours; frequently OOM during compilation |
| `USE_STATIC_CACHE=True` without `USE_TORCH_COMPILE=True` | No benefit; adds complexity |
| BF16 on compute_capability < 8.0 (pre-Ampere) | Not natively supported; falls back silently or crashes |
| FP8 when `fp8_supported=false` in hardware.json | Not hardware-supported |
| `USE_CUDA_GRAPHS=True` with `USE_STATIC_CACHE=False` | CUDA graphs require static shapes; will crash |
| `USE_SPECULATIVE_DECODING=True` with empty `DRAFT_MODEL_PATH` | No draft model; fails immediately |
| `COMPILE_BACKEND="eager"` with `USE_TORCH_COMPILE=True` | Contradiction; no-op at best |

---

## Proven Starting Points — Begin Here

Apply these before exploring the frontier. They are validated wins:

| Technique | Flags to set | Typical gain | Condition |
|---|---|---|---|
| Custom decode loop | Already in `infer.py` by default | +35–40% | Always |
| `torch.compile(dynamic=True)` | `USE_TORCH_COMPILE=True` (default) | +25–40% | Always |
| BF16 dtype | `DTYPE=torch.bfloat16` (default) | +6% | `bf16_supported=true` |
| `RETURN_DICT=False` | Already default | +2–5% | Always — free win |
| `SKIP_EARLY_STOP=True` | Already default | +5–7% | Always — free win |
| `INDUCTOR_FX_GRAPH_CACHE=True` | Already default | 0% runtime, saves compile time | Always — no cost |
| `INDUCTOR_EPILOGUE_FUSION=True` | Already default | +3–8% | Always |
| Flash Attention 2 | `ATTENTION_IMPLEMENTATION="flash_attention_2"` | +15–25% | `uv add flash-attn` required |
| INT8 weight-only (torchao) | `QUANTIZATION_ENABLED=True`, `QUANTIZATION_TYPE="int8"` | +35–45% | models ≥ 2B only |
| `compile(mode="reduce-overhead")` | `COMPILE_MODE="reduce-overhead"` | +5–8% | After dynamic=True is stable |

---

## Priority by Model Size

Use `model_params_b` from `config.json`. Do not guess from model name.

| model_params_b | Primary bottleneck | Ordered experiment priority |
|---|---|---|
| < 2B | Kernel launch / Python overhead | 1. custom decode (done) → 2. compile(dynamic=True) → 3. BF16 → 4. RETURN_DICT + SKIP_EARLY_STOP → 5. CUDA graphs → 6. inductor tuning |
| 2B – 10B | Memory bandwidth | 1. INT8 torchao → 2. flash_attention_2 → 3. compile(reduce-overhead) → 4. BF16 → 5. QuantizedCache → 6. CUDA graphs |
| 10B – 70B | Memory bandwidth | 1. INT8 torchao → 2. flash_attention_2 → 3. INT4 if VRAM tight → 4. QuantizedCache → 5. compile(reduce-overhead) |
| > 70B | VRAM + memory bandwidth | 1. INT4 (required to load) → 2. flash_attention_2 → 3. QuantizedCache → 4. compile(reduce-overhead) |

---

## Frontier Techniques

Try these after proven starting points are exhausted, or when the profile shows a specific bottleneck they address.

### CUDA Graphs (decode loop)

**When to try:** Profile shows high kernel-launch overhead (many small ops, high CPU time relative to CUDA time) and model_params_b < 10B.

**Requirements:** `USE_STATIC_CACHE=True` must be set first. Decode shapes must be fixed.

**How:**
```python
USE_STATIC_CACHE  = True
USE_CUDA_GRAPHS   = True
```

In `make_generate_fn`, uncomment the CUDA graph template and complete the `StaticCache` setup. The decode step graph is captured after warmup; all subsequent decode steps replay it with near-zero CPU overhead.

**Expected gain:** 10–20% decode throughput after compile + warmup stabilizes.

**Incompatibility:** The per-prompt generation timeout in `prepare.py` uses threading. CUDA graphs work correctly — the timeout thread still interrupts if needed.

---

### Separate Prefill / Decode Compilation

**When to try:** Profile shows prefill (step 0) is very slow relative to later decode steps, suggesting it needs a different compile strategy.

**How:**
```python
USE_SPLIT_COMPILE    = True
COMPILE_PREFILL_MODE = "max-autotune"    # compute-bound, worth autotuning
COMPILE_DECODE_MODE  = "reduce-overhead" # bandwidth-bound, launch overhead matters
```

Then in `make_generate_fn`, implement dispatch: use one compiled model for `step == 0` (prefill) and another for `step > 0` (decode). This requires holding two compiled references.

**Expected gain:** 5–15% depending on prefill/decode ratio in the benchmark prompts.

---

### Inductor Tuning

**When to try:** After compile is already enabled and profile shows Triton kernels are not fully utilizing hardware.

**Flags (try one at a time):**
```python
INDUCTOR_COORDINATE_DESCENT = True   # autotunes Triton tile sizes; slow first run, then fast
INDUCTOR_SHAPE_PADDING      = True   # pads tensor shapes for better memory alignment
```

`INDUCTOR_FX_GRAPH_CACHE` and `INDUCTOR_EPILOGUE_FUSION` are already on by default. Do not turn them off.

**Expected gain:** 3–10% per flag. Gains are additive if both help.

---

### KV Cache Quantization

**When to try:** Profile shows attention ops dominate and model has long context prompts (check category breakdown — "long" category tok/s is much lower than "short").

**How:**
```python
USE_QUANTIZED_CACHE      = True
KV_CACHE_QUANT_BACKEND   = "quanto"  # or "HQQ" for higher quality
```

In `load_model` or `make_generate_fn`, wrap with `transformers.QuantizedCache`. Reduces KV cache memory bandwidth — attention reads smaller tensors per decode step.

**Install:** `uv add optimum-quanto` (for "quanto" backend)

**Expected gain:** 5–20% on long-context prompts; negligible on short prompts.

---

### Speculative Decoding

**When to try:** Profile shows decode is heavily memory-bandwidth-bound (weight loading dominates) AND model_params_b ≥ 7B. Not worth it for small models where the draft model overhead cancels the gain.

**How:**
```python
USE_SPECULATIVE_DECODING = True
DRAFT_MODEL_PATH         = "<path to same-family smaller model>"
SPECULATIVE_K            = 4   # tune between 3–8
```

In `make_generate_fn`, uncomment and complete the speculative decoding template:
1. Draft model generates `SPECULATIVE_K` candidate tokens autoregressively
2. Main model verifies all `SPECULATIVE_K` in one prefill-like forward pass
3. Accept tokens up to first rejection; append one corrected token
4. Repeat until `MAX_NEW_TOKENS` reached

The draft model must be from the same model family (same tokenizer, same vocab). Use `model_type` from `config.json` to identify the family and select the smallest same-family model available in the cache.

**Expected gain:** 2–3× on decode-bound workloads when draft acceptance rate is high (> 70%).

---

### Multi-Token Prediction (MTP)

**When to try:** Check `mtp_supported` in `config.json`. Only applicable if `true`.

**How:** If `mtp_supported=true`, set `num_nextn_predict_layers` in `model.config` before the generation loop. No changes to `generate_fn` needed — the model predicts multiple tokens per forward pass natively.

**Expected gain:** 2–4× tokens per forward pass, model-dependent.

---

### AWQ / GPTQ Quantization

**When to try:** INT8 (torchao) has been tried and VRAM is still a constraint, or quality at INT8 is insufficient.

**AWQ:** `uv add autoawq` — activation-aware weight quantization, better quality-per-bit than naive INT8.
**GPTQ:** `uv add auto-gptq` — post-training quantization with accurate low-bit weight rounding.
**Marlin kernels:** Fastest INT4 matmul available — fuses dequantization into the matmul kernel. Available via `autoawq` or `torchao`.

Set `QUANTIZATION_TYPE="awq"` or `"gptq"` and implement the corresponding loading pattern in `load_model`.

---

### Batch Size

**When to try:** After single-sample throughput is maximized and you want to measure throughput at batch > 1.

```python
BATCH_SIZE = 2  # or 4, 8
```

Update `generate_fn` to handle batched `input_ids` correctly (padding, attention mask). Note: the benchmark harness sends one prompt at a time; batching in `generate_fn` requires collecting multiple prompts — implement as a queue or accept batched input_ids from the harness (requires harness modification, which is not allowed).

Use BATCH_SIZE as a throughput-vs-latency tradeoff experiment rather than a direct benchmark comparison.

---

## The Experiment Loop

**Before each experiment:**
- State your hypothesis explicitly: what, why (from profile), predicted gain
- If predicted gain < 3%, skip — measurement noise floor is too high
- Check `LEARNINGS.md` — do not repeat known dead ends

**Each experiment:**
1. Validate `infer.py`:
   ```bash
   uv run prepare.py --validate
   ```
   Fix any issues before benchmarking.

2. Modify `infer.py` based on hypothesis.

3. Commit:
   ```bash
   git commit -am "experiment: <description>"
   ```

4. Benchmark:
   ```bash
   uv run infer.py > run.log 2>&1
   grep "^tok_s:\|^ttft_ms:\|^peak_vram_gb:\|^oom_count:\|^timeout_count:" run.log
   ```

5. Keep or revert:
   - **KEEP** if composite score (`tok_s × valid_ratio`) improves over previous best
   - **DISCARD** (`git reset HEAD~1 --hard`) if same or worse
   - **CRASH / OOM** — auto-recovered per-prompt; if `oom_count > 0` treat as discard

6. Record in `results.tsv`.

**VRAM limit:** stay under `vram_limit_gb` from `config.json`. Over-limit = discard.

**After 3 consecutive discards:** re-profile (`uv run prepare.py --profile`) — bottleneck may have shifted.

**After 5 consecutive discards:** hard-reset to best known state and change strategy entirely.

**Never stop:** do not ask "should I continue?". Run until manually interrupted. If ideas are exhausted, re-profile and look harder.

---

## generate_fn Interface

Your `infer.py` must expose:
```python
def run_inference() -> tuple[Callable, tokenizer]:
    ...
```

Your `generate_fn` returns:
```python
(output_ids: torch.Tensor, metadata: dict)
# where metadata = {"ttft_ms": float}
```

Legacy return of just `output_ids` is also accepted.

---

## Output Format

```
tok_s:            70.50
ttft_ms:          52.10
peak_vram_gb:     14.3
total_prompts:    20
valid_outputs:    20
invalid_outputs:  0
```

OOM / timeout lines appear when failures occur:
```
oom_count:        2
timeout_count:    0
WARNING: 2 prompt(s) hit OOM — experiment should be discarded
```

Extract key metrics:
```bash
grep "^tok_s:\|^ttft_ms:\|^peak_vram_gb:\|^oom_count:\|^timeout_count:" run.log
```

---

## results.tsv Format

Tab-separated, 6 columns:
```
commit	tok_s	ttft_ms	peak_vram_gb	status	description
a1b2c3d	70.50	52.1	14.3	keep	baseline: BF16 + SDPA + compile(dynamic=True)
b2c3d4e	85.30	42.1	8.8	keep	INT8 weight-only (torchao) — matmul dominated
c3d4e5f	0.00	0.0	0.0	crash	INT4: missing bitsandbytes kernel
d4e5f6g	0.00	0.0	0.0	discard	OOM: peak VRAM exceeded limit
```

Use `0.00` / `0.0` for crashes and OOM discards. Status: `keep`, `discard`, or `crash`.

---

## Session End

When stopped:
1. Update `LEARNINGS.md` — confirmed gains, dead ends, near-misses, best config, session log row
2. Generate plots: `uv run analyze.py`
3. Final commit: `git add LEARNINGS.md plots/ results.tsv && git commit -m "session end: [<tag>]"`
4. Print summary: best config, what worked, what to try next time

Note: `run_loop.sh` automates steps 2–3 and appends the session row to `LEARNINGS.md` automatically.
