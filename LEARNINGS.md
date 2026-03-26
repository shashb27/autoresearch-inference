# LEARNINGS.md — Cross-Run Knowledge Base

This file persists **across all autoresearch branches**.  
The agent reads it at setup and updates it at the end of each session.  
Do NOT reset or delete between runs — it is your long-term memory.

---

## Hardware Notes

- **GPU:** NVIDIA GeForce RTX 5090 (compute 12.0) — BF16 ✅, FP8 ✅
- **VRAM:** 31.4 GB total, 28.2 GB limit (90%)
- **CUDA:** 13.0, PyTorch 2.11.0+cu130
- **Model:** Qwen2.5-7B (7.07B params, GQA 4/28 heads)
- **torch.compile:** Currently disabled — RTX 5090 compute 12.0 not fully supported yet
- **Baseline:** 85.36 tok/s @ 14.3GB VRAM (BF16 + SDPA, custom decode loop)

---

## What Works (confirmed gains)

> List techniques that reliably improved tok/s on this hardware.
> Format: `technique → approx gain | notes`

<!--
Example:
- flash_attention_2 → +18–22% | requires `uv add flash-attn`, slow first compile
- INT8 torchao → +35–45% | some quality loss on math prompts, acceptable
- torch.compile reduce-overhead → +5–8% | only helps after warmup
-->

_Not yet populated._

---

## What Doesn't Work (save time, skip these)

> Approaches that crashed, produced garbage, or gave no gain on this hardware.

<!--
Example:
- INT4 bitsandbytes → crashes on A100 (missing CUDA kernel for sm_80)
- fp8 quantization → no gain (GPU doesn't support native FP8)
- CUDA graphs → unstable with variable-length inputs, not worth debugging
- vllm → incompatible with current torch version
-->

_Not yet populated._

---

## Near-Misses Worth Revisiting

> Ideas that showed partial promise but weren't quite right.
> Include what to try differently next time.

<!--
Example:
- Speculative decoding with Qwen2.5-0.5B draft → 12% gain but 30% invalid outputs
  → Try: lower acceptance threshold, or use a better draft model
- compile max-autotune → 3h compile time, killed
  → Try: pre-cache compilation artifacts across runs
-->

_Not yet populated._

---

## Best Config Found (so far)

> Copy the winning `infer.py` configuration section here for easy reference.

<!--
Example:
DTYPE = torch.bfloat16
ATTENTION_IMPLEMENTATION = "flash_attention_2"
USE_TORCH_COMPILE = True
COMPILE_MODE = "reduce-overhead"
QUANTIZATION_ENABLED = True
QUANTIZATION_TYPE = "int8"
  → tok/s: 142.3, VRAM: 9.1 GB  (run: mar10, commit: a1b2c3d)
-->

_Not yet populated._

---

## Session Log

> One line per completed session. Agent appends to this.

| Date | Run tag | Experiments | Best tok/s | Best config |
|------|---------|-------------|------------|-------------|
| —    | —       | —           | —          | —           |
