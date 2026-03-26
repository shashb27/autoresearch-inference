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

- **FP16 dtype** → +0.1% throughput, -11% TTFT (18.65ms vs 20.95ms) | Marginal gain, but TTFT improvement is real

---

## What Doesn't Work (save time, skip these)

**RTX 5090 + PyTorch 2.11.0+cu130 + Qwen2.5-7B environment constraints:**

- **torch.compile**: Compilation subprocess crashes (compute 12.0 not fully supported yet)
- **flash_attention_2**: Build failure (CUDA version mismatch: 12.9 vs 13.0)
- **INT8 weight-only (torchao)**: -57% throughput degradation (cpp extensions incompatible with torch 2.11.0+cu130, falls back to slow Python path)
- **FP8 KV cache**: `cache_dtype` not supported by `from_pretrained` API
- **Preallocated output buffer**: No gain over torch.cat (within noise, 85.23 vs 85.36 tok/s)

**Root cause:** RTX 5090 (compute 12.0) is bleeding-edge hardware; PyTorch/torchao/flash-attn ecosystem not yet fully compatible.

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

```python
DTYPE = torch.float16
ATTENTION_IMPLEMENTATION = "sdpa"
USE_TORCH_COMPILE = False  # RTX 5090 not supported
RETURN_DICT = False
SKIP_EARLY_STOP = True
```

**Result:** 85.45 tok/s, 18.65ms TTFT, 14.3GB VRAM
**Commit:** 83350b0 (branch: autoresearch/rtx5090-qwen7b)

**Improvement:** +0.1% tok/s, -11% TTFT vs baseline (85.36 tok/s, 20.95ms)

---

## Session Log

> One line per completed session. Agent appends to this.

| Date | Run tag | Experiments | Best tok/s | Best config |
|------|---------|-------------|------------|-------------|
| 2026-03-26 | rtx5090-qwen7b | 8 (torch.compile, flash-attn, INT8, FP8 KV, preallocate, FP16, TF32, EOS) | 85.45 | FP16 + SDPA |
