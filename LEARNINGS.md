# LEARNINGS.md — Cross-Run Knowledge Base

This file persists **across all autoresearch branches**.
The agent reads it at setup and updates it at the end of each session.
Do NOT reset or delete between runs — it is your long-term memory.

---

## Hardware Notes

- GPU: 2x NVIDIA RTX PRO 6000 Blackwell Workstation Edition (95 GB VRAM each, compute 12.0)
- BF16 supported, FP8 supported
- CUDA 13.0, PyTorch 2.11.0+cu130
- VRAM budget: 85.5 GB (90% of one GPU)
- BF16 is faster than FP16 on Blackwell for bitsandbytes 4-bit compute
- Memory bandwidth is the dominant bottleneck for 70B decode (93% GEMV)

---

## What Works (confirmed gains)

- INT4 NF4 (bitsandbytes) on 70B model: +116% throughput (10.50 → 22.66 tok/s) | Halves weight size, but naive kernel is bottleneck
- HF model.generate() instead of custom decode loop: much better output quality (20/20 vs 14/20 valid) with minimal speed loss
- Speculative decoding with Llama-3.2-1B draft: best composite score (22.66 tok/s, 20/20 valid) | Draft model only adds 2.3 GB VRAM
- device_map="auto" for multi-GPU: essential for models > 95GB in BF16

---

## What Doesn't Work (save time, skip these)

- torch.compile with device_map="auto" multi-GPU: 33% slower (7.03 vs 10.50)
- torch.compile with bitsandbytes quantized models: 77% slower (5.29 vs 22.66)
- INT8 bitsandbytes on 70B: slower than BF16 on 2 GPUs (7.18 vs 10.50) — dequant overhead
- INT4 double quantization: 32% slower than standard INT4 (15.49 vs 22.66)
- FP16 compute dtype: 42% slower than BF16 for 4-bit compute on Blackwell
- INT4 quantized draft model for spec decode: 15% slower than BF16 draft (19.37 vs 22.66)
- 3B draft model: slower than 1B (21.33 vs 22.66) — overhead exceeds acceptance benefit
- Prompt lookup decoding: 42% slower (13.19 vs 22.66) — output doesn't match input patterns
- auto-gptq / gptqmodel: QuantizeConfig broken with torch 2.11+cu130
- torchao cpp extensions: incompatible with torch 2.11+cu130
- flash-attn: fails to build with torch 2.11+cu130

---

## Near-Misses Worth Revisiting

- Speculative decoding with Llama-3.1-8B draft (same arch family): not tested due to download issues. Higher acceptance rate could give 10-20% boost.
- GPTQ pre-quantized model with Marlin kernels: could be 2-4x faster than bitsandbytes naive kernels. Need compatible gptq library.
- AWQ quantization with Marlin kernels: similar potential to GPTQ. Need autoawq calibration or pre-quantized model.
- vLLM or SGLang for optimized inference: would bypass all bitsandbytes issues.

---

## Best Config Found (so far)

```
DTYPE = torch.bfloat16
ATTENTION_IMPLEMENTATION = "sdpa"
USE_TORCH_COMPILE = False
QUANTIZATION_ENABLED = True
QUANTIZATION_TYPE = "int4"  # bitsandbytes NF4
USE_SPECULATIVE_DECODING = True
DRAFT_MODEL_PATH = "~/.cache/autoresearch-inference/meta-llama-llama-3-2-1b/model"
device_map = "auto"
Using model.generate() with assistant_model for speculative decoding
```
Result: tok/s: 22.66, ttft_ms: 41.19, VRAM: 18.7 GB, valid: 20/20 (commit: 687a6cb)

---

## Session Log

| Date | Run tag | Experiments | Best tok/s | Best config |
|------|---------|-------------|------------|-------------|
| 2026-03-27 | llama70b-blackwell | 16 | 22.66 | INT4 NF4 + spec decode (1B draft) |
