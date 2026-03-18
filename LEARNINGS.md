# LEARNINGS.md — Cross-Run Knowledge Base

This file persists **across all autoresearch branches**.
The agent reads it at setup and updates it at the end of each session.
Do NOT reset or delete between runs — it is your long-term memory.

---

## Hardware Notes

- GPU: NVIDIA GeForce RTX 3060 Ti (compute 8.6) — BF16 ✅, FP8 ❌
- VRAM: 7.7 GB total, 6.9 GB limit (90%)
- Model: Qwen/Qwen2.5-3B (small model, memory-bandwidth bound)
- GEMV operations dominate (76% CUDA time) - batch size 1 decode bottleneck
- FP16 faster than BF16 on this hardware for this model size

---

## What Works (confirmed gains)

**None yet.** The baseline configuration (FP16 + SDPA + torch.compile default) is extremely well-tuned for this model/GPU combination.

---

## What Doesn't Work (save time, skip these)

For Qwen2.5-3B on RTX 3060 Ti (8.6):

- **BF16 dtype** → -7% (37.50 vs 40.37 tok/s) | FP16 is faster for this model size
- **reduce-overhead compile** → -3% (38.99 vs 40.37) | no gain, adds overhead
- **Disable torch.compile** → -6% (37.72 vs 40.37) | compile provides benefit
- **flash-attn** → crash (build failure - torch not in build deps)
- **INT8 weight-only (torchao)** → -55% (18.28 vs 40.37) + all outputs invalid | major regression
- **max-autotune compile** → -5% (38.52 vs 40.37) | slower compile, no runtime gain
- **Naive custom decode loop** → O(n²) timeout | must use proper KV cache
- **TF32 matmul precision** → -6% (37.85 vs 40.37) | no benefit for FP16 model
- **eager attention** → -19% (32.78 vs 40.37) | SDPA is much faster

---

## Near-Misses Worth Revisiting

None identified. All experiments showed clear regressions.

---

## Best Config Found (so far)

```python
DTYPE = torch.float16
ATTENTION_IMPLEMENTATION = "sdpa"
USE_TORCH_COMPILE = True
COMPILE_MODE = "default"
QUANTIZATION_ENABLED = False
USE_STATIC_CACHE = False
```
→ **tok/s: 40.37**, VRAM: 5.8 GB, TTFT: 24.72ms
→ (baseline, commit: a9aef10)

---

## Session Log

| Date | Run tag | Experiments | Best tok/s | Best config |
|------|---------|-------------|------------|-------------|
| 2026-03-17 | qwen-qwen2-5-3b-r2 | 9 experiments, 0 improvements | 40.37 | baseline (FP16+SDPA+compile) |
