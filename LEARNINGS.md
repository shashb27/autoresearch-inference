# LEARNINGS.md — Cross-Run Knowledge Base

This file persists **across all autoresearch branches**.
The agent reads it at setup and updates it at the end of each session.
Do NOT reset or delete between runs — it is your long-term memory.

---

## Hardware Notes

- GPU: NVIDIA GeForce RTX 3060 Ti (compute 8.6), 7.7 GB VRAM
- Model: Qwen/Qwen2.5-0.5B (~0.9 GB VRAM in FP16)
- VRAM budget: ~6.9 GB (90% of total)
- **FP16 is faster than BF16 on this specific setup** — counterintuitive, but confirmed across multiple runs
- TF32 tensor cores available on Ampere (sm_86) — enable via matmul precision + backends
- FP8 not supported (fp8_supported: false in hardware.json)

---

## What Works (confirmed gains)

- `torch.set_float32_matmul_precision('high')` → +0.26% | enables TF32 for FP32 matmuls
- `torch.backends.cudnn.benchmark = True` → +2.7% | auto-tunes cuDNN kernels
- Explicit TF32 enable (`torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`) → +0.9%
- `min_new_tokens=max_new_tokens` in generate() → +1.2% | skips early stopping checks

**Cumulative gain from R2 session: +5.2% (58.61 → 61.63 tok/s)**

---

## What Doesn't Work (save time, skip these)

### Never attempt on this hardware:
- **BF16 dtype** → consistently 1-2% slower than FP16 (unexpected, but confirmed 3x)
- **INT8 weight-only quantization (torchao)** → -44% (34.29 tok/s) | overhead dominates on tiny 0.5B model
- **INT4 quantization (torchao)** → crashes (missing mslk>=1.0.0 dependency, not installable)
- torch.compile `reduce-overhead` mode → no gain vs default
- torch.compile `max-autotune` mode → no gain vs default
- torch.compile `fullgraph=True` → no gain
- torch.compile `dynamic=False` (static shapes) → regression
- Static KV cache (from R1) → crashes with Triton compile error
- Custom decode loop (from R1) → crashes with Triton compile error

### Other failed optimizations:
- `use_cache=True + return_dict_in_generate=False` → regression
- `channels_last` memory format → no gain
- NVFuser enabled → regression
- `pad_to_multiple_of=8` in tokenizer → no gain
- `torch.set_grad_enabled(False)` globally → regression (-2.9%)
- `eos_token_id=None` → regression
- Explicitly disable `output_scores/attentions/hidden_states` → regression
- `num_return_sequences=1` explicitly → slight regression
- Remove `pad_token_id` from generate() → slight regression
- CUDA graphs via `torch._inductor.config.triton.cudagraphs = True` → slight regression (-0.5%)

---

## Near-Misses Worth Revisiting

- **Flash Attention 2** → NOT yet tried. Requires `uv add flash-attn` (long build ~10 min).
  → Worth trying on longer sequences or with a larger model (7B+)
- **Quantization at larger scale** → INT8/INT4 failed on 0.5B, but may help on 7B+ models where memory bandwidth is the true bottleneck
- **Speculative decoding** → NOT tried, complex to implement, needs draft model
- **Custom decode loop without compile** → Tried in R1, was slower. May be worth revisiting with better kernel implementation
- **vLLM or SGLang backends** → NOT tried, may offer better inference kernels

---

## Best Config Found (so far)

```python
# Model loading
DTYPE = torch.float16  # NOT bfloat16! (FP16 is faster on RTX 3060 Ti)
ATTENTION_IMPLEMENTATION = "sdpa"
USE_TORCH_COMPILE = True
COMPILE_MODE = "default"  # NOT reduce-overhead or max-autotune

# Memory management & kernel optimization
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Generation
max_new_tokens = 256
min_new_tokens = 256  # Skip early stopping
do_sample = False
```

→ **tok/s: 61.63, TTFT: 16.22ms, VRAM: 0.9 GB** (run: mar17-r2, commit: a021e3d)

---

## Session Log

| Date | Run tag | Experiments | Best tok/s | Best config |
|------|---------|-------------|------------|-------------|
| 2026-03-17 | autoresearch/mar17-r2 | 19 | 61.63 | FP16 + TF32 + cudnn.benchmark + min_new_tokens=max_new_tokens |
