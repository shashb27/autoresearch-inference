# LEARNINGS.md — Cross-Run Knowledge Base

This file persists **across all autoresearch branches**.
The agent reads it at setup and updates it at the end of each session.
Do NOT reset or delete between runs — it is your long-term memory.

---

## Hardware Notes

- GPU: NVIDIA GeForce RTX 5090 (Blackwell, compute 12.0 / sm120)
- VRAM: 31.4 GB total, 28.2 GB budget
- BF16 supported, FP8 supported
- Measured memory bandwidth: 1524.8 GB/s (theoretical peak: 1792 GB/s)
- CUDA 13.0, PyTorch 2.11.0+cu130
- **BF16 is correct dtype** — FP16 produces garbage output (model trained in BF16)
- **Blackwell uses sm120** — many libraries/kernels are compiled for sm80 (suboptimal)

---

## What Works (confirmed gains)

- `torch.compile(mode="default", dynamic=True)` → +140% over uncompiled | Essential, kernel fusion dominates
- Eager attention (`attn_implementation="eager"`) → +6% over SDPA | Triton generates sm120-optimized kernels; SDPA uses sm80 cutlass
- `aggressive_fusion=True` + `combo_kernels=True` → +8.8% | Reduces kernel count and CPU dispatch overhead
- `coordinate_descent_tuning=True` → +8.4% | Auto-tunes Triton tile sizes (slow first compile)
- **Manual CUDA graph with StaticCache → +18.7%** | Biggest single win. Eliminates per-step kernel launch + Python overhead
  - Key: use StaticCache (fixed tensor addresses), capture on default stream
  - Self-feeding: argmax + input copy + position advance inside graph
  - ~393 tok/s stable (78% of bandwidth-limited theoretical max)
- `shape_padding=True` → ~0% | Free, no regression
- `epilogue_fusion=True` → already default, keep on
- `fx_graph_cache=True` → no runtime cost, saves compile time on reruns
- `expandable_segments:True` (PYTORCH_CUDA_ALLOC_CONF) → ~0.5% | Reduces memory fragmentation
- `guard_nn_modules=False` → slight guard overhead reduction
- TF32 matmul precision (`set_float32_matmul_precision('high')`) → free for FP32 ops
- BF16 reduced precision reduction + cuDNN benchmark → free, negligible gain

---

## What Doesn't Work (save time, skip these)

- **FP8 weight-only quantization (torchao)** → -31% | Dequant overhead dominates on <2B models
- **INT8 weight-only (torchao) + CUDA graphs** → crash | torchao INT8 not CUDA graph compatible
- **FP16 dtype** → garbage output (19/20 invalid) | Model trained in BF16, FP16 range too narrow
- **torch.compile(mode="reduce-overhead")** → crash | "accessing tensor output of CUDAGraphs overwritten" even with StaticCache
- **Static cache without CUDA graphs** → -38% | Cache implementation overhead outweighs benefits
- **SDPA attention + CUDA graphs** → -24% vs eager | sm80 cutlass kernels suboptimal for Blackwell sm120
- **max-autotune / max_autotune_gemm** → -12.5% | Selects suboptimal kernels for batch=1 small matmuls
- **Speculative decoding (0.5B draft)** → -19% | Draft model overhead dominates for <2B main model
- **flash-attn package** → can't install | CUDA toolkit 12.0 vs PyTorch CUDA 13.0 mismatch
- **vLLM** → incompatible | vLLM 0.2.5 vs torch 2.11.0+cu130
- **Compiled decode wrapper** → ~0% | torch.compile on wrapper function adds overhead
- **fullgraph=True** → ~0% | No effect
- **Pre-allocated output buffer (without CUDA graph)** → -1% | Buffer slicing hurts compile optimization
- **Triton cudagraphs (`triton.cudagraph_trees`)** → crash | All prompts fail
- **triton.multi_kernel** → unstable | Works on first run, crashes on second
- **permute_fusion + group_fusion + benchmark_fusion** → ~0% | No measurable effect
- **max_fusion_size=128** → ~0% | No measurable effect
- **epilogue_fusion_first** → ~0% | No measurable effect
- **GC disable during decode** → ~0% | Python GC stalls not measurable
- **Full-sequence CUDA graph (255 steps)** → -5% vs per-step graph | Larger graph = more replay overhead
- **dynamic=False compile** → -97% | Constant recompilation for each prompt length
- **benchmark_kernel=True** → ~0% | No measurable effect
- **HuggingFace model.generate()** → -93% | Massive pipeline overhead vs custom decode loop

---

## Near-Misses Worth Revisiting

- **INT8 quantization**: Works with torch.compile but not CUDA graphs. If CUDA graph compatibility improves in future torchao, could give significant speedup by halving weight reads.
- **CuDNN SDP backend**: 16% faster than Flash SDP on Blackwell (0.0086ms vs 0.0102ms per call). But loses Triton fusion opportunities. Might help with a model that has more attention-heavy workload.
- **Speculative decoding with a better draft**: The 0.5B draft had 68.6% acceptance rate but overhead dominated. A tiny draft (100M?) or MTP (if supported) could help on larger models.

---

## Best Config Found (so far)

```python
# Environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.guard_nn_modules = False
torch.set_float32_matmul_precision('high')

# Model
DTYPE = torch.bfloat16
ATTENTION_IMPLEMENTATION = "eager"

# Compilation
USE_TORCH_COMPILE = True
COMPILE_MODE = "default"
COMPILE_BACKEND = "inductor"
dynamic = True

# Inductor
fx_graph_cache = True
epilogue_fusion = True
coordinate_descent_tuning = True
shape_padding = True
aggressive_fusion = True
combo_kernels = True

# Generation
StaticCache + manual per-step CUDA graph capture
Self-feeding: argmax + copy-back + position advance inside graph
```

**Best result: ~397 tok/s** (stable 387-398 range, avg ~393)
- Model: Qwen2.5-1.5B (1.54B params)
- GPU: RTX 5090
- VRAM: 3.0 GB peak
- Theoretical bandwidth limit: ~508 tok/s
- Efficiency: 78% of theoretical

---

## Session Log

| Date | Run tag | Experiments | Best tok/s | Best config |
|------|---------|-------------|------------|-------------|
| 2026-03-26 | qwen1.5b-rtx5090 | 20+ experiments | 397.76 | BF16 + eager attn + compile(default) + inductor(aggressive_fusion, combo_kernels, coord_descent) + StaticCache + CUDA graph |
