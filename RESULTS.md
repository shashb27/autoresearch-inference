# Autoresearch Inference Results

## NVIDIA GeForce RTX 4090 (24GB, Ada Lovelace, sm_89)

### Qwen2.5-0.5B

| # | Experiment | tok/s | ttft_ms | VRAM (GB) | vs Baseline | Status |
|---|---|---|---|---|---|---|
| 0 | Baseline: FP16 + SDPA + compile default | 145.79 | 6.87 | 0.9 | — | keep |
| 1 | BF16 dtype | 145.58 | 6.86 | 0.9 | -0.1% | discard |
| 2 | TF32 + min_new_tokens + use_cache | 141.59 | 7.04 | 0.9 | -2.9% | discard |
| 3 | TF32 matmul precision only | 143.22 | 7.00 | 0.9 | -1.8% | discard |
| 4 | Custom decode + no compile | 144.73 | 6.90 | 1.0 | -0.7% | discard |
| 5 | Custom decode + reduce-overhead | — | — | — | — | crash |
| 6 | Custom decode + compile + dynamo cache=64 | 201.75 | 3.36 | 1.0 | +38.4% | keep |
| 7 | Custom decode + compile + dynamic=True | 266.26 | 3.76 | 1.0 | +82.6% | keep |
| 8 | **+ BF16 dtype** | **275.14** | **3.63** | **1.0** | **+88.7%** | **keep** |
| 9 | max-autotune compile | — | — | — | — | crash |
| 10 | Preallocated output buffer | 268.68 | 3.72 | 1.0 | +15.7% | discard |
| 11 | float32 matmul precision high | 263.45 | 3.80 | 1.0 | +80.7% | discard |
| 12 | INT8 weight-only (torchao) | 225.62 | 4.29 | 0.9 | +54.8% | discard |

**Best config:** BF16 + custom decode loop + torch.compile(dynamic=True, mode="default") + dynamo cache_size_limit=64
**Best tok/s:** 275.14 (+88.7% over baseline)
**Date:** 2026-03-22
