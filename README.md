# autoresearch-inference

Autonomous LLM inference optimization. An AI agent iteratively modifies an inference script, benchmarks it, keeps improvements, and reverts failures — running 100+ experiments overnight to find the fastest possible configuration for your GPU.

Inspired by [autoresearch](https://github.com/karpathy/autoresearch), targeting **inference throughput (tok/s)** instead of training quality.

---

## How it works

```
Profile → Hypothesize → Modify infer.py → Benchmark → Keep or Revert → Repeat
```

1. **`prepare.py`** (read-only harness) — Discovers GPU, downloads model, writes `config.json` + `hardware.json`, runs the standardized benchmark, optionally profiles with `torch.profiler`
2. **`infer.py`** (agent-modified) — Inference pipeline: model loading, dtype, attention backend, KV cache, generation loop, `torch.compile` settings
3. **`program.md`** — Agent instructions: decision framework, hypothesis format, keep/discard rules

**The agent is evidence-driven, not exploratory:**
- Reads `profile.txt` to find the actual bottleneck before choosing what to try
- States a specific hypothesis with a predicted % gain before each experiment
- Skips experiments predicted to gain < 3% (below measurement noise floor)
- Re-profiles after 3 consecutive discards to check if the bottleneck shifted
- Hard-resets to best known state after 5 consecutive discards

**Loop reliability features (v0.2):**
- OOM recovery — caught per-prompt, loop continues; experiment is auto-discarded
- Generation timeout — kills hung generate_fn after 120s
- Pre-flight validation — infer.py is validated before every benchmark run
- True TTFT — generate_fn measures actual prefill latency (not avg token latency)
- Full profiler output — complete profile.txt, no truncation

---

## Quick start

```bash
# Install uv if needed
curl -fsSL https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/shashb27/autoresearch-inference
cd autoresearch-inference
uv sync

# Setup: discover GPU, download model, run baseline benchmark
uv run prepare.py

# Or with a different model:
uv run prepare.py --model "meta-llama/Llama-3.1-8B"
```

---

## Run the agent loop

```bash
# Full automated loop (setup → profile → agent → plots → commit)
./run_loop.sh

# With a specific model:
./run_loop.sh --model "Qwen/Qwen2.5-0.5B"
```

Or run steps manually:

```bash
# 1. Baseline benchmark + hardware discovery
uv run prepare.py --model "Qwen/Qwen2.5-7B"

# 2. Profile bottlenecks (agent reads this before experimenting)
uv run prepare.py --model "Qwen/Qwen2.5-7B" --profile

# 3. Validate infer.py before running (optional, also runs automatically)
uv run prepare.py --validate

# 4. Start the agent
claude -p "$(cat program.md)" --allowedTools "Bash,Read,Write,Edit,Glob,Grep"
```

**Tip:** Run inside `tmux` so the agent survives SSH disconnects:
```bash
tmux new -s autoresearch
./run_loop.sh
```

---

## CLI reference

| Command | Description |
|---|---|
| `uv run prepare.py` | Full setup: GPU discovery, model download, baseline benchmark |
| `uv run prepare.py --model MODEL` | Use a specific HuggingFace model |
| `uv run prepare.py --check` | Verify project structure (no GPU required) |
| `uv run prepare.py --profile` | Profile with torch.profiler → writes `profile.txt` |
| `uv run prepare.py --validate` | Validate `infer.py` interface without benchmarking |
| `uv run infer.py` | Run inference + benchmark directly |
| `uv run analyze.py` | Generate plots from `results.tsv` |
| `uv run pytest tests/ -v` | Run unit test suite |

---

## Visualize results

```bash
uv run analyze.py                  # outputs to plots/
uv run analyze.py --out my_plots/  # custom output directory
uv run analyze.py --show           # also open in browser
```

Generates 5 plots:

| Plot | Description |
|---|---|
| `tok_s_progression.png` | Speed over time with best-so-far line and improvement annotations |
| `vram_vs_toks.png` | VRAM vs tok/s efficiency frontier |
| `improvement_deltas.png` | % gain per kept experiment vs baseline |
| `outcomes_donut.png` | Keep / discard / crash ratio |
| `tok_s_vs_ttft.png` | Throughput vs first-token latency (kept experiments only) |

---

## What the agent can optimize

Everything in **`infer.py`** is fair game:

- **Dtype** — BF16, FP16, FP8 (hardware-dependent)
- **Attention backend** — SDPA, Flash Attention 2, eager
- **Quantization** — INT8, INT4, FP8 via TorchAO or bitsandbytes
- **Compilation** — `torch.compile` mode, backend, dynamic shapes, cache size
- **Generation loop** — Custom decode loop vs HF `generate()`, KV cache strategy
- **CUDA graphs** — Graph capture for fixed-shape decode steps
- **Package installation** — `uv add flash-attn`, `bitsandbytes`, etc.

**`prepare.py`** and **`prompts/prompts.json`** are never modified — they are the objective measurement ground truth.

---

## generate_fn interface

`infer.py` must expose `run_inference()` returning `(generate_fn, tokenizer)`.

`generate_fn` accepts a single input tensor and can return either:
- `output_ids: torch.Tensor` — legacy format, still supported
- `(output_ids: torch.Tensor, metadata: dict)` — preferred; metadata may include:
  - `"ttft_ms": float` — true time-to-first-token in milliseconds (prefill latency)

The benchmark harness handles both return forms automatically.

---

## Benchmark output format

```
tok_s:            275.14
ttft_ms:          6.80
peak_vram_gb:     4.1
total_prompts:    20
total_tokens:     5120
valid_outputs:    20
invalid_outputs:  0
```

Extract key metrics:
```bash
grep "^tok_s:\|^ttft_ms:\|^peak_vram_gb:" run.log
```

---

## results.tsv format

Tab-separated, one row per experiment:

```
commit  tok_s   ttft_ms peak_vram_gb    status  description
a1b2c3d 145.79  8.50    4.1     keep    baseline: BF16 + SDPA + compile default
b2c3d4e 200.30  7.20    4.2     keep    experiment: custom decode loop
c3d4e5f 0.00    0.0     0.0     crash   experiment: INT4 missing bitsandbytes
```

Status values: `keep`, `discard`, `crash`

---

## Results

Actual measured results from autonomous agent runs.

### Run log

| Date | Branch | Model | GPU | Experiments | Best tok/s | Gain |
|---|---|---|---|---|---|---|
| Mar 17 2026 | [mar17](../../tree/autoresearch/mar17) | Qwen2.5-0.5B | RTX 3060 Ti | 16 | 61.45 | +6.6% |
| Mar 17 2026 | [mar17-r2](../../tree/autoresearch/mar17-r2) | Qwen2.5-0.5B | RTX 3060 Ti | 18 | 61.63 | +5.1% |

### Optimization findings — Qwen2.5-0.5B · RTX 3060 Ti (Ampere sm_86)

Baseline: **57.6 tok/s** (FP16 + SDPA + torch.compile default)

| Technique | tok/s | vs baseline | Verdict | Notes |
|---|---|---|---|---|
| BF16 dtype | 61.3 | +6.4% | keep | Native on Ampere — biggest single win |
| `min_new_tokens = max_new_tokens` | 61.6 | +6.9% | keep | Skips early-stop overhead |
| TF32 matmul precision | 60.9 | +5.7% | keep | `torch.backends.cuda.matmul.allow_tf32 = True` |
| `use_cache=True` + `return_dict=False` | 61.5 | +6.6% | keep | Minor but free |
| torch.compile reduce-overhead | 58.0 | +0.6% | discard | No meaningful gain over default |
| torch.compile max-autotune | 58.8 | +2.1% | discard | Not worth compile time |
| torch.compile fullgraph=True | 60.2 | +4.5% | discard | Marginal, less stable |
| BF16 eager (no compile) | 59.3 | +3.0% | discard | Compile helps |
| BetterTransformer | 58.4 | +1.4% | discard | Negligible |
| INT8 weight-only (torchao) | 35.0 | -39% | discard | Overhead dominates at 0.5B scale |
| INT4 quantization | — | — | crash | Missing `mslk>=1.0.0` |
| Static cache + compile | — | — | crash | Triton compile error on RTX 3060 Ti |

> **Key insight:** At 0.5B scale the model is too small for quantization to help — kernel overhead outweighs memory savings. Expect INT8/INT4 to be beneficial at 7B+.

---

## Running tests

```bash
# Run full test suite (no GPU required)
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ -v --cov --cov-report=term-missing
```

The test suite (49 tests) covers:
- `InferConfig` schema validation (4 tests)
- Hardware capability detection for BF16/FP8 (8 tests)
- Output validation — all rejection paths (8 tests)
- Prompt loading integrity (3 tests)
- Generation timeout + OOM recovery (5 tests)
- `analyze.py` data loading + validation (7 tests)
- Plot generation smoke tests (6 tests)
- Summary statistics correctness (3 tests)
- Helper utilities (3 tests) + pre-flight validation (2 tests)

---

## Project structure

```
prepare.py             # READ-ONLY: GPU discovery, benchmark harness, profiling
infer.py               # MUTABLE: Agent modifies this each experiment
analyze.py             # Visualization: generates plots from results.tsv
program.md             # Agent instructions + decision framework
LEARNINGS.md           # Cross-run knowledge base (persists across branches)
run_loop.sh            # Full session launcher (setup → agent → plots → commit)
run_multi.sh           # Multi-model experiment runner
run_resume.sh          # Resume an interrupted session
results.tsv            # Experiment log (tab-separated, one row per experiment)
prompts/prompts.json   # Fixed benchmark prompts (READ-ONLY, 20 prompts)
tests/                 # Unit test suite (pytest, no GPU required)
pyproject.toml         # Dependencies + test configuration
config.json            # Generated: model path, device, VRAM limit
hardware.json          # Generated: GPU capabilities for agent
profile.txt            # Generated: full torch.profiler output (top 50 ops)
plots/                 # Generated: visualization output from analyze.py
```

---

## Architecture notes

### Why two files?

`prepare.py` is the immutable ground truth — it never changes between experiments. `infer.py` is the experimental surface — the agent rewrites it freely. This separation guarantees that every measurement is taken with the same yardstick.

### Why subprocess validation?

Before running a 5-minute benchmark, `prepare.py` validates that `infer.py` exports `run_inference()` in a subprocess. This catches syntax errors and missing functions without contaminating the main process's state.

### Why threads for timeout?

`generate_fn` runs in a daemon thread with a hard timeout. If it hangs (bad CUDA kernel, infinite decode loop), the thread is abandoned and the benchmark continues with the next prompt rather than blocking forever.

### TTFT measurement

`generate_fn` measures true time-to-first-token by recording a `torch.cuda.synchronize()` timestamp after the first forward pass (prefill + first decode step). This is returned in the metadata dict and replaces the previous proxy (avg latency per token, which conflated prefill and decode).

---

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA
- `uv` package manager

Models are downloaded from HuggingFace Hub on first run. Default model is `Qwen/Qwen2.5-7B` (~15 GB download).
