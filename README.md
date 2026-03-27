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

The agent is **evidence-driven**:
- Reads `profile.txt` to find the actual bottleneck before choosing what to try
- States a specific hypothesis with a predicted % gain before each experiment
- Skips experiments predicted to gain < 3% (below measurement noise floor)
- Re-profiles after 3 consecutive discards to check if the bottleneck shifted

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
uv run prepare.py --model "meta-llama/Llama-3.2-1B"

# Profile bottlenecks (agent reads this before experimenting)
uv run prepare.py --profile
```

> **Note:** `--model` is required on first run. After that, `prepare.py` reads from `config.json` automatically.

> **Note:** For gated models (Llama, etc.), set your HuggingFace token first: `export HF_TOKEN=your_token`

---

## Run the agent

```bash
# Start in tmux so it survives SSH disconnects
tmux new -s autoresearch

# Launch the agent (pipe program.md, don't use $() — backticks in the prompt break shell expansion)
claude --allowedTools "Bash,Read,Write,Edit,Glob,Grep" < program.md
```

The agent will:
1. Read `config.json`, `hardware.json`, and `results.tsv`
2. Form a hypothesis based on profiling data
3. Edit `infer.py`
4. Run the benchmark
5. Keep or discard based on tok/s
6. Repeat until stopped

Detach from tmux with `Ctrl+B, D`. Reattach with `tmux attach -t autoresearch`.

---

## Submit to the leaderboard

After the agent finishes:

```bash
uv run submit_run.py --contributor "yourname"
uv run leaderboard.py
git add leaderboard/ && git commit -m "leaderboard: add run"
git push
```

View the leaderboard at `leaderboard/index.html`.

---

## CLI reference

| Command | Description |
|---|---|
| `uv run prepare.py --model MODEL` | Setup: GPU discovery, model download, baseline benchmark |
| `uv run prepare.py --profile` | Profile with torch.profiler → writes `profile.txt` |
| `uv run prepare.py --check` | Verify project structure (no GPU required) |
| `uv run prepare.py --validate` | Validate `infer.py` interface without benchmarking |
| `uv run submit_run.py` | Submit run results to the leaderboard |
| `uv run leaderboard.py` | Regenerate `leaderboard/index.html` |
| `uv run analyze.py` | Generate plots from `results.tsv` |

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

## Project structure

```
prepare.py             # READ-ONLY: GPU discovery, benchmark harness, profiling
infer.py               # MUTABLE: Agent modifies this each experiment
program.md             # Agent instructions + decision framework
submit_run.py          # Submit results to community leaderboard
leaderboard.py         # Regenerate leaderboard HTML from run JSONs
leaderboard/           # Community leaderboard (index.html + runs/*.json)
analyze.py             # Visualization: generates plots from results.tsv
LEARNINGS.md           # Cross-run knowledge base (persists across branches)
results.tsv            # Experiment log (tab-separated, one row per experiment)
prompts/prompts.json   # Fixed benchmark prompts (READ-ONLY, 20 prompts)
tests/                 # Unit test suite (pytest, no GPU required)
pyproject.toml         # Dependencies + test configuration
config.json            # Generated: model path, device, VRAM limit
hardware.json          # Generated: GPU capabilities for agent
profile.txt            # Generated: full torch.profiler output
```

---

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA
- `uv` package manager
- [Claude Code](https://claude.ai/code) CLI (for running the agent)

Models are downloaded from HuggingFace Hub on first run.

---

## Running tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ -v --cov --cov-report=term-missing
```
