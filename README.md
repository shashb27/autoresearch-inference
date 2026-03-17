# autoresearch-inference

Autonomous LLM inference optimization research. An AI agent iteratively modifies an inference script, benchmarks it, keeps improvements, and reverts failures — running 100+ experiments overnight.

Inspired by [autoresearch](https://github.com/karpathy/autoresearch), but targeting **inference speed (tok/s)** instead of training quality.

## How it works

1. `prepare.py` (read-only) — Discovers GPU, downloads model, writes config, runs benchmark
2. `infer.py` (agent-modified) — Inference pipeline: model loading, optimization, generation
3. `program.md` — Instructions for the AI agent's autonomous experiment loop

The agent modifies `infer.py`, runs `uv run infer.py`, checks if tok/s improved, keeps or reverts, and loops forever.

## Features

- **Auto GPU discovery** — Picks the GPU with the most free VRAM (handles shared machines)
- **Any model** — Pass `--model` to use any HuggingFace model
- **Hardware-aware agent** — Writes `hardware.json` so the agent knows GPU capabilities (FP8, BF16, compute capability)
- **Profiling** — `--profile` flag writes `profile.txt` so the agent can find actual bottlenecks
- **Package installation** — Agent can `uv add` packages (flash-attn, bitsandbytes, etc.)

## Quick start

```bash
# Install uv if needed
curl -fsSL https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/shashb27/autoresearch-inference
cd autoresearch-inference
uv sync

# Setup: discover GPU, download model, run baseline
uv run prepare.py

# Or use a different model:
uv run prepare.py --model "meta-llama/Llama-3.1-8B"
```

## Run the agent

```bash
# Start Claude Code with the experiment instructions
claude -p "$(cat program.md)" --allowedTools "Bash,Read,Write,Edit,Glob,Grep"
```

The agent will run autonomously, modifying `infer.py` and benchmarking every ~5 minutes. Check `results.tsv` for experiment history.

**Tip**: Run in `tmux` so the agent survives SSH disconnects.

## CLI options

```bash
uv run prepare.py                            # Default: Qwen 2.5 7B, auto GPU
uv run prepare.py --model "Qwen/Qwen2.5-7B"  # Explicit model
uv run prepare.py --check                     # Verify project structure (no GPU)
uv run prepare.py --profile                   # Profile bottlenecks
```

## Visualize results

After a run (or during one), generate plots from `results.tsv`:

```bash
uv run analyze.py                  # outputs to plots/
uv run analyze.py --out my_plots/  # custom output dir
uv run analyze.py --show           # also open in browser
```

Generates:
- `tok_s_progression.png` — speed over time with best-so-far line and improvement annotations
- `vram_vs_toks.png` — VRAM vs tok/s efficiency frontier
- `improvement_deltas.png` — % gain per kept experiment vs baseline
- `outcomes_donut.png` — keep / discard / crash ratio
- `tok_s_vs_ttft.png` — throughput vs latency dual-axis (kept experiments only)

## Project structure

```
prepare.py             # READ-ONLY: GPU discovery, benchmark harness
infer.py               # MUTABLE: Agent modifies this
analyze.py             # Visualization: generates plots from results.tsv
program.md             # Agent instructions + decision framework
LEARNINGS.md           # Cross-run knowledge base (persists across branches)
results.tsv            # Experiment log (created per run)
prompts/prompts.json   # Fixed benchmark prompts
analysis.ipynb         # Legacy visualization notebook
config.json            # Generated: model path, device, VRAM limit
hardware.json          # Generated: GPU capabilities for agent
profile.txt            # Generated: torch.profiler output
plots/                 # Generated: visualization output from analyze.py
```

## Expected progression

| Optimization | Est. tok/s | Gain |
|---|---|---|
| Baseline (FP16 + compile) | ~70 | — |
| + Flash Attention 2 | ~85 | +21% |
| + INT8 quantization | ~120 | +41% |
| + INT4 quantization | ~160 | +33% |
| + Custom decode loop | ~180 | +12% |
| + CUDA graphs | ~200 | +11% |
| + Speculative decoding | ~250+ | +25% |
