# autoresearch-inference

Autonomous LLM inference optimization research. An AI agent iteratively modifies an inference script, benchmarks it, keeps improvements, and reverts failures — running 100+ experiments overnight.

Inspired by [autoresearch](https://github.com/karpathy/autoresearch), but targeting **inference speed (tok/s)** instead of training quality.

## How it works

1. `prepare.py` (read-only) — Downloads model, runs standardized benchmark, validates output
2. `infer.py` (agent-modified) — Inference pipeline: model loading, optimization, generation
3. `program.md` — Instructions for the AI agent's autonomous experiment loop

The agent modifies `infer.py`, runs `uv run infer.py`, checks if tok/s improved, keeps or reverts, and loops forever.

## Target

- **Model**: Qwen 2.5 7B
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell (96GB GDDR7)
- **Metric**: tokens/second (higher is better)
- **Agent**: Claude Code running autonomously on the GPU machine

## Setup

```bash
# Install uv if needed
curl -fsSL https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/shashb27/autoresearch-inference
cd autoresearch-inference
uv sync

# Download model and run baseline
uv run prepare.py
```

## Run the agent

```bash
# Start Claude Code with the experiment instructions
claude -p "$(cat program.md)"
```

The agent will run autonomously, modifying `infer.py` and benchmarking every ~5 minutes. Check `results.tsv` for experiment history.

## Project structure

```
prepare.py             # READ-ONLY: Benchmark harness
infer.py               # MUTABLE: Agent modifies this
program.md             # Agent instructions
results.tsv            # Experiment log
prompts/prompts.json   # Fixed benchmark prompts
analysis.ipynb         # Visualization notebook
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
