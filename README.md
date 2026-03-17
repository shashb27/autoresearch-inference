# autoresearch-inference

Autonomous LLM inference optimization research. An AI agent iteratively modifies an inference script, benchmarks it, keeps improvements, and reverts failures — running 100+ experiments overnight.

Inspired by [autoresearch](https://github.com/karpathy/autoresearch), but targeting **inference speed (tok/s)** instead of training quality.

## How it works

1. `prepare.py` (read-only) — Discovers GPU, downloads model, writes `config.json` + `hardware.json`, runs baseline benchmark, optionally profiles with torch.profiler
2. `infer.py` (agent-modified) — Inference pipeline: model loading, optimization, generation
3. `program.md` — Instructions for the AI agent's autonomous experiment loop

**The agent is evidence-driven, not exploratory:**
- Reads `profile.txt` to identify the actual bottleneck before choosing what to try
- States a hypothesis with a predicted % gain before each experiment
- Skips experiments with predicted gain < 3% (below measurement noise floor)
- Re-profiles after 3 consecutive discards to check if the bottleneck shifted

The loop: read profile → form hypothesis → modify `infer.py` → benchmark → improved? keep commit : revert → repeat.

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
# 1. Baseline benchmark
uv run prepare.py --model "Qwen/Qwen2.5-7B"

# 2. Profile bottlenecks (required — agent reads this before experimenting)
uv run prepare.py --model "Qwen/Qwen2.5-7B" --profile

# 3. Start the agent
claude -p "$(cat program.md)" --allowedTools "Bash,Read,Write,Edit,Glob,Grep"
```

The agent reads `profile.txt`, identifies the bottleneck, states a hypothesis, then modifies `infer.py` and benchmarks. Check `results.tsv` for experiment history.

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

## Results

Actual measured results from autonomous agent runs. Each branch is one session; the agent runs ~16 experiments per session and commits findings to `results.tsv`.

### Run log

| Date | Branch | Model | GPU | Experiments | Best tok/s | Gain |
|---|---|---|---|---|---|---|
| Mar 17 2026 | [mar17](../../tree/autoresearch/mar17) | Qwen2.5-0.5B | RTX 3060 Ti | 16 | 61.45 | +6.6% |
| Mar 17 2026 | [mar17-r2](../../tree/autoresearch/mar17-r2) | Qwen2.5-0.5B | RTX 3060 Ti | 18 | 61.63 | +5.1% |

### Optimization findings — Qwen2.5-0.5B · RTX 3060 Ti (Ampere sm_86)

Baseline: **57.6 tok/s** (FP16 + SDPA + torch.compile default)

| Technique | tok/s | vs baseline | Verdict | Notes |
|---|---|---|---|---|
| BF16 dtype | 61.3 | +6.4% | ✅ keep | Native on Ampere — biggest single win |
| min_new_tokens = max_new_tokens | 61.6 | +6.9% | ✅ keep | Skips early-stop overhead |
| TF32 matmul precision | 60.9 | +5.7% | ✅ keep | `torch.backends.cuda.matmul.allow_tf32 = True` |
| use_cache=True + return_dict=False | 61.5 | +6.6% | ✅ keep | Minor but free |
| torch.compile (default mode) | baseline | — | ✅ baseline | Already in baseline |
| torch.compile reduce-overhead | 58.0 | +0.6% | ❌ discard | No meaningful gain over default |
| torch.compile max-autotune | 58.8 | +2.1% | ❌ discard | Not worth compile time |
| torch.compile fullgraph=True | 60.2 | +4.5% | ❌ discard | Marginal, less stable |
| BF16 eager mode (no compile) | 59.3 | +3.0% | ❌ discard | Compile helps |
| BetterTransformer | 58.4 | +1.4% | ❌ discard | Negligible |
| INT8 weight-only (torchao) | 35.0 | **-39%** | ❌ discard | Overhead dominates at 0.5B scale |
| INT4 quantization | — | — | 💥 crash | Missing `mslk>=1.0.0` |
| Static cache + compile | — | — | 💥 crash | Triton compile error on RTX 3060 Ti |
| Custom decode loop + compile | — | — | 💥 crash | Triton compile error |

> **Key insight:** At 0.5B scale the model is too small for quantization to help — kernel overhead outweighs memory savings. Expect INT8/INT4 to be beneficial at 7B+.

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

