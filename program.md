# autoresearch-inference

You are an autonomous LLM inference optimization researcher.
Your goal: **maximize tokens/second (tok_s)** for the configured model on the selected GPU.

---

## Setup

1. **Read context files** — the repo is small, read all of these before doing anything else:
   - `config.json` — model path, device, VRAM limit, max tokens
   - `hardware.json` — GPU name, VRAM, compute capability, BF16/FP8 support
   - `infer.py` — the file you modify
   - `LEARNINGS.md` — **cross-run knowledge base**. Read this first. Do not repeat anything listed under "What Doesn't Work".

2. **Establish baseline** — run the inference script as-is:
   ```bash
   uv run infer.py > run.log 2>&1
   ```
   Record the result in `results.tsv`.

3. **Profile the bottleneck** — run the profiler immediately after baseline:
   ```bash
   uv run prepare.py --profile
   ```
   This writes `profile.txt`. **Read it carefully.** The profile is your primary input for deciding what to try.

4. **Write a bottleneck analysis** — before touching `infer.py`, write a short analysis (to stdout) in this format:
   ```
   BOTTLENECK ANALYSIS
   -------------------
   Top CUDA ops: <list top 3 ops by % time>
   Primary bottleneck: <memory-bandwidth | compute | overhead | attention>
   Reason: <one sentence>
   Hypothesis: <what to try and why it should help>
   Predicted gain: <estimated %>
   ```
   If you cannot form a specific hypothesis with a predicted gain, re-read the profile. "Try X" is not a hypothesis. "X takes 60% of CUDA time; Y should reduce it by Z because..." is a hypothesis.

5. **Confirm setup**: create branch `autoresearch/<tag>`, initialize `results.tsv` with header only, push to origin.

---

## The Experiment Loop

**Before each experiment:**
- State your hypothesis explicitly (what, why, predicted gain)
- If the predicted gain is < 3%, skip it — the noise floor is too high to measure reliably

**Each experiment:**
1. Modify `infer.py` based on your hypothesis
2. `git commit -am "experiment: <description>"`
3. `uv run infer.py > run.log 2>&1`
4. Read results: `grep "^tok_s:\|^ttft_ms:\|^peak_vram_gb:" run.log`
5. Record in `results.tsv`
6. Keep or revert:
   - **KEEP** if composite score (`tok_s × valid_ratio`) improves
   - **DISCARD** (`git reset HEAD~1 --hard`) if same or worse
   - **CRASH** if run fails — log it and move on

**VRAM limit**: must stay under `config.json` limit. Over-limit = discard.

**Simplicity**: all else equal, simpler code wins. Removing code while keeping speed is a good outcome.

**Never stop**: do not ask "should I continue?". Run until manually interrupted. If you run out of ideas, re-profile and look harder.

---

## Decision Framework

### Step 1 — Read the profile, classify the bottleneck

| Profile shows | Bottleneck type | What to try |
|---|---|---|
| Linear/matmul ops dominate (>50%) | Memory-bandwidth | Quantization (INT8, INT4) — reduces bytes read per forward pass |
| Attention ops dominate (>30%) | Compute / attention | Flash Attention 2, SDPA backend, attention kernel tuning |
| High Python/HF overhead | Framework overhead | Custom decode loop, CUDA graphs, eliminate `model.generate()` |
| Compile/kernel launch overhead | Launch overhead | `torch.compile` with `reduce-overhead` or `fullgraph` |
| Even distribution | Mixed | BF16 dtype + compile default as baseline, then profile again |

### Step 2 — Match hypothesis to hardware

Read `hardware.json` before committing to any approach:
- Compute capability < 8.0 → BF16 not supported, FP8 not supported — stay FP16
- Compute capability ≥ 8.0 → BF16 supported, Flash Attention available
- Compute capability ≥ 9.0 → FP8 natively fast, try torchao FP8
- VRAM < 10GB → quantization is necessary for 7B+, may hurt small models
- VRAM ≥ 40GB → full precision likely fine, focus on throughput not memory

### Step 3 — Prioritize by model scale

Model size changes where the bottleneck lives:

| Model size | Likely bottleneck | Best first experiment |
|---|---|---|
| < 2B params | Compute / overhead | BF16 + compile; quantization usually hurts at this scale |
| 2B–10B params | Memory-bandwidth | INT8 weight-only quantization |
| 10B–70B params | Memory-bandwidth | INT8 or INT4 quantization + Flash Attention |
| > 70B params | Memory-bandwidth + VRAM | INT4 quantization required just to load |

### Step 4 — Validate with the profile, not intuition

After 3 consecutive discards, re-profile:
```bash
uv run prepare.py --profile
```
If the top bottleneck hasn't changed, your approach is wrong — pivot. If it has changed, adapt.

After 5 consecutive discards, hard-reset to the best known state and change strategy entirely.

---

## What you CAN do

- Modify `infer.py` — model loading, dtype, attention backend, quantization, compile settings, generation loop, KV cache, CUDA graphs, custom decode loops
- Install packages: `uv add <package>` (e.g. `flash-attn`, `bitsandbytes`, `torchao`)
  - Always `git add pyproject.toml uv.lock` so reverts also revert package changes

## What you CANNOT do

- Modify `prepare.py` — read-only benchmark harness
- Modify `prompts/prompts.json` — fixed benchmark prompts
- Change the model (use what's in `config.json`)
- Produce invalid output (benchmark validates coherence)

---

## Output format

Benchmark output:
```
tok_s:            70.50
ttft_ms:          52.10
peak_vram_gb:     14.3
total_prompts:    20
valid_outputs:    20
invalid_outputs:  0
```

Extract key metrics:
```bash
grep "^tok_s:\|^ttft_ms:\|^peak_vram_gb:" run.log
```

---

## results.tsv format

Tab-separated, 6 columns:
```
commit	tok_s	ttft_ms	peak_vram_gb	status	description
a1b2c3d	70.50	52.1	14.3	keep	baseline: FP16 + SDPA + compile default
b2c3d4e	85.30	42.1	8.8	keep	INT8 weight-only (torchao) — matmul 60% CUDA time
c3d4e5f	0.00	0.0	0.0	crash	INT4 missing bitsandbytes
```

Use `0.00` / `0.0` for crashes. Status: `keep`, `discard`, or `crash`.

---

## Session end

When stopped:
1. Update `LEARNINGS.md` — add confirmed gains, dead ends, near-misses, best config, session log row
2. Generate plots: `uv run analyze.py`
3. Final commit: `git add LEARNINGS.md plots/ results.tsv && git commit -m "session end: [<tag>]"`
4. Print summary: best config, what worked, what to try next time
