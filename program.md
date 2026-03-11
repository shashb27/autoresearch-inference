# autoresearch-inference

You are an autonomous LLM inference optimization researcher.
Your goal: **maximize tokens/second (tok_s)** for the configured model on the selected GPU.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read context files**: The repo is small. Read these files:
   - `config.json` — model path, device, VRAM limit, max tokens (written by prepare.py)
   - `hardware.json` — GPU name, VRAM, compute capability, FP8/BF16 support
   - `infer.py` — the file you modify. Model loading, optimization, generation loop.
   - `README.md` — repository context.
   - `prepare.py` — fixed benchmark harness, metrics, validation. Do not modify.
4. **Verify model exists**: Check that the model path from `config.json` contains model files. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The inference benchmark runs for a **fixed set of 20 prompts**, generating tokens with greedy decoding. You launch it simply as: `uv run infer.py`.

**What you CAN do:**
- Modify `infer.py` — this is the only file you edit. Everything is fair game: model loading strategy, dtype, attention implementation, quantization, torch.compile settings, generation loop, KV cache management, memory allocation, speculative decoding, CUDA graphs, custom decode loops.
- Install packages with `uv add <package>` — when an optimization requires a package not in pyproject.toml (e.g. flash-attn, bitsandbytes, vllm), install it first. Always `git add pyproject.toml uv.lock` with the experiment commit so reverted experiments also revert package changes.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed benchmark harness, output validation, and metric computation.
- Modify `prompts/prompts.json`. The benchmark prompts are fixed.
- Change the model (must use the model specified in `config.json`).
- Produce invalid output (the benchmark validates that generated text is coherent).

**The goal is simple: get the highest tok_s.** Since the benchmark is fixed (same prompts, same token count), you don't need to worry about evaluation variance. Everything is fair game: change how the model loads, compiles, quantizes, generates. The only constraint is that the code runs without crashing and produces valid output.

**VRAM** must stay under the limit in `config.json`. Experiments exceeding this are discarded.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code while maintaining or improving tok_s is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so run the inference script as-is.

## Decision Framework

**Before each experiment, think systematically:**

1. **Read `hardware.json`** — Understand what your GPU supports.
   - Compute capability ≥ 9.0 → FP8 is natively supported (fast path)
   - Compute capability ≥ 8.0 → BF16 is supported, flash attention available
   - VRAM determines how aggressive you can be with quantization vs. keeping full precision
   - GPU architecture hints at which kernels will be fastest

2. **Read `profile.txt`** (if it exists) — Find the actual bottleneck.
   - If attention dominates → try flash attention, different attention backends
   - If linear layers dominate → try quantization (reduces memory bandwidth)
   - If overhead is high → try custom decode loop, CUDA graphs
   - If compile overhead is high → try different compile modes or skip compile

3. **Read `results.tsv`** — Learn from past experiments.
   - What worked? What didn't? Why?
   - Don't repeat failed approaches unless you have a new angle
   - Look for patterns: did quantization crash (missing package?) or produce garbage (needs calibration?)

4. **Form a hypothesis**: "The bottleneck is X, changing Y should help because Z."
   - Be specific. "Try INT8" is weak. "Linear layers are 60% of CUDA time, INT8 weight-only quantization halves memory bandwidth for these layers" is strong.

5. **After 3 consecutive discards** — Re-profile to check if the bottleneck shifted:
   ```bash
   uv run prepare.py --profile
   ```
   Read the new `profile.txt` and adjust your strategy.

**Why this matters**: LLM inference is memory-bandwidth-bound during decode (one token at a time, each requiring a full pass through all weights). Reducing weight precision (quantization) directly reduces bytes read per token, increasing tok/s. But the GPU must support the target precision natively, or the overhead of dequantization eats the gains. Hardware awareness prevents wasted experiments.

## Output format

Once the benchmark finishes it prints a summary like this:

```
---
tok_s:            70.50
ttft_ms:          52.10
peak_vram_gb:     14.3
total_prompts:    20
total_tokens:     5120
valid_outputs:    20
invalid_outputs:  0
```

You can extract the key metric from the log file:

```
grep "^tok_s:\|^peak_vram_gb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	tok_s	ttft_ms	peak_vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. tok_s achieved (e.g. 70.50) — use 0.00 for crashes
3. ttft_ms (e.g. 52.10) — use 0.00 for crashes
4. peak_vram_gb, round to .1f (e.g. 14.3) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	tok_s	ttft_ms	peak_vram_gb	status	description
a1b2c3d	70.50	52.1	14.3	keep	baseline: FP16 + torch.compile default
b2c3d4e	85.30	42.1	8.8	keep	INT8 weight-only quantization via torchao
c3d4e5f	0.00	0.0	0.0	crash	INT4 quant missing bitsandbytes
d4e5f6g	82.10	40.5	8.6	discard	INT8 + compile reduce-overhead (slower)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar10`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `infer.py` with an experimental idea by directly hacking the code.
3. If you installed a new package, stage pyproject.toml and uv.lock too.
4. git commit -am "experiment: <description>"
5. Run the experiment: `uv run infer.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^tok_s:\|^ttft_ms:\|^peak_vram_gb:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv
9. If tok_s improved (higher), you "advance" the branch, keeping the git commit
10. If tok_s is equal or worse, you git reset back to where you started: `git reset HEAD~1 --hard`

**Timeout**: Each experiment should take ~5 minutes total. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import, a missing package you can `uv add`), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the code, re-read hardware.json, re-profile, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

## Research directions (adapt to hardware.json capabilities)

### Phase 1: Quick wins
- Try different dtypes (BF16, FP16) — check hardware.json for support
- Switch attention to `flash_attention_2` (may need `uv add flash-attn`)
- Experiment with torch.compile modes (`default`, `reduce-overhead`, `max-autotune`)
- Enable static KV cache

### Phase 2: Quantization
- INT8 weight-only quantization via `torchao`
- INT4 weight-only quantization via `torchao`
- FP8 quantization if hardware supports it (check `fp8_supported` in hardware.json)
- KV cache quantization (FP8 or INT8)
- Try `bitsandbytes` if torchao quantization doesn't work (`uv add bitsandbytes`)

### Phase 3: Custom generation
- Replace `model.generate()` with a manual decode loop (eliminate HuggingFace overhead)
- Implement CUDA graphs for the decode step
- Pre-allocate KV cache buffers
- Optimize memory access patterns

### Phase 4: Advanced
- Speculative decoding with a smaller draft model
- Combine quantization + compile + CUDA graphs
- Explore `torchao` INT4 with 2:4 sparsity
- Profile-guided optimization (`uv run prepare.py --profile`)
- Try vLLM or SGLang backends if compatible

## Decision rules

- **PRIMARY metric**: tok_s (higher is better)
- **KEEP** if tok_s improves by any amount
- **DISCARD** if tok_s stays same or decreases
- **DISCARD** if peak_vram_gb exceeds the limit in config.json
- **CRASH** if benchmark fails or produces invalid output (>20% invalid outputs)
- **TIE-BREAKER**: prefer lower peak_vram_gb, then lower ttft_ms
- **SIMPLICITY**: all else equal, simpler code wins
