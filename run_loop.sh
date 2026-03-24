#!/usr/bin/env bash
# run_loop.sh — Launch the autonomous inference optimization agent.
#
# Usage:
#   ./run_loop.sh                         # default model (Qwen 2.5 7B)
#   ./run_loop.sh --model "Qwen/Qwen2.5-0.5B"
#
# Tip: Run inside tmux so the agent survives SSH disconnects:
#   tmux new -s autoresearch
#   ./run_loop.sh

set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

MODEL_ARG="${1:-}"

echo "========================================"
echo "  autoresearch-inference agent loop"
echo "========================================"
echo "Started: $(date)"
echo ""

# Step 1: Setup (GPU discovery, model download, baseline benchmark)
echo "--- Setup & Baseline ---"
if [ -n "$MODEL_ARG" ]; then
    uv run prepare.py "$MODEL_ARG"
else
    uv run prepare.py
fi

# Step 2: Initial profile
echo ""
echo "--- Initial Profile ---"
if [ -n "$MODEL_ARG" ]; then
    uv run prepare.py "$MODEL_ARG" --profile
else
    uv run prepare.py --profile
fi

# Step 3: Run agent
echo ""
echo "--- Agent Loop ---"
PROMPT=$(cat program.md)
claude --allowedTools "Bash,Read,Write,Edit,Glob,Grep" -p "$PROMPT"
AGENT_EXIT=$?

# Step 4: Auto-append session summary to LEARNINGS.md
echo ""
echo "--- Session Summary ---"
SESSION_DATE=$(date +"%Y-%m-%d %H:%M")
SESSION_TAG=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# Extract best tok/s from results.tsv if it exists
BEST_TOKS="—"
EXPERIMENT_COUNT="—"
if [ -f results.tsv ]; then
    EXPERIMENT_COUNT=$(tail -n +2 results.tsv | wc -l | tr -d ' ')
    BEST_TOKS=$(tail -n +2 results.tsv | awk -F'\t' '$5=="keep" {print $2}' | sort -n | tail -1)
    [ -z "$BEST_TOKS" ] && BEST_TOKS="—"
fi

# Append to LEARNINGS.md session log
SESSION_ROW="| ${SESSION_DATE} | ${SESSION_TAG} | ${EXPERIMENT_COUNT} | ${BEST_TOKS} | (see results.tsv) |"
python3 - <<PYEOF
import re, os
path = "LEARNINGS.md"
with open(path) as f:
    content = f.read()

# Replace the placeholder row or append
placeholder = "| —    | —       | —           | —          | —           |"
new_row = "${SESSION_ROW}"
if placeholder in content:
    content = content.replace(placeholder, new_row, 1)
else:
    # Append after the last table row
    content = content.rstrip() + "\n" + new_row + "\n"

with open(path, "w") as f:
    f.write(content)
print(f"Updated LEARNINGS.md session log")
PYEOF

# Step 5: Generate plots
echo ""
echo "--- Generating Plots ---"
if [ -f results.tsv ] && [ "$(tail -n +2 results.tsv | wc -l)" -gt "0" ]; then
    uv run analyze.py
else
    echo "No results yet — skipping plots."
fi

# Step 6: Final commit
echo ""
echo "--- Final Commit ---"
git add LEARNINGS.md results.tsv 2>/dev/null || true
if [ -d plots ]; then
    git add plots/ 2>/dev/null || true
fi
git diff --cached --quiet || git commit -m "session end: ${SESSION_TAG} — best ${BEST_TOKS} tok/s"

echo ""
echo "========================================"
echo "  Session complete: $(date)"
echo "  Branch: ${SESSION_TAG}"
echo "  Experiments: ${EXPERIMENT_COUNT}"
echo "  Best tok/s: ${BEST_TOKS}"
echo "========================================"

exit $AGENT_EXIT
