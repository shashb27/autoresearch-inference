#!/usr/bin/env bash
# run_loop.sh — Launch the autonomous inference optimization agent.
#
# Usage:
#   ./run_loop.sh                                  # use model already in config.json
#   ./run_loop.sh --model "org/model-name"         # download + use this model
#
# Tip: Run inside tmux so the session survives SSH disconnects:
#   tmux new -s autoresearch && ./run_loop.sh --model "org/model-name"

set -e
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"

# Parse --model flag
MODEL_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_FLAG="--model $2"; shift 2 ;;
        *)        echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  autoresearch-inference agent loop"
echo "========================================"
echo "Started: $(date)"
echo ""

# Step 1: Setup — GPU discovery, model download, architecture detection, baseline
echo "--- Setup & Baseline ---"
uv run prepare.py $MODEL_FLAG

# Step 2: Profile — separate call so baseline results are committed first
echo ""
echo "--- Initial Profile ---"
uv run prepare.py $MODEL_FLAG --profile

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
import os
path = "LEARNINGS.md"
if not os.path.exists(path):
    exit(0)
with open(path) as f:
    content = f.read()
placeholder = "| —    | —       | —           | —          | —           |"
new_row = "${SESSION_ROW}"
if placeholder in content:
    content = content.replace(placeholder, new_row, 1)
else:
    content = content.rstrip() + "\n" + new_row + "\n"
with open(path, "w") as f:
    f.write(content)
print("Updated LEARNINGS.md session log")
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
[ -d plots ] && git add plots/ 2>/dev/null || true
git diff --cached --quiet || git commit -m "session end: ${SESSION_TAG} — best ${BEST_TOKS} tok/s"

echo ""
echo "========================================"
echo "  Session complete: $(date)"
echo "  Branch:      ${SESSION_TAG}"
echo "  Experiments: ${EXPERIMENT_COUNT}"
echo "  Best tok/s:  ${BEST_TOKS}"
echo ""
echo "  Publish results:  uv run analyze.py --publish"
echo "========================================"

exit $AGENT_EXIT
