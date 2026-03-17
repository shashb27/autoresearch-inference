#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="$HOME/autoresearch-inference"
cd "$REPO"

run_model() {
  local MODEL_ID="$1"
  local MODEL_SLUG="$2"

  for RUN in r1 r2; do
    BRANCH="autoresearch/$MODEL_SLUG-$RUN"
    REMOTE_LOG=$(git log --oneline "origin/$BRANCH" 2>/dev/null | wc -l)
    if [ "$REMOTE_LOG" -gt 2 ]; then
      echo "Skipping $BRANCH — already has $REMOTE_LOG commits"
      continue
    fi

    echo "============================================"
    echo "  MODEL: $MODEL_ID  |  RUN: $RUN | BRANCH: $BRANCH"
    echo "============================================"

    git checkout main && git pull origin main --ff-only
    git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
    git push -u origin "$BRANCH" 2>/dev/null || true
    git show autoresearch/mar17-r2:LEARNINGS.md > LEARNINGS.md 2>/dev/null || true

    echo "Running prepare (baseline + profile)..."
    uv run prepare.py --model "$MODEL_ID" 2>&1 | tee "$REPO/prepare-${MODEL_SLUG}-${RUN}.log"
    uv run prepare.py --model "$MODEL_ID" --profile 2>&1 | tee -a "$REPO/prepare-${MODEL_SLUG}-${RUN}.log"

    echo "Running agent loop..."
    bash "$REPO/run_loop.sh" 2>&1 | tee "$REPO/agent-${MODEL_SLUG}-${RUN}.log"
    echo "Done: $BRANCH"
  done
}

echo "Waiting for 3B R1 (tmux: autoresearch) to finish..."
while tmux has-session -t autoresearch 2>/dev/null; do sleep 30; done
echo "3B R1 done. Starting remaining runs..."

run_model "Qwen/Qwen2.5-3B" "qwen-qwen2-5-3b"
run_model "Qwen/Qwen2.5-1.5B" "qwen-qwen2-5-1-5b"

echo "All runs complete!"
