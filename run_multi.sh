#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="$HOME/autoresearch-inference"
cd "$REPO"

run_model() {
  local MODEL_ID="$1"
  local MODEL_SLUG="$2"
  local CACHE_PATH="$HOME/.cache/autoresearch-inference/$MODEL_SLUG/model"

  # Wait for model weights to be available
  echo "Waiting for $MODEL_ID weights..."
  while [ ! -f "$CACHE_PATH/model.safetensors" ] && [ ! -f "$CACHE_PATH/model-00001-of-00002.safetensors" ] && [ ! -f "$CACHE_PATH/model-00001-of-00004.safetensors" ]; do
    sleep 10 && printf "."
  done
  echo " $MODEL_ID weights ready!"

  for RUN in r1 r2; do
    BRANCH="autoresearch/$MODEL_SLUG-$RUN"
    echo ""
    echo "============================================"
    echo "  MODEL: $MODEL_ID  |  RUN: $RUN"
    echo "  BRANCH: $BRANCH"
    echo "============================================"

    git checkout main
    git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
    git push -u origin "$BRANCH" 2>/dev/null || true

    # Carry over LEARNINGS.md
    git show autoresearch/mar17-r2:LEARNINGS.md > LEARNINGS.md 2>/dev/null || true

    # Write config.json
    python3 -c "
import json
cfg = {
  'model_id': '$MODEL_ID',
  'model_path': '$CACHE_PATH',
  'device': 'cuda:0',
  'max_new_tokens': 256,
  'vram_limit_gb': 6.9
}
json.dump(cfg, open('config.json','w'), indent=2)
print('config -> $MODEL_ID')
"
    # Run prepare.py to establish baseline first
    echo "Running prepare.py..."
    uv run prepare.py --model "$MODEL_ID" 2>&1 | tee "$REPO/prepare-${MODEL_SLUG}-${RUN}.log"

    # Fresh results (baseline already added by prepare.py)
    # Run agent loop
    echo "Running agent loop..."
    bash "$REPO/run_loop.sh" 2>&1 | tee "$REPO/agent-${MODEL_SLUG}-${RUN}.log"
    echo "Done: $BRANCH"
  done
}

run_model "Qwen/Qwen2.5-1.5B" "qwen-qwen2-5-1-5b"
run_model "Qwen/Qwen2.5-3B" "qwen-qwen2-5-3b"

echo ""
echo "All runs complete!"
