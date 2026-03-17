#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="$HOME/autoresearch-inference"
cd "$REPO"

run_model() {
  local MODEL_ID="$1"
  local MODEL_SLUG="$2"
  local CACHE_PATH="$HOME/.cache/autoresearch-inference/$MODEL_SLUG/model"

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

    # Carry over LEARNINGS.md from 0.5B r2
    git show autoresearch/mar17-r2:LEARNINGS.md > LEARNINGS.md 2>/dev/null || true

    # Write config.json for this model
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
    # Fresh results
    printf 'commit\ttok_s\tttft_ms\tpeak_vram_gb\tstatus\tdescription\n' > results.tsv

    bash "$REPO/run_loop.sh" 2>&1 | tee "$REPO/agent-${MODEL_SLUG}-${RUN}.log"
    echo "Done: $BRANCH"
  done
}

echo "=== Waiting for 1.5B model ==="
while [ ! -f "$HOME/.cache/autoresearch-inference/qwen-qwen2-5-1-5b/model/config.json" ]; do
  sleep 10 && printf "."
done
echo " ready!"
run_model "Qwen/Qwen2.5-1.5B" "qwen-qwen2-5-1-5b"

echo "=== Waiting for 3B model ==="
while [ ! -f "$HOME/.cache/autoresearch-inference/qwen-qwen2-5-3b/model/config.json" ]; do
  sleep 10 && printf "."
done
echo " ready!"
run_model "Qwen/Qwen2.5-3B" "qwen-qwen2-5-3b"

echo ""
echo "All runs complete!"
