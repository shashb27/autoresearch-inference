#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:$PATH"
cd ~/autoresearch-inference

PROMPT=$(cat program.md)
echo "Starting autoresearch agent loop..."
claude --allowedTools "Bash,Read,Write,Edit,Glob,Grep" -p "$PROMPT"
