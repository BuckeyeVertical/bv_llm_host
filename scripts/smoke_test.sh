#!/usr/bin/env bash
# Hit an OpenAI-compatible endpoint: list models, then one chat completion.
# Usage: scripts/smoke_test.sh <model> [base_url]
#   e.g. scripts/smoke_test.sh llama3.2:3b http://localhost:11434/v1
# Set OPENAI_API_KEY when the server requires a key (vllm with VLLM_API_KEY).
set -euo pipefail
model="${1:?usage: smoke_test.sh <model> [base_url]}"
base="${2:-http://localhost:11434/v1}"
auth=(-H "Authorization: Bearer ${OPENAI_API_KEY:-none}")

echo "== GET $base/models"
curl -sf "${auth[@]}" "$base/models"; echo

echo "== POST $base/chat/completions ($model)"
curl -sf "${auth[@]}" "$base/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 32}"
echo
