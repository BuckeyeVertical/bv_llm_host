#!/usr/bin/env bash
# Check a server end to end: list its models, then one chat completion.
#
# Usage: scripts/smoke_test.sh <base_url | name> [--image]
#   scripts/smoke_test.sh http://localhost:8000/v1       # laptop, through tunnel.sh
#   scripts/smoke_test.sh qwen3-8b                       # on OSC, by server name
#   --image  also send a small image (for vision models)
# Set OPENAI_API_KEY when the server requires a key (public vLLM jobs).
set -euo pipefail
target="${1:?usage: scripts/smoke_test.sh <base_url | name> [--image]}"
if [[ "$target" == http* ]]; then
  base="$target"
else
  base="$("$(dirname "$0")/url.sh" "$target")"
fi
auth=(-H "Authorization: Bearer ${OPENAI_API_KEY:-none}")

echo "== GET $base/models"
models="$(curl -sS --fail-with-body "${auth[@]}" "$base/models")"
echo "$models"
model="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' <<< "$models")"

if [[ "${2:-}" == --image ]]; then
  # 32x32 solid red PNG
  img='iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzCMP5/un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg=='
  content="[{\"type\": \"text\", \"text\": \"What color is this image? One word.\"},
            {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$img\"}}]"
else
  content='"Say hello in five words."'
fi

echo "== POST $base/chat/completions ($model)"
curl -sS --fail-with-body "${auth[@]}" "$base/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": $content}], \"max_tokens\": 512}"
echo
