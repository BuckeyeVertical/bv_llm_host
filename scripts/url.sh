#!/usr/bin/env bash
# Print a running server's OpenAI-compatible base URL. Run on OSC.
#
# Usage: scripts/url.sh [name] [--wait]
#   --wait   block until the server is ready (fails if its job ends first)
# In a harness:  BASE_URL="$(scripts/url.sh qwen3-8b --wait)"
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

name= wait=
for a in "$@"; do
  case "$a" in
    --wait) wait=1 ;;
    *) name=$a ;;
  esac
done

if [[ -n "$wait" ]]; then
  : "${name:?--wait needs a server name}"
  until [[ -f "$STATE_DIR/endpoints/$name" ]]; do
    if [[ -z "$(squeue -h -n "llm-$name" -o %i 2>/dev/null)" ]]; then
      echo "no job named llm-$name is queued or running on this cluster" >&2
      exit 1
    fi
    sleep 10
  done
fi

file="$(endpoint_file "$name")"
[[ -f "$file" ]] || { echo "$(basename "$file") is not running (scripts/status.sh lists servers)" >&2; exit 1; }
source "$file"
echo "$URL"
