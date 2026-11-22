#!/usr/bin/env bash
# Stop a model server: cancels every job with that name, including a queued successor
# from AUTO_RESUBMIT. Run on OSC.
# Usage: scripts/stop.sh <name> | --all
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

target="${1:?usage: scripts/stop.sh <name> | --all}"
if [[ "$target" == --all ]]; then
  ids="$(squeue -h -u "$USER" -o '%i %j' | awk '$2 ~ /^llm-/ {print $1}')"
else
  ids="$(squeue -h -u "$USER" -n "llm-$target" -o %i)"
fi
[[ -n "$ids" ]] || { echo "no jobs of yours to stop" >&2; exit 1; }
# shellcheck disable=SC2086
scancel $ids
echo "cancelled: $(echo $ids)"
