#!/usr/bin/env bash
# List model servers: ready ones (with URL) and jobs still queued or loading. Run on OSC.
# Usage: scripts/status.sh
set -euo pipefail
source "$(dirname "$0")/lib.sh"
load_config

fmt='%-24s %-40s %-8s %-10s %s\n'
printf "$fmt" NAME MODEL JOB STATE URL
ready_jobs=" "
for f in "$STATE_DIR"/endpoints/*; do
  [[ -f "$f" ]] || continue
  ( source "$f"
    state=up
    curl -sf -m 5 -o /dev/null -H "Authorization: Bearer ${OPENAI_API_KEY:-none}" "$URL/models" || state=no-answer
    printf "$fmt" "$NAME" "$MODEL" "$JOB_ID" "$state" "$URL${PUBLIC_URL:+  $PUBLIC_URL/v1}" )
  ready_jobs+="$(sed -n 's/^JOB_ID=//p' "$f") "
done

# Jobs of this project that have no endpoint yet: queued, or loading the model.
squeue -h -A "$OSC_ACCOUNT" -o '%i|%j|%T|%u' 2>/dev/null | while IFS='|' read -r id job state user; do
  [[ "$job" == llm-* && "$ready_jobs" != *" $id "* ]] || continue
  [[ "$state" == RUNNING ]] && state=loading
  printf "$fmt" "${job#llm-}" "($user)" "$id" "${state,,}" -
done
