#!/usr/bin/env bash
# Forward a running server on an OSC compute node to localhost. Run on your laptop.
# Usage: scripts/tunnel.sh <backend> [local_port]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; source "$ROOT/config.env"; set +a

backend="${1:?usage: tunnel.sh <backend> [local_port]}"
case "$backend" in
  ollama) default_port=11434 ;;
  *)      default_port=8000 ;;
esac
local_port="${2:-$default_port}"
login="$OSC_USER@$OSC_CLUSTER.osc.edu"

ep="$(ssh "$login" cat ".bv_llm_host/$backend.endpoint" 2>/dev/null)" \
  || { echo "no running $backend server (no ~/.bv_llm_host/$backend.endpoint on $OSC_CLUSTER)" >&2; exit 1; }
node="$(sed -n 's/^NODE=//p' <<< "$ep")"
port="$(sed -n 's/^PORT=//p' <<< "$ep")"
job="$(sed -n 's/^JOB_ID=//p' <<< "$ep")"

echo "localhost:$local_port -> $node:$port (job $job). Ctrl-C to close."
# OSC's start helpers bind 0.0.0.0, so the login node can reach node:port
# directly; no SSH into the compute node needed.
exec ssh -N -o ExitOnForwardFailure=yes -L "$local_port:$node:$port" "$login"
