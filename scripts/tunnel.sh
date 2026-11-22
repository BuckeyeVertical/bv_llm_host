#!/usr/bin/env bash
# Forward a server running on an OSC compute node to localhost. Run on your laptop.
# Usage: scripts/tunnel.sh [name] [local_port]
#   name may be omitted when only one server is running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; source "$ROOT/config.env"; set +a

name="${1:-}"
login="$OSC_USER@$OSC_CLUSTER.osc.edu"
# Resolved on OSC, so an empty STATE_DIR means the remote ~/.bv_llm_host.
dir="${STATE_DIR:-.bv_llm_host}/endpoints"
if [[ -n "$name" ]]; then
  ep="$(ssh "$login" "cat '$dir/$name'" 2>/dev/null)" || { echo "no server named $name is running" >&2; exit 1; }
else
  ep="$(ssh "$login" "f=('$dir'/*); [[ \${#f[@]} -eq 1 && -f \$f ]] && cat \"\$f\" || ls '$dir' >&2" 2>&1)" \
    && [[ "$ep" == *URL=* ]] || { echo "name one of the running servers: $ep" >&2; exit 1; }
fi
get() { sed -n "s/^$1=//p" <<< "$ep"; }

case "$(get BACKEND)" in
  ollama) default_port=11434 ;;
  *)      default_port=8000 ;;
esac
local_port="${2:-$default_port}"
# The job may run on another cluster; tunnel through that cluster's login node.
cluster="$(get CLUSTER)"
[[ -z "$cluster" || "$cluster" == "$OSC_CLUSTER" ]] || login="$OSC_USER@$cluster.osc.edu"

echo "http://localhost:$local_port/v1 -> $(get NAME) ($(get MODEL)) on $(get NODE), job $(get JOB_ID). Ctrl-C to close."
# Servers bind 0.0.0.0, so the login node reaches node:port directly.
exec ssh -N -o ExitOnForwardFailure=yes -L "$local_port:$(get NODE):$(get PORT)" "$login"
