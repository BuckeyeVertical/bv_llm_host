# Public URL through Tailscale Funnel (EXPOSE=funnel). Sourced by common.sh.
# See docs/public-access.md.

FUNNEL_OWNER="$PRIVATE_DIR/funnel.owner"

# expose_funnel <port>
# Publishes localhost:<port> at https://$TS_HOSTNAME.<tailnet>.ts.net. tailscaled runs
# in userspace mode (no root); its state persists in $PRIVATE_DIR/tailscale, so every
# job comes back on the same URL.
expose_funnel() {
  local port=$1 dir="$PRIVATE_DIR/tailscale"
  # Socket on node-local disk: in the shared state dir, one job's tailscaled
  # removes it on exit and orphans any other job's daemon.
  local sock="${TMPDIR:-/tmp}/tailscaled-$SLURM_JOB_ID.sock"
  : "${TS_AUTHKEY:?TS_AUTHKEY missing from $SECRETS_FILE}"
  # If another job holds the URL, keep serving privately and take it once released.
  if ! claim_funnel; then
    echo "waiting for the public URL to be released" >&2
    until claim_funnel 2>/dev/null; do sleep 60; done
  fi
  mkdir -p "$dir"
  "$TAILSCALE_DIR/tailscaled" --tun=userspace-networking \
    --statedir="$dir" --socket="$sock" > "$dir/tailscaled.log" 2>&1 &
  TAILSCALED_PID=$!
  for _ in {1..30}; do [[ -S "$sock" ]] && break; sleep 1; done
  local ts=("$TAILSCALE_DIR/tailscale" --socket="$sock")
  # Pass the key by file so it doesn't show up in `ps` on a shared node.
  (umask 077; printf '%s' "$TS_AUTHKEY" > "$dir/authkey")
  "${ts[@]}" up --auth-key="file:$dir/authkey" --hostname="$TS_HOSTNAME"
  # The state dir restores the previous job's serve config (old port); the public
  # relay can stay stuck on it, so clear it before publishing the new port.
  "${ts[@]}" funnel reset
  sleep 2
  "${ts[@]}" funnel --bg "$port"
  PUBLIC_URL="https://$("${ts[@]}" status --json \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
  echo "PUBLIC_URL=$PUBLIC_URL" >> "$ENDPOINT_FILE"
  echo "public: $PUBLIC_URL/v1"
}

# claim_funnel
# Only one job may hold the Tailscale identity. The holder refreshes $FUNNEL_OWNER
# every 30s; a fresh file held by anyone but our predecessor means it is taken
# (stale after 2 min if its job died uncleanly).
claim_funnel() {
  if [[ -f "$FUNNEL_OWNER" ]]; then
    local holder age
    holder="$(cat "$FUNNEL_OWNER")"
    age=$(( $(date +%s) - $(stat -c %Y "$FUNNEL_OWNER") ))
    if (( age < 120 )) && [[ "$holder" != *":${PREV_JOB_ID:-none}" ]]; then
      echo "funnel held by $holder" >&2
      return 1
    fi
  fi
  echo "${SLURM_CLUSTER_NAME:-?}:$SLURM_JOB_ID" > "$FUNNEL_OWNER"
  ( while :; do touch "$FUNNEL_OWNER"; sleep 30; done ) &
  HEARTBEAT_PID=$!
}

# release_funnel: stop tailscaled and give up the claim if it is still ours.
release_funnel() {
  [[ -z "${TAILSCALED_PID:-}" ]] || kill "$TAILSCALED_PID" 2>/dev/null
  [[ -n "${HEARTBEAT_PID:-}" ]] || return 0
  kill "$HEARTBEAT_PID" 2>/dev/null
  [[ "$(cat "$FUNNEL_OWNER" 2>/dev/null)" == *":$SLURM_JOB_ID" ]] && rm -f "$FUNNEL_OWNER"
  return 0
}
