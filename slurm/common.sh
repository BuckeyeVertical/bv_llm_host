# Shared helpers for server jobs. Sourced by slurm/*.sbatch.

STATE_DIR="$HOME/.bv_llm_host"
SECRETS_FILE="$STATE_DIR/secrets.env"   # TS_AUTHKEY, VLLM_API_KEY; chmod 600, never in git

# load_secrets: export whatever secrets.env defines (silently, so nothing hits the log).
load_secrets() {
  [[ -f "$SECRETS_FILE" ]] || return 0
  set -a; source "$SECRETS_FILE"; set +a
}

# publish_endpoint <backend> <port>
# Writes ~/.bv_llm_host/<backend>.endpoint so scripts/tunnel.sh can find the
# server, and removes it again when the job exits.
publish_endpoint() {
  mkdir -p "$STATE_DIR"
  ENDPOINT_FILE="$STATE_DIR/$1.endpoint"
  # CLUSTER lets tunnel.sh go through the right login node: home is shared
  # across OSC clusters, so this file is visible from all of them.
  printf 'NODE=%s\nPORT=%s\nJOB_ID=%s\nCLUSTER=%s\n' \
    "$(hostname -s)" "$2" "$SLURM_JOB_ID" "${SLURM_CLUSTER_NAME:-}" > "$ENDPOINT_FILE"
  echo "endpoint: ${SLURM_CLUSTER_NAME:-?} $(hostname -s):$2 -> $ENDPOINT_FILE"
}

# free_port: a random unused high port (same scheme as OSC's start helpers).
free_port() {
  local port
  while :; do
    port=$(shuf -i 49152-65535 -n 1)
    ss -tuln | grep -q ":$port " || { echo "$port"; return; }
  done
}

# wait_http <url> [timeout_s] [pid]
# Fails early if pid (the server process) exits before the url answers.
wait_http() {
  local url=$1 timeout=${2:-120} pid=$3
  for ((i = 0; i < timeout; i++)); do
    curl -sf "$url" > /dev/null && return 0
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "server process $pid exited" >&2
      return 1
    fi
    sleep 1
  done
  echo "timed out waiting for $url" >&2
  return 1
}

# expose_funnel <port>
# Publishes localhost:<port> at https://$TS_HOSTNAME.<tailnet>.ts.net via
# Tailscale Funnel. tailscaled runs in userspace mode (no root); its state
# persists in $STATE_DIR/tailscale, so every job comes back on the same URL.
expose_funnel() {
  local port=$1 dir="$STATE_DIR/tailscale"
  local sock="$dir/tailscaled.sock"
  : "${TS_AUTHKEY:?TS_AUTHKEY missing from $SECRETS_FILE}"
  claim_funnel || return 0
  mkdir -p "$dir"
  "$TAILSCALE_DIR/tailscaled" --tun=userspace-networking \
    --statedir="$dir" --socket="$sock" > "$dir/tailscaled.log" 2>&1 &
  TAILSCALED_PID=$!
  for _ in {1..30}; do [[ -S "$sock" ]] && break; sleep 1; done
  local ts=("$TAILSCALE_DIR/tailscale" --socket="$sock")
  # Pass the key by file so it doesn't show up in `ps` on a shared node.
  (umask 077; printf '%s' "$TS_AUTHKEY" > "$dir/authkey")
  "${ts[@]}" up --auth-key="file:$dir/authkey" --hostname="$TS_HOSTNAME"
  "${ts[@]}" funnel --bg "$port"
  PUBLIC_URL="https://$("${ts[@]}" status --json \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
  echo "public: $PUBLIC_URL/v1"
}

# claim_funnel
# Only one job may hold the Tailscale identity: two would fight over it (home,
# and so the state dir, is shared by all clusters). The holder refreshes
# $FUNNEL_OWNER every 30s; a fresh file held by anyone but our predecessor
# means stay private.
FUNNEL_OWNER="$STATE_DIR/funnel.owner"
claim_funnel() {
  if [[ -f "$FUNNEL_OWNER" ]]; then
    local holder age
    holder="$(cat "$FUNNEL_OWNER")"
    age=$(( $(date +%s) - $(stat -c %Y "$FUNNEL_OWNER") ))
    if (( age < 120 )) && [[ "$holder" != *":${PREV_JOB_ID:-none}" ]]; then
      echo "funnel held by $holder; staying private (endpoint/tunnel still work)" >&2
      return 1
    fi
  fi
  echo "${SLURM_CLUSTER_NAME:-?}:$SLURM_JOB_ID" > "$FUNNEL_OWNER"
  ( while :; do touch "$FUNNEL_OWNER"; sleep 30; done ) &
  HEARTBEAT_PID=$!
}

# take_over_from <job_id>
# A successor job calls this once its server is ready: stop the predecessor so
# it releases the shared Tailscale identity, then wait until it is gone.
take_over_from() {
  [[ -n "$1" ]] || return 0
  echo "taking over from job $1"
  scancel "$1" || true
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 2; done
}

# resubmit <this_job_id>
# Queues a successor with the same backend and sbatch args. Fired by
# --signal=B:USR1 shortly before walltime when AUTO_RESUBMIT=1.
resubmit() {
  echo "near walltime: submitting successor"
  # Drop this job's SLURM_* vars so they don't leak into the new job.
  # shellcheck disable=SC2046,SC2086  # intentional word-splitting
  ( unset $(compgen -e | grep '^SLURM_')
    PREV_JOB_ID=$1 "$REPO_ROOT/scripts/submit.sh" "$BACKEND" $SUBMIT_ARGS )
}

# release_funnel: stop tailscaled and give up the claim if it is still ours.
release_funnel() {
  [[ -n "$TAILSCALED_PID" ]] && kill "$TAILSCALED_PID" 2>/dev/null
  [[ -n "$HEARTBEAT_PID" ]] || return 0
  kill "$HEARTBEAT_PID" 2>/dev/null
  [[ "$(cat "$FUNNEL_OWNER" 2>/dev/null)" == *":$SLURM_JOB_ID" ]] && rm -f "$FUNNEL_OWNER"
  return 0
}

# serve_forever <stop_cmd>
# Blocks until the job is cancelled or hits walltime, then runs stop_cmd.
serve_forever() {
  trap "$1; release_funnel; rm -f \"\$ENDPOINT_FILE\"" EXIT
  trap 'exit 0' TERM INT
  trap "resubmit $SLURM_JOB_ID" USR1
  echo "serving until walltime or scancel $SLURM_JOB_ID"
  sleep infinity &
  local sleeper=$!
  # A trapped signal (USR1) interrupts `wait`; keep waiting until the job ends.
  while kill -0 "$sleeper" 2>/dev/null; do wait "$sleeper" || true; done
}
