# Helpers for slurm/*.sbatch. scripts/serve.sh exports NAME, BACKEND, MODEL, ARGS,
# STATE_DIR, REPO_ROOT, SERVE_SPEC and SBATCH_EXTRA into the job.

PRIVATE_DIR="$HOME/.bv_llm_host"          # per user: secrets.env, Tailscale state
SECRETS_FILE="$PRIVATE_DIR/secrets.env"   # HF_TOKEN, TS_AUTHKEY, VLLM_API_KEY; chmod 600, never in git
ENDPOINT_FILE="$STATE_DIR/endpoints/$NAME"
# OSC's shared, read-only model stores. Models found there aren't downloaded again.
OSC_HF_HOME=/fs/ess/pub_data/vllm/.cache/huggingface
OSC_OLLAMA_MODELS=/fs/ess/pub_data/ollama/models

source "$REPO_ROOT/slurm/funnel.sh"

# Caches and the endpoint dir may be shared by a group: keep what we create group-writable.
umask 002

# load_secrets: export whatever secrets.env defines, without echoing it to the log.
load_secrets() {
  [[ -f "$SECRETS_FILE" ]] || return 0
  set -a; source "$SECRETS_FILE"; set +a
}

# free_port: a random unused high port.
free_port() {
  local port
  while :; do
    port=$(shuf -i 49152-65535 -n 1)
    ss -tuln | grep -q ":$port " || { echo "$port"; return; }
  done
}

# wait_http <url> <timeout_s> <pid>
# Waits for url to answer; fails early if the server process exits.
wait_http() {
  local url=$1 timeout=$2 pid=$3
  for ((i = 0; i < timeout; i += 5)); do
    curl -sf -o /dev/null "$url" && return 0
    kill -0 "$pid" 2>/dev/null || { echo "server exited before it was ready" >&2; return 1; }
    sleep 5
  done
  echo "timed out after ${timeout}s waiting for $url" >&2
  return 1
}

# publish_endpoint <port>
# Lists this server in $STATE_DIR/endpoints/<name>, which url.sh, status.sh and
# tunnel.sh read. Written only once the server is ready.
publish_endpoint() {
  local node tmp
  node="$(hostname -s)"
  mkdir -p "$(dirname "$ENDPOINT_FILE")"
  tmp="$ENDPOINT_FILE.$SLURM_JOB_ID"
  cat > "$tmp" <<EOF
NAME=$NAME
BACKEND=$BACKEND
MODEL=$MODEL
URL=http://$node:$1/v1
NODE=$node
PORT=$1
JOB_ID=$SLURM_JOB_ID
CLUSTER=${SLURM_CLUSTER_NAME:-}
USER=$USER
EOF
  chmod 664 "$tmp"
  mv "$tmp" "$ENDPOINT_FILE"
  echo "ready: http://$node:$1/v1  ($ENDPOINT_FILE)"
}

# release_endpoint: remove the endpoint file unless a newer job has replaced it.
release_endpoint() {
  grep -qx "JOB_ID=$SLURM_JOB_ID" "$ENDPOINT_FILE" 2>/dev/null && rm -f "$ENDPOINT_FILE"
  return 0
}

# serve_forever <stop_cmd>
# Blocks until scancel or walltime, then cleans up. With AUTO_RESUBMIT=1, serve.sh
# asks Slurm for USR1 10 min before walltime, which queues a successor job.
serve_forever() {
  trap "$1; release_funnel; release_endpoint" EXIT
  trap 'exit 0' TERM INT
  trap "resubmit $SLURM_JOB_ID" USR1
  echo "serving until walltime or scancel $SLURM_JOB_ID"
  sleep infinity &
  local sleeper=$!
  # A trapped signal interrupts `wait`; keep waiting until the job really ends.
  while kill -0 "$sleeper" 2>/dev/null; do wait "$sleeper" || true; done
}

# resubmit <this_job_id>: queue a successor with the same model and sbatch args.
resubmit() {
  echo "near walltime: submitting successor"
  # Drop this job's SLURM_* vars so they don't leak into the new job.
  # shellcheck disable=SC2046,SC2086  # intentional word-splitting
  ( unset $(compgen -e | grep '^SLURM_')
    PREV_JOB_ID=$1 "$REPO_ROOT/scripts/serve.sh" "$SERVE_SPEC" $SBATCH_EXTRA )
}

# take_over_from <job_id>
# A successor calls this once its server is ready: stop the predecessor and wait
# until it is gone, so the two never hold the endpoint or public URL at once.
take_over_from() {
  [[ -n "$1" ]] || return 0
  echo "taking over from job $1"
  scancel "$1" || true
  while squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 2; done
}
