# Shared helpers for server jobs. Sourced by slurm/*.sbatch.

STATE_DIR="$HOME/.bv_llm_host"

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

# wait_http <url> [timeout_s]
wait_http() {
  local url=$1 timeout=${2:-120}
  for ((i = 0; i < timeout; i++)); do
    curl -sf "$url" > /dev/null && return 0
    sleep 1
  done
  echo "timed out waiting for $url" >&2
  return 1
}

# serve_forever <stop_cmd>
# Blocks until the job is cancelled or hits walltime, then runs stop_cmd.
serve_forever() {
  trap "$1; rm -f \"\$ENDPOINT_FILE\"" EXIT
  trap 'exit 0' TERM INT
  echo "serving until walltime or scancel $SLURM_JOB_ID"
  sleep infinity &
  wait $!
}
