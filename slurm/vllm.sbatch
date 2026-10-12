#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Account, time, GPUs and log path are passed by scripts/submit.sh.

set -eo pipefail
source "$REPO_ROOT/slurm/common.sh"

# OSC's vllm module where it exists (Cardinal, Ascend); otherwise our own venv (Pitzer).
if module is-avail "vllm/$VLLM_VERSION" 2>/dev/null; then
  module load "vllm/$VLLM_VERSION"
  use_module=1
else
  : "${VLLM_VENV:?no vllm/$VLLM_VERSION module here; set VLLM_VENV}"
  source "$VLLM_VENV/bin/activate"
  export HF_HOME="${HF_HOME:-$OSC_HF_HOME}"
fi

if [[ -n "$VLLM_HF_HOME" ]]; then
  mkdir -p "$VLLM_HF_HOME"
  export HF_HOME="$VLLM_HF_HOME"
fi
echo "vllm: $(command -v vllm)  HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv || true

# Pre-Ampere GPUs (V100 = 7.0) have no bf16.
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
if [[ "${cap%%.*}" -lt 8 && "$VLLM_ARGS" != *--dtype* ]]; then
  VLLM_ARGS="--dtype half $VLLM_ARGS"
fi

# shellcheck disable=SC2086  # VLLM_ARGS is intentionally word-split
if [[ -n "$use_module" ]]; then
  # `vllm_start` is an interactive alias for this; aliases don't expand in scripts.
  # The helper backgrounds `vllm serve` and sleeps 180s before returning.
  source vllm_start_helper "$VLLM_MODEL" $VLLM_ARGS
  : "${VLLM_API_PORT:?vllm failed to start}"
  pid=$VLLM_PID
  stop_cmd=vllm_stop
else
  VLLM_API_PORT="$(free_port)"
  vllm serve "$VLLM_MODEL" --host 0.0.0.0 --port "$VLLM_API_PORT" $VLLM_ARGS &
  pid=$!
  stop_cmd="kill $pid"
fi
wait_http "http://localhost:$VLLM_API_PORT/v1/models" 900 "$pid"

echo "available models:"
curl -s "http://localhost:$VLLM_API_PORT/v1/models"; echo

publish_endpoint vllm "$VLLM_API_PORT"
serve_forever "$stop_cmd"
