#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Account, time, GPUs and log path are passed by scripts/submit.sh.

set -eo pipefail
source "$REPO_ROOT/slurm/common.sh"

module load "vllm/$VLLM_VERSION"

if [[ -n "$VLLM_HF_HOME" ]]; then
  mkdir -p "$VLLM_HF_HOME"
  export HF_HOME="$VLLM_HF_HOME"
fi
echo "HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# `vllm_start` is an interactive alias for this; aliases don't expand in scripts.
# The helper backgrounds `vllm serve` and sleeps 180s before returning.
# shellcheck disable=SC2086  # VLLM_ARGS is intentionally word-split
source vllm_start_helper "$VLLM_MODEL" $VLLM_ARGS
: "${VLLM_API_PORT:?vllm failed to start}"
wait_http "http://localhost:$VLLM_API_PORT/v1/models" 600

echo "available models:"
curl -s "http://localhost:$VLLM_API_PORT/v1/models"; echo

publish_endpoint vllm "$VLLM_API_PORT"
serve_forever vllm_stop
