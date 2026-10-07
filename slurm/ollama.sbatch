#!/bin/bash
#SBATCH --job-name=ollama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Account, time, GPUs and log path are passed by scripts/submit.sh.

set -eo pipefail
source "$REPO_ROOT/slurm/common.sh"

module load "ollama/$OLLAMA_VERSION"

if [[ -n "$OLLAMA_MODELS_DIR" ]]; then
  mkdir -p "$OLLAMA_MODELS_DIR"
  export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
fi
echo "OLLAMA_MODELS=${OLLAMA_MODELS:-<module default>}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

ollama_start
: "${OLLAMA_PORT:?ollama_start did not set OLLAMA_PORT}"
wait_http "http://localhost:$OLLAMA_PORT/api/tags"

if [[ -n "$OLLAMA_MODELS_DIR" && -n "$OLLAMA_MODEL" ]]; then
  ollama_pull "$OLLAMA_MODEL"
fi

echo "available models:"
curl -s "http://localhost:$OLLAMA_PORT/api/tags"; echo

publish_endpoint ollama "$OLLAMA_PORT"
serve_forever ollama_stop
