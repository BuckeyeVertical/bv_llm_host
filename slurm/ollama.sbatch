#!/bin/bash
#SBATCH --job-name=ollama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Account, time, GPUs and log path are passed by scripts/submit.sh.

set -eo pipefail
source "$REPO_ROOT/slurm/common.sh"

# Clusters without the module (pitzer) run the same Apptainer image the
# module wraps, copied somewhere every cluster can read (OLLAMA_SIF).
if module load "ollama/$OLLAMA_VERSION" 2>/dev/null; then
  use_module=1
else
  use_module=
  : "${OLLAMA_SIF:?no ollama/$OLLAMA_VERSION module on this cluster; set OLLAMA_SIF in config.env (see README)}"
  export OLLAMA_MODELS="${OLLAMA_MODELS:-/fs/ess/pub_data/ollama/models}"
fi

if [[ -n "$OLLAMA_MODELS_DIR" ]]; then
  mkdir -p "$OLLAMA_MODELS_DIR"
  export OLLAMA_MODELS="$OLLAMA_MODELS_DIR"
fi
echo "OLLAMA_MODELS=${OLLAMA_MODELS:-<module default>}"
[[ -n "$use_module" ]] || echo "no module here; running $OLLAMA_SIF"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

if [[ -n "$use_module" ]]; then
  # `ollama_start` is an interactive alias for this; aliases don't expand in scripts.
  source ollama_start_helper
else
  # What ollama_start_helper does: a free port, then the image on the GPU.
  while :; do
    OLLAMA_PORT=$(shuf -i 49152-65535 -n 1)
    ss -ltn | grep -q ":$OLLAMA_PORT " || break
  done
  apptainer run --nv --cleanenv \
    --env "OLLAMA_MODELS=$OLLAMA_MODELS,OLLAMA_HOST=0.0.0.0:$OLLAMA_PORT,OLLAMA_FLASH_ATTENTION=1" \
    "$OLLAMA_SIF" &
  OLLAMA_PID=$!
fi
: "${OLLAMA_PORT:?ollama_start did not set OLLAMA_PORT}"
wait_http "http://localhost:$OLLAMA_PORT/api/tags" 120 "$OLLAMA_PID"

if [[ -n "$OLLAMA_MODELS_DIR" && -n "$OLLAMA_MODEL" ]]; then
  if [[ -n "$use_module" ]]; then
    ollama_pull "$OLLAMA_MODEL"
  else
    apptainer exec --cleanenv --env "OLLAMA_HOST=localhost:$OLLAMA_PORT" "$OLLAMA_SIF" \
      ollama pull "$OLLAMA_MODEL"
  fi
fi

echo "available models:"
curl -s "http://localhost:$OLLAMA_PORT/api/tags"; echo

publish_endpoint ollama "$OLLAMA_PORT"
if [[ -n "$use_module" ]]; then
  serve_forever ollama_stop
else
  serve_forever "kill $OLLAMA_PID"
fi
