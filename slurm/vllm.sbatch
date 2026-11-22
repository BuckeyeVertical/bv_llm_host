#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Submitted by scripts/serve.sh, which sets account, time, GPUs, CPUs, name and log.

set -eo pipefail
source "$REPO_ROOT/slurm/common.sh"
load_secrets
# vllm requires VLLM_API_KEY on /v1/* when it is set. Only public jobs need a key.
[[ "$EXPOSE" == funnel ]] || unset VLLM_API_KEY

# OSC's module where the cluster has one (Ascend, Cardinal), else your own venv (Pitzer).
if module is-avail "vllm/$VLLM_VERSION" 2>/dev/null; then
  module load "vllm/$VLLM_VERSION"
else
  : "${VLLM_VENV:?no vllm/$VLLM_VERSION module on this cluster; set VLLM_VENV (docs/notes.md)}"
  source "$VLLM_VENV/bin/activate"
fi

# Weights: OSC's shared cache if it has the model, else HF_CACHE (downloaded on first use).
if [[ -d "$OSC_HF_HOME/hub/models--${MODEL//\//--}" ]]; then
  export HF_HOME="$OSC_HF_HOME"
else
  mkdir -p "$HF_CACHE"
  export HF_HOME="$HF_CACHE"
fi

# ARGS is parsed like a shell command line, so quoted JSON values work.
eval "args=($ARGS)"
# One model across all the job's GPUs, unless the profile chose its own split.
ngpus="$(nvidia-smi -L | wc -l)"
if (( ngpus > 1 )) && [[ " $ARGS " != *" --tensor-parallel-size"* && " $ARGS " != *" -tp "* ]]; then
  args+=(--tensor-parallel-size "$ngpus")
fi
# Pre-Ampere GPUs (V100 = 7.0) have no bf16, and the Triton kernel behind chunked
# prefill and prefix caching fails to compile on them.
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
if [[ "${cap%%.*}" -lt 8 ]]; then
  [[ "$ARGS" == *--dtype* ]] || args+=(--dtype half)
  args+=(--no-enable-chunked-prefill --no-enable-prefix-caching)
fi

echo "model:  $MODEL"
echo "vllm:   $(command -v vllm)  HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "+ vllm serve $MODEL ${args[*]}"

port="$(free_port)"
vllm serve "$MODEL" --host 0.0.0.0 --port "$port" "${args[@]}" &
pid=$!
# Generous: the first start may download the weights.
wait_http "http://localhost:$port/health" 3600 "$pid"

take_over_from "${PREV_JOB_ID:-}"
publish_endpoint "$port"
if [[ "$EXPOSE" == funnel ]]; then
  : "${VLLM_API_KEY:?refusing to expose publicly without VLLM_API_KEY in $SECRETS_FILE}"
  expose_funnel "$port"
fi
serve_forever "kill $pid"
