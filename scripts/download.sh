#!/usr/bin/env bash
# Pre-download a vLLM model's weights into HF_CACHE with a CPU job, so a GPU job doesn't
# spend its time downloading. Optional: vLLM downloads on first start anyway. Run on OSC.
#
# Usage: scripts/download.sh <profile | hf-id> [sbatch args...]
# Gated models (Llama, Gemma) need HF_TOKEN in ~/.bv_llm_host/secrets.env.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

resolve_model "${1:?usage: scripts/download.sh <profile | hf-id> [sbatch args...]}"
shift
[[ "$BACKEND" == vllm ]] || { echo "only vLLM models; Ollama pulls when the server starts" >&2; exit 1; }
osc_cache=/fs/ess/pub_data/vllm/.cache/huggingface
if [[ -d "$osc_cache/hub/models--${MODEL//\//--}" ]]; then
  echo "$MODEL is already in OSC's shared cache; nothing to download"
  exit 0
fi

mkdir -p "$ROOT/logs" "$HF_CACHE"
sbatch --account="$OSC_ACCOUNT" --job-name="dl-$NAME" --time=02:00:00 --cpus-per-task=4 \
  --output="$ROOT/logs/download-$NAME-%j.out" "$@" --wrap "
    set -e
    [ -f ~/.bv_llm_host/secrets.env ] && { set -a; . ~/.bv_llm_host/secrets.env; set +a; }
    module load vllm/$VLLM_VERSION
    HF_HOME='$HF_CACHE' hf download '$MODEL'"
echo "downloading $MODEL to $HF_CACHE (log: logs/download-$NAME-<jobid>.out)"
