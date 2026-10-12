#!/usr/bin/env bash
# Submit a model-server job. Run on an OSC login node from the repo checkout.
# Usage: scripts/submit.sh <backend> [extra sbatch args...]
#   e.g. scripts/submit.sh ollama --time=04:00:00 --gpus-per-node=2
# Exported env vars override config.env, e.g. VLLM_MODEL=Qwen/Qwen3-0.6B scripts/submit.sh vllm
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
overrides="$(export -p)"
set -a; source "$ROOT/config.env"; set +a
eval "$overrides"

backend="${1:?usage: submit.sh <backend> [sbatch args...]}"; shift
job="$ROOT/slurm/$backend.sbatch"
[[ -f "$job" ]] || { echo "no job script: $job" >&2; exit 1; }

mkdir -p "$ROOT/logs"
sbatch \
  --account="$OSC_ACCOUNT" \
  --time="$JOB_TIME" \
  --gpus-per-node="$GPUS" \
  --output="$ROOT/logs/%x-%j.out" \
  --export=ALL,REPO_ROOT="$ROOT" \
  "$@" "$job"
