#!/usr/bin/env bash
# Submit a model-server job. Run on an OSC login node from the repo checkout.
# Usage: scripts/submit.sh <backend> [extra sbatch args...]
#   e.g. scripts/submit.sh ollama --time=04:00:00 --gpus-per-node=2
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; source "$ROOT/config.env"; set +a

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
