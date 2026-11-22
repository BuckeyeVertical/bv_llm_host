#!/usr/bin/env bash
# Start a model server as a Slurm job. Run on an OSC login node.
#
# Usage: scripts/serve.sh <model> [sbatch args...]
#   <model>  a profile in models/        scripts/serve.sh qwen3-8b
#            a Hugging Face id (vLLM)    scripts/serve.sh Qwen/Qwen2.5-7B-Instruct
#            ollama:<tag> (Ollama)       scripts/serve.sh ollama:llama3.2:3b
#
# Any setting can be overridden from the environment, and sbatch args pass through:
#   GPUS=2 JOB_TIME=04:00:00 scripts/serve.sh qwen3-coder-30b --partition=quad
set -euo pipefail
source "$(dirname "$0")/lib.sh"

spec="${1:?usage: scripts/serve.sh <profile | hf-id | ollama:tag> [sbatch args...]}"
shift
resolve_model "$spec"

job="$ROOT/slurm/$BACKEND.sbatch"
[[ -f "$job" ]] || { echo "unknown backend '$BACKEND' (no $job)" >&2; exit 1; }
if [[ "$BACKEND" == ollama && "${EXPOSE:-none}" == funnel ]]; then
  echo "refusing EXPOSE=funnel for Ollama: it has no API key" >&2
  exit 1
fi

opts=(
  --job-name="llm-$NAME"
  --account="$OSC_ACCOUNT"
  --time="$JOB_TIME"
  --gpus-per-node="$GPUS"
  --output="$ROOT/logs/$NAME-%j.out"
  --export=ALL
)
[[ -z "${PARTITION:-}" ]] || opts+=(--partition="$PARTITION")
[[ "${AUTO_RESUBMIT:-0}" != 1 ]] || opts+=(--signal=B:USR1@600)

# The job (and any successor it resubmits) needs these.
export BACKEND MODEL NAME STATE_DIR ARGS="${ARGS:-}" REPO_ROOT="$ROOT" SERVE_SPEC="$spec" SBATCH_EXTRA="$*"

mkdir -p "$ROOT/logs"
jobid="$(sbatch --parsable "${opts[@]}" "$@" "$job")"
jobid="${jobid%%;*}"
echo "job $jobid: $NAME ($BACKEND, $MODEL)"
echo "  log:   tail -f $ROOT/logs/$NAME-$jobid.out"
echo "  url:   scripts/url.sh $NAME --wait"
echo "  stop:  scripts/stop.sh $NAME"
