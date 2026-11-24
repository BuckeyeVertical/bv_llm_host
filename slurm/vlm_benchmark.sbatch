#!/bin/bash
#SBATCH --job-name=vlm_benchmark
#SBATCH --account=PAS3560
#SBATCH --partition=quad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Benchmark the VLMs in scripts/vlm_models.json on one Ascend A100 80 GB (quad).
# Run from the repo root on an Ascend login node; needs ROBOFLOW_API_KEY in
# ~/.bv_llm_host/secrets.env. Extra words go to vlm_benchmark.py:
#   mkdir -p logs
#   VLM_DATASET=workspace/project/1 sbatch slurm/vlm_benchmark.sbatch
#   VLM_DATASET=... sbatch slurm/vlm_benchmark.sbatch --limit 50 --models Qwen/Qwen3-VL-8B-Instruct
#   VLM_DATASET=... sbatch --gpus-per-node=2 slurm/vlm_benchmark.sbatch   # for InternVL3.5-38B
# Resubmit with the same dataset to continue after walltime.

set -eo pipefail
REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
overrides="$(export -p)"
set -a; source "$REPO_ROOT/config.env"; set +a
eval "$overrides"   # submit-time env beats config.env
source "$REPO_ROOT/slurm/common.sh"
load_secrets
: "${VLM_DATASET:?set VLM_DATASET=workspace/project/version}"

# vlm_test.py uses the module's vllm and Python packages; nothing to install.
module load "vllm/${VLLM_VERSION:-0.23.0}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# exec: Slurm's SIGTERM reaches Python, which stops the model server.
exec python -u "$REPO_ROOT/scripts/vlm_benchmark.py" \
  --dataset "$VLM_DATASET" \
  --out "${VLM_OUT:-$REPO_ROOT/results/vlm/${VLM_DATASET//\//__}}" "$@"
