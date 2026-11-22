#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Submitted by scripts/serve.sh, which sets account, time, GPUs, CPUs, name and log.

set -eo pipefail
source "$REPO_ROOT/slurm/common.sh"

# Ollama runs from OSC's Apptainer image: the module's, or a copy (OLLAMA_SIF) on
# clusters without the module (Pitzer).
if module load "ollama/$OLLAMA_VERSION" 2>/dev/null; then
  img="$OLLAMA_IMG"
else
  img="${OLLAMA_SIF:?no ollama/$OLLAMA_VERSION module on this cluster; set OLLAMA_SIF (docs/notes.md)}"
fi

# Models live in OLLAMA_CACHE. If OSC's shared store has this tag, link it in instead of
# downloading: newer Ollama versions rewrite their store on startup, so they can't use
# the read-only shared store directly.
models="$OLLAMA_CACHE"
mkdir -p "$models/blobs"
repo="${MODEL%%:*}"; tag="${MODEL#*:}"
[[ "$MODEL" == *:* ]] || tag=latest
[[ "$repo" == */* ]] || repo="library/$repo"
shared="$OSC_OLLAMA_MODELS/manifests/registry.ollama.ai/$repo/$tag"
pull=1
if [[ -f "$shared" ]]; then
  mkdir -p "$models/manifests/registry.ollama.ai/$repo"
  cp "$shared" "$models/manifests/registry.ollama.ai/$repo/$tag"
  for digest in $(grep -o 'sha256:[0-9a-f]*' "$shared" | sort -u); do
    ln -sfn "$OSC_OLLAMA_MODELS/blobs/${digest/:/-}" "$models/blobs/${digest/:/-}"
  done
  pull=
fi

# Pass the profile's OLLAMA_* server settings (e.g. OLLAMA_CONTEXT_LENGTH) into the
# container; --cleanenv drops everything else. Keep the model loaded between requests.
env="OLLAMA_MODELS=$models,OLLAMA_FLASH_ATTENTION=1,OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:--1}"
for v in $(compgen -e | grep '^OLLAMA_'); do
  case "$v" in
    OLLAMA_VERSION|OLLAMA_SIF|OLLAMA_CACHE|OLLAMA_IMG|OLLAMA_MODELS|OLLAMA_HOST|OLLAMA_KEEP_ALIVE) ;;
    *) env+=",$v=${!v}" ;;
  esac
done

echo "model:  $MODEL  (OLLAMA_MODELS=$models)"
echo "image:  $img"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

port="$(free_port)"
apptainer run --nv --cleanenv --env "$env,OLLAMA_HOST=0.0.0.0:$port" "$img" &
pid=$!
wait_http "http://localhost:$port/api/tags" 300 "$pid"

if [[ -n "$pull" ]]; then
  apptainer exec --cleanenv --env "OLLAMA_HOST=localhost:$port" "$img" ollama pull "$MODEL"
fi
# Load the model onto the GPU now, so the first real request isn't slow.
curl -sf "http://localhost:$port/api/generate" -d "{\"model\": \"$MODEL\"}" > /dev/null

publish_endpoint "$port"
serve_forever "kill $pid"
