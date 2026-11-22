# Shared by scripts/*.sh. Not run directly.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Logs, caches and the endpoint dir may be shared by a group: keep them group-writable.
umask 002

# load_config [profile_file]
# Reads config.env, then the profile, then re-applies whatever the caller
# exported, so `GPUS=2 scripts/serve.sh ...` beats both files.
load_config() {
  local overrides
  overrides="$(export -p)"
  if [[ ! -f "$ROOT/config.env" ]]; then
    echo "missing $ROOT/config.env: cp config.env.example config.env and fill it in" >&2
    exit 1
  fi
  set -a
  source "$ROOT/config.env"
  [[ -z "${1:-}" ]] || source "$1"
  set +a
  eval "${overrides//declare -x /declare -gx }"
  STATE_DIR="${STATE_DIR:-$HOME/.bv_llm_host}"
}

# resolve_model <spec>
# Sets BACKEND, MODEL, NAME from a profile name, a Hugging Face id or ollama:<tag>.
resolve_model() {
  local spec=$1 profile="$ROOT/models/$1.env"
  if [[ -f "$profile" ]]; then
    load_config "$profile"
    NAME="${NAME:-$spec}"
  else
    load_config
    if [[ "$spec" == ollama:* ]]; then
      BACKEND=ollama MODEL="${spec#ollama:}"
    elif [[ "$spec" == */* ]]; then
      BACKEND=vllm MODEL="$spec"
    else
      echo "no profile models/$spec.env. Pass a profile, a Hugging Face id (org/model) or ollama:<tag>." >&2
      echo "profiles: $(list_profiles)" >&2
      exit 1
    fi
    # A short name for squeue, logs and the endpoint file: Qwen/Qwen3-8B -> qwen3-8b
    NAME="${NAME:-$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9.\n' '-')}"
  fi
  : "${BACKEND:?profile sets no BACKEND}" "${MODEL:?profile sets no MODEL}"
}

list_profiles() {
  local f out=()
  for f in "$ROOT"/models/*.env; do out+=("$(basename "$f" .env)"); done
  echo "${out[*]}"
}

# endpoint_file [name]
# Prints the endpoint file of a running server. With no name, the only one running.
endpoint_file() {
  local dir="$STATE_DIR/endpoints"
  if [[ -n "${1:-}" ]]; then
    echo "$dir/$1"
    return
  fi
  local files=("$dir"/*)
  if [[ ${#files[@]} -eq 1 && -f "${files[0]}" ]]; then
    echo "${files[0]}"
  else
    echo "name a server: $(ls "$dir" 2>/dev/null | tr '\n' ' ')" >&2
    return 1
  fi
}
