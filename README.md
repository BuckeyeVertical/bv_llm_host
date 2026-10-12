# bv_llm_host

Run LLM inference servers (Ollama, later vLLM) on OSC GPU nodes and reach them
from your laptop over an SSH tunnel. Thin wrappers around OSC's own modules:
[Ollama](https://www.osc.edu/resources/available_software/software_list/ollama),
[vLLM](https://www.osc.edu/resources/available_software/software_list/vllm).

```
config.env.example   per-user settings (copy to config.env)
slurm/common.sh      shared job helpers: publish endpoint, wait, serve until killed
slurm/<backend>.sbatch
scripts/submit.sh    [OSC]    sbatch a backend with your account/time/GPUs
scripts/tunnel.sh    [laptop] forward the running server to localhost
scripts/smoke_test.sh         list models + one chat completion
```

A running job writes `~/.bv_llm_host/<backend>.endpoint` (node, port, job id)
on OSC; `tunnel.sh` reads it. Adding a backend = one new `slurm/<name>.sbatch`
that starts the server, calls `publish_endpoint <name> <port>` and `serve_forever <stop_cmd>`.

## Setup

On OSC (login node) and on your laptop:

```bash
git clone <this repo> && cd bv_llm_host
cp config.env.example config.env   # fill in OSC_USER, OSC_ACCOUNT, OSC_CLUSTER
```

## Ollama

```bash
# OSC
scripts/submit.sh ollama
squeue -u $USER
tail -f logs/ollama-<jobid>.out      # shows OLLAMA_MODELS, GPU, available models

# laptop
scripts/tunnel.sh ollama             # localhost:11434 -> compute node
scripts/smoke_test.sh llama3.2:3b    # in another terminal

# OSC, when done
scancel <jobid>
```

By default the job uses OSC's central read-only model store. To pull your own
models, set `OLLAMA_MODELS_DIR` (project storage, not home) and `OLLAMA_MODEL`.

## vLLM

Same flow, backend `vllm`, local port 8000:

```bash
scripts/submit.sh vllm                 # OSC; ~3-5 min until the endpoint line appears
scripts/tunnel.sh vllm                 # laptop: localhost:8000 -> compute node
scripts/smoke_test.sh Qwen/Qwen3-8B http://localhost:8000/v1
```

Models come from OSC's central Hugging Face cache unless `VLLM_HF_HOME` is set.
Where OSC has no vllm module (Pitzer), the job activates `VLLM_VENV` instead, a venv
you build once with `uv venv` + `uv pip install vllm==<ver>` (pin `transformers` to
the 4.x of that vllm's era; uv otherwise pulls 5.x, which breaks older vllm). Pick a vllm built for
CUDA 12 for V100s; `--dtype half` and no chunked prefill / prefix caching are set automatically on
pre-Ampere GPUs.
Env vars override config.env per submit: `VLLM_MODEL=Qwen/Qwen3-0.6B scripts/submit.sh vllm`.
Multi-GPU: `scripts/submit.sh vllm --gpus-per-node=2` with
`VLLM_ARGS="--tensor-parallel-size 2 --disable_custom_all_reduce"`.

## Public access (no OSC account needed)

```bash
EXPOSE=funnel scripts/submit.sh vllm                    # OSC
EXPOSE=funnel AUTO_RESUBMIT=1 scripts/submit.sh vllm    # keep it up across walltimes
```

The job joins Tailscale (userspace, no root) and publishes vLLM via Funnel at
`https://$TS_HOSTNAME.<tailnet>.ts.net/v1`. Clients need only that URL and the API key;
no Tailscale or OSC login. Requires `~/.bv_llm_host/secrets.env` on OSC (chmod 600)
with `TS_AUTHKEY` and `VLLM_API_KEY`, and the static Tailscale binaries in `TAILSCALE_DIR`.

- Only one job holds the URL at a time (`~/.bv_llm_host/funnel.owner`, refreshed every 30s);
  other jobs stay private. With `AUTO_RESUBMIT=1` the successor takes over once loaded.
- Ollama is never exposed: it has no auth.
- A brand-new Funnel name can take several minutes before public DNS and its TLS cert work.

Tip: `tunnel.sh` makes two SSH hops; an `ssh` ControlMaster for
`*.osc.edu` saves re-authenticating.
