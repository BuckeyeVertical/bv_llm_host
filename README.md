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

Tip: `tunnel.sh` makes two SSH hops; an `ssh` ControlMaster for
`*.osc.edu` saves re-authenticating.
