# bv_llm_host

Host LLMs and vision models on OSC GPU nodes behind an OpenAI-compatible API, with
[vLLM](https://docs.vllm.ai) or [Ollama](https://ollama.com). Pick a model, submit one job,
get a URL.

```bash
scripts/serve.sh qwen3-8b            # start a server (a Slurm job)
scripts/url.sh qwen3-8b --wait       # -> http://a0123.ten.osc.edu:51234/v1 once it is ready
scripts/smoke_test.sh qwen3-8b       # one test request
scripts/stop.sh qwen3-8b
```

## Setup (once per checkout)

On an OSC login node:

```bash
git clone https://github.com/BuckeyeVertical/bv_llm_host.git && cd bv_llm_host
cp config.env.example config.env     # set OSC_USER, OSC_ACCOUNT, HF_CACHE, OLLAMA_CACHE
```

Use the same checkout from a laptop only for `scripts/tunnel.sh`. It needs just
`OSC_USER`, `OSC_CLUSTER` and, if you set one, `STATE_DIR`.

## Picking a model

`scripts/serve.sh <model>` takes any of:

| `<model>` | Example | Runs |
|---|---|---|
| a profile in [`models/`](models) | `qwen3-8b` | the profile's backend, model and flags |
| a Hugging Face id | `Qwen/Qwen2.5-7B-Instruct` | vLLM, default flags |
| `ollama:<tag>` | `ollama:qwen3:14b` | Ollama |

A profile is a few lines of shell variables:

```bash
# models/qwen3-8b.env
BACKEND=vllm                 # vllm | ollama
MODEL=Qwen/Qwen3-8B          # Hugging Face id or local path (vLLM), tag (Ollama)
ARGS="--max-model-len 40960 --reasoning-parser qwen3"   # extra `vllm serve` flags
# Optional: GPUS, CPUS, JOB_TIME, PARTITION, NAME, and OLLAMA_* server settings for Ollama
```

To add a model, copy a profile and change it. Settings apply in this order, last wins:
`config.env`, the profile, your environment. So one-off changes need no edits:

```bash
GPUS=2 JOB_TIME=04:00:00 scripts/serve.sh qwen3-coder-30b
scripts/serve.sh gemma-3-27b --partition=quad --mail-type=BEGIN   # extra args go to sbatch
```

Several GPUs: set `GPUS=n` and vLLM splits the model across them
(`--tensor-parallel-size n`) unless `ARGS` says otherwise. CPUs default to 8 per GPU;
set `CPUS` to change that.

**Vision models** need no extra setup. Send images as OpenAI `image_url` content parts
(`scripts/smoke_test.sh <server> --image` checks this). vLLM's `--limit-mm-per-prompt`
caps images per request; see `models/gemma-3-27b.env`.

**Weights.** Models already in OSC's shared stores (`/fs/ess/pub_data/vllm/.cache/huggingface`,
`/fs/ess/pub_data/ollama/models`) are used from there. Anything else is downloaded to
`HF_CACHE` or `OLLAMA_CACHE` on first start. To download big models ahead of time without
using GPU hours: `scripts/download.sh <model>`.

## Using a server

Each server lists itself in `$STATE_DIR/endpoints/<name>` once it is ready.

| Command | Does |
|---|---|
| `scripts/status.sh` | every server of the project: name, model, job, up / loading / pending, URL |
| `scripts/url.sh <name> [--wait]` | the base URL, e.g. for a harness: `BASE_URL=$(scripts/url.sh qwen3-8b --wait)` |
| `scripts/smoke_test.sh <name or url> [--image]` | list models and send one chat request |
| `scripts/stop.sh <name>` / `--all` | cancel the job (and any queued successor) |
| `tail -f logs/<name>-<jobid>.out` | the server's log |

**From OSC** (login nodes, other jobs): use the URL as is. Any OpenAI client works:

```python
from openai import OpenAI
client = OpenAI(base_url=BASE_URL, api_key="none")
model = client.models.list().data[0].id
reply = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Hi"}])
```

**From a laptop:** `scripts/tunnel.sh <name>` forwards it to `http://localhost:8000/v1`
(Ollama: 11434).

**From anywhere, without an OSC account:** see [docs/public-access.md](docs/public-access.md)
(vLLM only, needs an API key).

**Sharing servers with a team:** set `STATE_DIR` in `config.env` to a group-writable
directory in your project space. Everyone using that directory sees the same servers in
`status.sh`, `url.sh` and `tunnel.sh`.

Jobs bill GPU time to `OSC_ACCOUNT` until they end, so stop servers you are done with.

## Layout

```
config.env.example   settings for this checkout (copy to config.env)
models/*.env         model profiles
scripts/             commands you run (serve, url, status, stop, tunnel, smoke_test, download)
slurm/               the job scripts: one per backend, plus shared helpers
docs/notes.md        clusters, GPUs, vLLM flags, troubleshooting
docs/public-access.md  public HTTPS URL through Tailscale Funnel
```

Adding a backend means one new `slurm/<backend>.sbatch`. It starts the server on a free
port, waits until it answers, then calls `publish_endpoint <port>` and
`serve_forever <stop_cmd>` from `slurm/common.sh`.
