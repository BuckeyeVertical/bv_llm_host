# Running the vLLM server

How to bring the shared vLLM server up on OSC and use it. One-time setup is already
done (see [One-time setup](#one-time-setup) if something is missing).

## Start it (on OSC)

```bash
ssh <osc_user>@pitzer.osc.edu                # cluster preference: pitzer > ascend > cardinal
cd /fs/scratch/PAS3560/bv_llm_host && git pull
EXPOSE=funnel scripts/submit.sh vllm         # public; add AUTO_RESUBMIT=1 to keep it up past walltime
```

Watch it come up:

```bash
squeue -u $USER
tail -f logs/vllm-<jobid>.out
```

Ready when the log prints `public: https://bv-llm.tail43a32e.ts.net/v1`. Expect ~1-2 min of
model loading on Pitzer (3-5 on Ascend/Cardinal), then up to ~3 min before Tailscale's public
relay starts forwarding.

Per-submit overrides (anything in `config.env`):

```bash
EXPOSE=funnel VLLM_MODEL=Qwen/Qwen3-0.6B scripts/submit.sh vllm --time=04:00:00
```

### Keeping it up on the 1-hour debug queues

The debug queues (Cardinal `debug` H100, Ascend `debug-nextgen`/`debug-quad` A100, Pitzer
`gpudebug` V100) start fast but cap jobs at 1 h and 2 running jobs per user. `AUTO_RESUBMIT=1`
chains jobs: 10 min before walltime a job queues its successor, which loads its model while
the old one keeps serving, then cancels it and takes over the same URL.

```bash
JOB_TIME=01:00:00 EXPOSE=funnel AUTO_RESUBMIT=1 scripts/submit.sh vllm --partition=debug
scancel -u $USER -n vllm      # stop the chain (running job and any queued successor)
```

Tested on Cardinal `debug`: a handoff costs ~1-2 min of public downtime when the successor
starts before the old job's walltime. If it waits in the queue past that, the URL is down
until it starts and loads. Keep both debug slots free for the handoff, and remember the chain
bills until you cancel it (~$2.75/day per GPU).

On Pitzer the GPU may be a 16 GB V100; for models over ~12 GB in fp16, request
`--gres=gpu:v100-32g:1`. Centrally cached models: `ls /fs/ess/pub_data/vllm/.cache/huggingface/hub`.

For agents that call tools, enable vLLM's tool-call parser for the model (`hermes` for Qwen3,
`qwen3_coder` for Qwen3-Coder); without it tool calls come back as plain text:

```bash
EXPOSE=funnel VLLM_ARGS="--enable-auto-tool-choice --tool-call-parser hermes" scripts/submit.sh vllm
```

## The URL and the key

| Setting | Value |
|---|---|
| Base URL | `https://bv-llm.tail43a32e.ts.net/v1` |
| API key | `VLLM_API_KEY` in `~/.bv_llm_host/secrets.env` on OSC (public jobs only) |
| Model | whatever the current job serves; read it from `/v1/models` |

The URL stays the same across jobs: every job reuses the Tailscale identity saved in
`~/.bv_llm_host/tailscale` on OSC. It is also printed as the `public:` line in each job log and
shown under Machines → `bv-llm` in the Tailscale admin console. It only changes if:

- that state dir is deleted (the job registers anew; if the old `bv-llm` machine still exists
  in the console, the new one becomes `bv-llm-1`, so delete the old one first),
- `TS_HOSTNAME` in `config.env` changes, or
- the tailnet is renamed (the `tail43a32e` part).

If the machine's key expires, the URL stops working until it is re-authorized; disable key
expiry on `bv-llm` in the admin console to avoid that.

The key is the only thing keeping strangers out. Share it privately, never in a repo. Clients
pass it as an OpenAI-style bearer token, so OpenAI SDKs take it as `OPENAI_API_KEY`. It is not
an OpenAI account key, and nothing goes to OpenAI.

## Use it (from anywhere)

No OSC account or Tailscale needed. Get the API key from whoever runs the server.

```bash
curl https://bv-llm.tail43a32e.ts.net/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "What is a drone? /no_think"}], "max_tokens": 200}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="https://bv-llm.tail43a32e.ts.net/v1", api_key=KEY)
print(client.models.list().data[0].id)   # the model currently served
```

| Response | Meaning |
|---|---|
| `401 Unauthorized` | Server is up; key missing or wrong (a browser always gets this) |
| Can't resolve / connect | No job is serving, or it just started; check `squeue` |
| Empty `content` | Qwen3 spent `max_tokens` thinking; raise it or add `/no_think` |

### From bv_llm_sdk

The agent's `openai` backend works as-is; configure it with environment variables (or the
matching `bv-agent` flags):

```bash
export BV_AGENT_BACKEND=openai
export BV_AGENT_BASE_URL=https://bv-llm.tail43a32e.ts.net/v1
export BV_AGENT_MODEL=Qwen/Qwen3-8B      # must match the served model
export OPENAI_API_KEY=<VLLM_API_KEY>     # or BV_AGENT_API_KEY
bv-agent
```

In Python, `AgentConfig.from_env()` reads the same `BV_AGENT_*` variables.

## Stop it

```bash
scancel <jobid>
```

GPU time bills to PAS3560 (~$0.09/GPU-hour + $0.003/core-hour) until walltime. Check spend with
`OSCusage -P PAS3560 <start> <end>`.

## Private use (no public URL)

Submit without `EXPOSE=funnel`, then tunnel from a laptop that has OSC SSH access. Private
jobs need no API key (anyone on OSC's internal network could reach them, which is accepted):

```bash
scripts/tunnel.sh vllm                                   # laptop: localhost:8000 -> job
scripts/smoke_test.sh <model> http://localhost:8000/v1
```

## One-time setup

Already done for PAS3560; only redo what's missing.

| Piece | Where | Notes |
|---|---|---|
| `config.env` | repo checkout on OSC | copy `config.env.example`; keep `EXPOSE=none` (checkout is shared) |
| Secrets | `~/.bv_llm_host/secrets.env` on OSC, chmod 600 | `TS_AUTHKEY=tskey-auth-...`, `VLLM_API_KEY=<openssl rand -hex 24>` |
| Tailscale binaries | `TAILSCALE_DIR` | static tarball from pkgs.tailscale.com |
| vLLM venv (Pitzer) | `VLLM_VENV` | `uv venv` + `uv pip install vllm==0.10.1.1 "transformers>=4.55,<4.56"` in a CPU sbatch job |
| Tailscale account | admin console | MagicDNS, HTTPS, Funnel on; disable key expiry on machine `bv-llm` |

Notes:
- `/fs/scratch` is purged of long-unused files; if the venv or Tailscale binaries vanish, rebuild them.
- To rotate the API key: edit `secrets.env`, restart the job, tell the team.
- Only one job holds the public URL at a time; others stay private.
