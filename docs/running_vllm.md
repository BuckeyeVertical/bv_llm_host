# Running the vLLM server

How to bring the shared vLLM server up on OSC and use it. One-time setup is already
done (see [One-time setup](#one-time-setup) if something is missing).

- Preparing for an event (what to do the week and day before): [event_setup.md](event_setup.md)
- Handout for people using the server: [connect.md](connect.md)

## What we serve

| | |
|---|---|
| Model | `Qwen/Qwen3.6-35B-A3B-FP8` (official FP8, ~35 GB) |
| GPU | one A100 80 GB: Ascend `debug-quad` (debug) or `quad` |
| vLLM | OSC module 0.23.0 (Ascend) |
| Weights | `/fs/scratch/PAS3560/allenthomas/hf` (`VLLM_HF_HOME`), already downloaded |
| Backup model | `openai/gpt-oss-20b` (OSC's central cache), same GPU |

Chosen from the bv_llm_sdk agent evals (`bv_llm_sdk/evals/RESULTS.md`):
- **Accuracy:** 32/32 on the 16 drone-agent scenarios on vLLM, and 46/46 on Ollama.
- **Load:** one A100 handled 20 simultaneous agent sessions with no errors.

Use an **80 GB** A100. On a 40 GB card (Ascend `debug-nextgen`, `nextgen`) the weights
leave almost no room for KV cache. Pitzer can't run it: its vLLM 0.10 venv predates
the Qwen3.5/3.6 architecture, and V100s have no FP8.

## Start it (on OSC)

```bash
ssh <osc_user>@ascend.osc.edu
cd /fs/scratch/PAS3560/bv_llm_host && git pull
EXPOSE=funnel AUTO_RESUBMIT=1 JOB_TIME=01:00:00 \
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B-FP8 \
VLLM_HF_HOME=/fs/scratch/PAS3560/allenthomas/hf \
VLLM_ARGS="--max-model-len 32768 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching" \
GPUS=1 scripts/submit.sh vllm --partition=debug-quad
```

`AUTO_RESUBMIT=1` keeps the server up past the debug queue's 1-hour limit by chaining jobs
on the same URL (see [Keeping it up](#keeping-it-up-on-the-1-hour-debug-queues)). It runs
until you stop it with `scancel -u $USER -n vllm`.

Every flag matters:

| Flag | Why |
|---|---|
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Turns the model's tool calls into real `tool_calls`. Without it they come back as plain text and the agent can't fly. |
| `--reasoning-parser qwen3` | Strips the model's `<think>` reasoning out of the reply. |
| `--enable-prefix-caching` | **Essential.** Qwen3.6's hybrid (Mamba-style) layers make vLLM turn prompt caching off by default, so every call reprocessed the agent's ~9k-token prompt (cache hit rate 0%). With the flag the hit rate was 92%, and calls with 20 users were 2-3× faster. |
| `--max-model-len 32768` | Plenty for an agent conversation; the model's 262k default wastes memory. |

Watch it come up:

```bash
squeue -u $USER
tail -f logs/vllm-<jobid>.out
```

Ready when the log prints `public: https://bv-llm.tail43a32e.ts.net/v1` (in the
2026-10-12 rehearsal the URL answered the moment that line appeared, ~5 min after the job started). Expect ~3-5 min of
loading and warm-up on Ascend, then up to ~3 min before Tailscale's public relay starts
forwarding. The log's `GPU KV cache size` line should say about 1.5M tokens.

### Backup: gpt-oss-20b

Use it if Qwen3.6 is unavailable, or if far more than ~20 people will be using it at once. It
was the fastest under load (about 2 s per call with 20 users, nothing queued) but a little
less reliable (27/31).

```bash
EXPOSE=funnel AUTO_RESUBMIT=1 JOB_TIME=01:00:00 \
VLLM_MODEL=openai/gpt-oss-20b VLLM_HF_HOME= \
VLLM_ARGS="--max-model-len 32768 --enable-auto-tool-choice --tool-call-parser openai" \
GPUS=1 scripts/submit.sh vllm --partition=debug-quad
```

### Keeping it up on the 1-hour debug queues

The debug queues (Cardinal `debug` H100, Ascend `debug-nextgen`/`debug-quad` A100, Pitzer
`gpudebug` V100) start fast but cap jobs at 1 h, 1 GPU and 2 running jobs per user.
`AUTO_RESUBMIT=1` chains jobs: 10 min before walltime a job queues its successor, which loads
its model while the old one keeps serving, then cancels it and takes over the same URL.

The start commands above already set it. To stop the chain (the running job and any
queued successor):

```bash
scancel -u $USER -n vllm
```

Tested on Cardinal `debug`: a handoff costs ~1-2 min of public downtime when the successor
starts before the old job's walltime. **On Ascend `debug-quad` (2026-10-12 rehearsal) the
successor stayed queued until the old job's walltime**, waiting for that same GPU because no
other was free. So each hourly handoff there means ~5-8 min of downtime while the new job
loads the model. If it waits in the queue past that, the URL is down
until it starts and loads. Keep both debug slots free for the handoff, and remember the chain
bills until you cancel it (~$2.75/day per GPU). For an event, a regular `quad` job booked a
day or two ahead avoids handoffs entirely. Regular A100/H100 queues were 1.5-12 days out
when we checked.

Avoid test jobs on Ascend debug while the public chain runs: they take one of the two debug
slots the handoff needs. Test jobs elsewhere are fine with `EXPOSE=none`; they can't take the
public URL. (`~/.bv_llm_host/vllm.endpoint` is shared, so a test job's endpoint replaces the
server's for `tunnel.sh` until it exits.)

## How many people it handles

From `bv_llm_sdk/scripts/load_test.py` on one A100 80 GB. N simulated users each fire
real multi-step agent requests back to back (full prompt and 43 tools; tool results instant).
Seconds are typical (p50) / slowest 5% (p95).

| Users at once | Model call | Whole request | Peak KV-cache use | Queued requests |
|---|---|---|---|---|
| 1 | 0.7 / 2.8 | 2.8 / 8.4 | 0.7% | 0 |
| 10 | 2.6 / 12.1 | 11.5 / 22.1 | 3.4% | 2 |
| 20 | 3.4 / 27.5 | 14.8 / 225 | 5.9% | 8 |

- **Memory is not the limit.** The GPU holds ~1.5M tokens of KV cache, about 45 full
  32k-token conversations at once, and the tests used under 6% of it. More users means slower
  replies, not failures.
- **Real people are lighter than the test.** Most of a real request is the drone flying, and
  people stop to read and type. 20 always-busy sessions is roughly 40-60 active people
  (an estimate from the timings, not measured).
- **Rough guide:**
  - Up to ~20 teams flying at once: comfortable.
  - 20-40: works, with some 30-60 s replies at busy moments.
  - Beyond that: run a second job on another GPU and split people between the two URLs, or
    switch to gpt-oss-20b.
- **The slow tail** (one request in 20 taking minutes at 20 users) is Qwen3.6 reasoning at
  length. Turning thinking off per request removes most of it but cost one eval failure
  (it turned left when told right), so we keep it on.

## The URL and the key

| Setting | Value |
|---|---|
| Base URL | `https://bv-llm.tail43a32e.ts.net/v1` |
| API key | `VLLM_API_KEY` in `~/.bv_llm_host/secrets.env` on OSC (public jobs only) |
| Model | `Qwen/Qwen3.6-35B-A3B-FP8` (whatever the current job serves; check `/v1/models`) |

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
  -d '{"model": "Qwen/Qwen3.6-35B-A3B-FP8", "messages": [{"role": "user", "content": "What is a drone?"}],
       "max_tokens": 1000, "chat_template_kwargs": {"enable_thinking": false}}'
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
| Empty `content` | Qwen3.6 spent `max_tokens` thinking. Raise `max_tokens`, or send `"chat_template_kwargs": {"enable_thinking": false}` (Qwen3.6 has no `/no_think`) |

### From bv_llm_sdk

The agent's `openai` backend works as-is. Configure it with environment variables (or the
matching `bv-agent` flags):

```bash
export BV_AGENT_BACKEND=openai
export BV_AGENT_BASE_URL=https://bv-llm.tail43a32e.ts.net/v1
export BV_AGENT_MODEL=Qwen/Qwen3.6-35B-A3B-FP8   # must match the served model
export BV_AGENT_VISION=true                      # Qwen3.6 sees images: lets the agent use `look`
export OPENAI_API_KEY=<VLLM_API_KEY>             # or BV_AGENT_API_KEY
bv-agent
```

Leave `BV_AGENT_THINK` unset (thinking on) for the best results. For gpt-oss-20b, set
`BV_AGENT_MODEL=openai/gpt-oss-20b` and drop `BV_AGENT_VISION`. To check a server
end to end, `scripts/agent_eval.py` in bv_llm_sdk runs the 16 drone scenarios against it, and
`scripts/load_test.py` the many-users test; both take the same flags.

In Python, `AgentConfig.from_env()` reads the same `BV_AGENT_*` variables.

## Stop it

```bash
scancel <jobid>
```

GPU time bills to PAS3560 (~$0.09/GPU-hour + $0.003/core-hour) until walltime. Check spend with
`OSCusage -P PAS3560 <start> <end>`.

## Private use (no public URL)

Submit without `EXPOSE=funnel` (or set `EXPOSE=none`), then tunnel from a laptop that has
OSC SSH access. Private jobs need no API key (anyone on OSC's internal network could reach
them, which is accepted):

```bash
scripts/tunnel.sh vllm                                   # laptop: localhost:8000 -> job
scripts/smoke_test.sh Qwen/Qwen3.6-35B-A3B-FP8 http://localhost:8000/v1
```

Always pass `EXPOSE=none` explicitly for test jobs, so a test can't take the public
identity if `config.env` was left set to `funnel`.

## One-time setup

Already done for PAS3560; only redo what's missing.

| Piece | Where | Notes |
|---|---|---|
| `config.env` | repo checkout on OSC | copy `config.env.example`; keep `EXPOSE=none` (checkout is shared) |
| Secrets | `~/.bv_llm_host/secrets.env` on OSC, chmod 600 | `TS_AUTHKEY=tskey-auth-...`, `VLLM_API_KEY=<openssl rand -hex 24>` |
| Tailscale binaries | `TAILSCALE_DIR` | static tarball from pkgs.tailscale.com |
| Model weights | `/fs/scratch/PAS3560/allenthomas/hf` | `snapshot_download("Qwen/Qwen3.6-35B-A3B-FP8")` with `HF_HOME` set there, in a CPU sbatch job (~1 min) |
| vLLM venv (Pitzer) | `VLLM_VENV` | `uv venv` + `uv pip install vllm==0.10.1.1 "transformers>=4.55,<4.56"` in a CPU sbatch job; too old for Qwen3.6 |
| Tailscale account | admin console | MagicDNS, HTTPS, Funnel on; disable key expiry on machine `bv-llm` |

Notes:
- `/fs/scratch` is purged of long-unused files. If the weights, venv or Tailscale binaries
  vanish, rebuild them, and check the weights the week before an event.
- To rotate the API key: edit `secrets.env`, restart the job, tell the team.
- Only one job holds the public URL at a time. Another public job serves privately until the URL is released, then takes it within ~1-2 min.
