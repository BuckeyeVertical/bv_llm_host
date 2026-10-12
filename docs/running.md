# Running the LLM server

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

On Pitzer the GPU may be a 16 GB V100; for models over ~12 GB in fp16, request
`--gres=gpu:v100-32g:1`. Centrally cached models: `ls /fs/ess/pub_data/vllm/.cache/huggingface/hub`.

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

## Stop it

```bash
scancel <jobid>
```

GPU time bills to PAS3560 (~$0.09/GPU-hour + $0.003/core-hour) until walltime. Check spend with
`OSCusage -P PAS3560 <start> <end>`.

## Private use (no public URL)

Submit without `EXPOSE=funnel`, then tunnel from a laptop that has OSC SSH access:

```bash
scripts/tunnel.sh vllm                                   # laptop: localhost:8000 -> job
OPENAI_API_KEY=$KEY scripts/smoke_test.sh <model> http://localhost:8000/v1
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
