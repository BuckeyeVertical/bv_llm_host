# Setting up the server for an event

What the person running the server does before an event, so it is already up when people
arrive. Every command below is complete, with the event's model and flags; background on
the flags is in [running_vllm.md](running_vllm.md). What to send users is in
[connect.md](connect.md).

Event configuration: `Qwen/Qwen3.6-35B-A3B-FP8` on one Ascend A100 80 GB, public at
`https://bv-llm.tail43a32e.ts.net/v1`, API key required.

All OSC commands run from the repo checkout on Ascend:

```bash
ssh <osc_user>@ascend.osc.edu
cd /fs/scratch/PAS3560/bv_llm_host && git pull
```

## Pick how it will run

| Option | When it starts | Downtime during the event | Use when |
|---|---|---|---|
| **A. 7-day `quad` job** | Whenever the queue gets to it (was 1.5-12 days) | None | You can submit about a week ahead |
| **B. `debug-quad` chain** | Within minutes | ~5-8 min every hour (handoffs) | A didn't start in time, or a short notice event |

Submit A about a week ahead and keep B ready as the backup.

### A. 7-day `quad` job

```bash
EXPOSE=funnel AUTO_RESUBMIT=0 JOB_TIME=7-00:00:00 \
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B-FP8 \
VLLM_HF_HOME=/fs/scratch/PAS3560/allenthomas/hf \
VLLM_ARGS="--max-model-len 32768 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching" \
GPUS=1 scripts/submit.sh vllm --partition=quad
```

`quad`'s limit is 7 days, so the job covers the event only if it starts within the 7 days
before the event ends. Timing:

- **Submit ~7-8 days before the event.** The queue has taken 1.5-12 days, so it builds
  priority while waiting. A job that starts too early expires before the event.
- Optionally add `--begin=<event end minus 7 days>` (e.g. `--begin=2026-10-28T00:00:00` for
  an event ending Nov 3) so it can't start too early. The catch: Slurm doesn't count the job
  as waiting until that time, so it builds no priority before then.
- Check it daily:

  ```bash
  squeue -u $USER --start
  ```

  If it started so early that it ends before the event does, cancel it (`scancel <jobid>`)
  and resubmit. If it still hasn't started the evening before the event, plan on B.

It bills from the moment it starts: ~$2.75/day, ~$19 for the full 7 days.

### B. `debug-quad` chain

```bash
EXPOSE=funnel AUTO_RESUBMIT=1 JOB_TIME=01:00:00 \
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B-FP8 \
VLLM_HF_HOME=/fs/scratch/PAS3560/allenthomas/hf \
VLLM_ARGS="--max-model-len 32768 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching" \
GPUS=1 scripts/submit.sh vllm --partition=debug-quad
```

Start it ~1 hour before the event. Debug jobs are capped at 1 hour, so each job queues the
next one before it ends and the URL stays the same, with a few minutes of downtime per hour.
The handoff needs both of your Ascend debug slots: don't run anything else on Ascend debug
during the event. It runs until you stop it (see [After](#after)).

### Backup model: gpt-oss-20b

If Qwen3.6 won't start, run the same option with gpt-oss-20b (in OSC's central cache, so
`VLLM_HF_HOME` is empty). Users must switch the model name to `openai/gpt-oss-20b`.

```bash
# A (7-day quad job)
EXPOSE=funnel AUTO_RESUBMIT=0 JOB_TIME=7-00:00:00 \
VLLM_MODEL=openai/gpt-oss-20b VLLM_HF_HOME= \
VLLM_ARGS="--max-model-len 32768 --enable-auto-tool-choice --tool-call-parser openai" \
GPUS=1 scripts/submit.sh vllm --partition=quad

# B (debug-quad chain)
EXPOSE=funnel AUTO_RESUBMIT=1 JOB_TIME=01:00:00 \
VLLM_MODEL=openai/gpt-oss-20b VLLM_HF_HOME= \
VLLM_ARGS="--max-model-len 32768 --enable-auto-tool-choice --tool-call-parser openai" \
GPUS=1 scripts/submit.sh vllm --partition=debug-quad
```

## A week before

```bash
ls /fs/scratch/PAS3560/allenthomas/hf/hub | grep Qwen3.6       # model weights still there
ls /fs/scratch/PAS3560/tools/tailscale/tailscaled               # Tailscale binaries
test -s ~/.bv_llm_host/secrets.env && echo secrets ok           # TS_AUTHKEY + VLLM_API_KEY
OSCusage -P PAS3560 2026-01-01 2026-12-31                       # "Remaining Budget" covers it
```

`/fs/scratch` deletes long-unused files; rebuild anything missing (see
[One-time setup](running_vllm.md#one-time-setup)). In the Tailscale admin console, check that
machine `bv-llm` has key expiry disabled.

Then submit option A.

## The day before

1. **Dry run** (skip if option A's job is already running and passes step 2). A 30-minute
   public test on the debug queue, without chaining:

   ```bash
   EXPOSE=funnel AUTO_RESUBMIT=0 JOB_TIME=00:30:00 \
   VLLM_MODEL=Qwen/Qwen3.6-35B-A3B-FP8 \
   VLLM_HF_HOME=/fs/scratch/PAS3560/allenthomas/hf \
   VLLM_ARGS="--max-model-len 32768 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching" \
   GPUS=1 scripts/submit.sh vllm --partition=debug-quad
   ```

   Wait for `public:` in `logs/vllm-<jobid>.out`, run step 2, then `scancel <jobid>`. This
   catches missing weights, a broken setup or an expired key while there is time to fix it.
   If option A's job is already serving, the dry run waits privately and never gets the URL,
   so skip the dry run in that case. If A starts while the dry run holds the URL, A serves
   privately until you cancel the dry run, then takes the URL within ~1-2 min.
2. **Check it from outside**, from any laptop (no Tailscale needed):

   ```bash
   curl -s https://bv-llm.tail43a32e.ts.net/v1/models -H "Authorization: Bearer $KEY"
   ```

   It should list `Qwen/Qwen3.6-35B-A3B-FP8`. For a full check, run
   `scripts/agent_eval.py` from bv_llm_sdk against the URL (see
   [From bv_llm_sdk](running_vllm.md#from-bv_llm_sdk)).
3. **Send users** [connect.md](connect.md) and the API key. Send the key privately (DM or
   the team's private channel), never in a repo or public post.

## Day of

- **Before people arrive:** check the job and the URL:

  ```bash
  squeue -u $USER
  curl -s https://bv-llm.tail43a32e.ts.net/v1/models -H "Authorization: Bearer $KEY"
  ```

  If option A isn't running, start option B right away. It needs ~5 min to load, plus up to
  ~3 min before the public URL answers.
- **During:** `tail -f logs/vllm-<jobid>.out` shows requests. If the URL stops answering for
  more than ~10 min, check `squeue`; if no job is running, start option B.
- **If Qwen3.6 can't start:** use the gpt-oss-20b command for the same option (above) and
  tell users to change the model name to `openai/gpt-oss-20b`.

## After

```bash
scancel -u $USER -n vllm      # stops the job and any queued successor
```

A job left running bills until walltime (option A: up to 7 days). If the key went to many
people, rotate it afterwards: new `VLLM_API_KEY` in `~/.bv_llm_host/secrets.env`.
