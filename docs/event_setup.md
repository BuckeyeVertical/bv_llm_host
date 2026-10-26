# Setting up the server for an event

What the person running the server does before an event, so it is already up when people
arrive. Details of each command are in [running_vllm.md](running_vllm.md). What to send
users is in [connect.md](connect.md).

## Pick how it will run

| Option | When it starts | Downtime during the event | Use when |
|---|---|---|---|
| **A. Long `quad` job** | Whenever the queue gets to it (was 1.5-12 days) | None | You can submit 3+ days ahead |
| **B. OSC reservation** | The window OSC reserves for you | None | You can ask OSC 1-2 weeks ahead |
| **C. `debug-quad` chain** | Within minutes | ~5-8 min every hour (handoffs) | Last resort, or a short event |

A or B are better for an event. Keep C ready as the backup either way.

**A. Long `quad` job.** `quad` allows up to 7 days. Submit early with a walltime that still
covers the event if the job starts right away:

```bash
EXPOSE=funnel JOB_TIME=5-00:00:00 \
VLLM_MODEL=Qwen/Qwen3.6-35B-A3B-FP8 VLLM_HF_HOME=/fs/scratch/PAS3560/allenthomas/hf \
VLLM_ARGS="--max-model-len 32768 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching" \
GPUS=1 scripts/submit.sh vllm --partition=quad
```

It bills from the moment it starts (~$2.75/day; 5 days ≈ $14), so an early start costs a
little but is fine. Check `squeue -u $USER --start` daily. If it hasn't started by the
evening before, plan on C.

**B. Reservation.** Email oschelp@osc.edu 1-2 weeks ahead: one Ascend A100 80 GB node
(`quad`), project PAS3560, the event's dates and hours, for a student event. If they grant
it, add `--reservation=<name they give you>` and set `JOB_TIME` to the reservation's length.

**C. `debug-quad` chain.** The start command in [running_vllm.md](running_vllm.md#start-it-on-osc)
(`AUTO_RESUBMIT=1 ... --partition=debug-quad`). Start it ~1 hour before the event. Keep both of
your debug slots free; don't run anything else on Ascend debug during the event.

## A week before

Run on OSC (`ssh <osc_user>@ascend.osc.edu`):

```bash
ls /fs/scratch/PAS3560/allenthomas/hf/hub | grep Qwen3.6       # model weights still there
ls /fs/scratch/PAS3560/tools/tailscale/tailscaled               # Tailscale binaries
test -s ~/.bv_llm_host/secrets.env && echo secrets ok           # TS_AUTHKEY + VLLM_API_KEY
OSCusage -P PAS3560 2026-01-01 2026-12-31                       # "Remaining Budget" covers it
```

`/fs/scratch` deletes long-unused files; rebuild anything missing (see
[One-time setup](running_vllm.md#one-time-setup)). In the Tailscale admin console, check that
machine `bv-llm` has key expiry disabled.

Then submit option A, or email OSC for B.

## The day before

1. **Dry run** (skip if option A's job is already running and passing step 2). Start the
   server with the option C command, wait for `public:` in the log, run step 2, then
   `scancel -u $USER -n vllm`. This catches a broken venv, missing weights or an expired key
   while there is still time to fix it.
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

- **Before people arrive:** check `squeue -u $USER` on Ascend and run the curl check
  above. If option A/B isn't running, start option C right away. It needs ~5 min to load,
  plus up to ~3 min before the public URL answers.
- **During:** `tail -f logs/vllm-<jobid>.out` shows requests. If the URL stops answering for
  more than ~10 min, check `squeue`; if no job is running, start option C.
- **If Qwen3.6 can't start:** use the gpt-oss-20b backup command in
  [running_vllm.md](running_vllm.md#backup-gpt-oss-20b) and tell users to change the model
  name to `openai/gpt-oss-20b`.

## After

```bash
scancel -u $USER -n vllm      # stops the job and any queued successor
```

A job left running bills until walltime (option A: up to its full `JOB_TIME`). If the key
went to many people, rotate it afterwards: new `VLLM_API_KEY` in `~/.bv_llm_host/secrets.env`.
