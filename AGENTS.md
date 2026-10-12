# bv_llm_host

Runs LLM servers (Ollama; vLLM next) on OSC GPU nodes, reached from a laptop via SSH tunnel. See README.md for layout and usage.

## OSC access

```bash
ssh <osc_user>@cardinal.osc.edu            # or ascend.osc.edu
cd /fs/scratch/PAS3560/bv_llm_host         # repo checkout on OSC (project PAS3560)
ssh <osc_user>@cardinal.osc.edu 'squeue -u $USER'   # one-off command
```

## Workflow

- OSC: `scripts/submit.sh ollama` → job writes `~/.bv_llm_host/ollama.endpoint`; log in `logs/`.
- Laptop: `scripts/tunnel.sh ollama` → `http://localhost:11434/v1` (OpenAI-compatible).
- Stop: `scancel <jobid>`. Jobs bill GPU time until walltime, so cancel when done.
- Change code locally, push, then `git pull` on OSC. `config.env` is per-machine and gitignored.

## OSC gotchas

- `ollama_start` is an interactive alias; batch jobs must `source ollama_start_helper`.
- OSC's helpers bind `0.0.0.0` with no auth, so the tunnel forwards login node → `node:port`. SSH into compute nodes fails host-key checks; don't use `-J`.
- Ollama runs in apptainer with `--cleanenv`; extra `OLLAMA_*` env vars aren't passed through.
- Central Ollama models (read-only, the default): `/fs/ess/pub_data/ollama/models`.
