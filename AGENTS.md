# bv_llm_host

Runs LLM / VLM inference servers (vLLM, Ollama) as Slurm jobs on OSC GPU nodes. Usage is in
README.md; cluster quirks and vLLM flags in docs/notes.md.

## OSC access

```bash
ssh <user>@<cluster>.osc.edu                  # cluster: ascend | cardinal | pitzer
ssh <user>@<cluster>.osc.edu '<command>'      # one-off command
```

User, project and storage live in `config.env` (per checkout, gitignored; template
`config.env.example`).

## Layout and conventions

- `models/<name>.env`: one model profile each (`BACKEND`, `MODEL`, `ARGS`, optional `GPUS`,
  `JOB_TIME`, `PARTITION`, `NAME`, `OLLAMA_*`). Precedence: config.env < profile < environment.
- `scripts/`: user commands. `lib.sh` holds the config loading shared by all of them.
  `tunnel.sh` runs on a laptop; the rest on OSC.
- `slurm/<backend>.sbatch`: starts the server, waits for it, then `publish_endpoint` and
  `serve_forever` from `slurm/common.sh`. `slurm/funnel.sh` is the optional public URL.
- A running server is listed in `$STATE_DIR/endpoints/<name>` (KEY=VALUE lines, including `URL`).
  Per-user secrets and Tailscale state stay in `~/.bv_llm_host/` regardless of `STATE_DIR`.
- Jobs are named `llm-<name>`; `status.sh` and `stop.sh` rely on that.
- Change code locally, push, then `git pull` on the OSC checkout.
- Cancel test jobs when done; they hold GPUs until walltime.
- The OSC checkout and `config.env` may be shared. Make one-off changes through environment
  overrides (`GPUS=2 scripts/serve.sh ...`), not by editing shared config.
- Never print secrets (`VLLM_API_KEY`, `TS_AUTHKEY`, `HF_TOKEN`).

## OSC notes

- Prefer OSC's modules (`module load vllm/<v>`, `ollama/<v>`). Their `*_start` helpers are shell
  aliases that don't expand in batch scripts; the jobs here run `vllm serve` and the Ollama
  Apptainer image directly instead.
- Build venvs in a CPU batch job with `uv`: `ensurepip` hangs on scratch and login nodes cap
  CPU time.
- Servers bind `0.0.0.0`, so other OSC nodes reach `node:port` directly and tunnels go through
  the login node.
