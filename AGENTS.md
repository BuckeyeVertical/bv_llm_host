# bv_llm_host

Runs LLM inference servers on OSC GPU nodes and exposes them to a laptop over an SSH tunnel. See README.md for usage.

## OSC access

```bash
ssh <user>@<cluster>.osc.edu                  # cluster: cardinal | ascend | pitzer
ssh <user>@<cluster>.osc.edu '<command>'      # one-off command
```

Clusters: Cardinal (H100), Ascend (A100), Pitzer (V100). GPU availability is usually Pitzer > Ascend > Cardinal, so submit on Pitzer when possible and fall back in that order.

User, project, and cluster live in `config.env` (per-machine, gitignored; template in `config.env.example`).

## Conventions

- Each backend is one `slurm/<backend>.sbatch`. It starts the server, then calls `publish_endpoint` and `serve_forever` from `slurm/common.sh`.
- `scripts/` holds the entry points: `submit.sh` runs on OSC, `tunnel.sh` runs on the laptop.
- Change code locally, push, then `git pull` on the OSC checkout.
- Cancel jobs when done; they hold GPUs until walltime.
- The OSC checkout, `config.env`, and `~/.bv_llm_host/` are shared by every cluster and every session. Don't change shared config for one test; set overrides per submit (`EXPOSE=funnel scripts/submit.sh vllm`).
- vLLM jobs always require `VLLM_API_KEY` from `~/.bv_llm_host/secrets.env` (servers bind `0.0.0.0`), even with `EXPOSE=none`. Clients send it as `OPENAI_API_KEY`; never print it.
- Running guide: `docs/running_vllm.md`.

## OSC notes

- Where a cluster lacks the module, the job falls back to a self-installed venv (e.g. `VLLM_VENV`). Build venvs with `uv` in a CPU batch job: `ensurepip` hangs on scratch and login nodes cap CPU time.
- Use OSC's modules (`module load <backend>/<version>`) and their helper scripts; check `module show` for what they set.
- Module helpers may be shell aliases, which don't expand in batch scripts. Call the underlying script.
- OSC's helpers bind servers to `0.0.0.0`, so tunnel to `node:port` via the login node rather than SSHing into compute nodes.
