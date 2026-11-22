# Notes

Things learned running models on OSC that aren't obvious from the scripts.

## Clusters and GPUs

| Cluster | GPU | vLLM / Ollama modules | Notes |
|---|---|---|---|
| Ascend | A100: 80 GB on `quad`, 40 GB on `nextgen` | yes | `debug`, `debug-nextgen`, `debug-quad` start fast but cap jobs at 1 h |
| Cardinal | H100 | yes | native FP8; `debug` is 1 h |
| Pitzer | V100 16/32 GB | no | needs `VLLM_VENV` / `OLLAMA_SIF`; no bf16 or FP8 |

- Size the GPU from the weights: bf16 takes about 2 GB per billion parameters, FP8 about 1 GB,
  and vLLM needs room for KV cache beyond that. A model that only just fits loads, but then
  serves few requests at once.
- Regular queues can take hours to days to start. Debug queues start in minutes. `squeue --me --start`
  shows Slurm's estimate.
- Check spending with `OSCusage -P <project> <start> <end>`.

## vLLM flags worth knowing

Set them in a profile's `ARGS`:

| Flag | Why |
|---|---|
| `--max-model-len N` | Models default to their full context (often 128k+), which reserves KV cache memory you may not need. 32k is plenty for chat. |
| `--reasoning-parser qwen3` (or the model's own) | Moves `<think>` text out of `content` into `reasoning_content` |
| `--enable-auto-tool-choice --tool-call-parser <p>` | Turns tool calls into real `tool_calls` instead of text. Parsers: `qwen3_coder`, `openai` (gpt-oss), `hermes`, `llama3_json` |
| `--enable-prefix-caching` | Reuses the cache for shared prompt prefixes. On by default except for hybrid (Mamba-style) models such as Qwen3.5/3.6, where turning it on made long prompts 2-3x faster |
| `--limit-mm-per-prompt '{"image": 4}'` | Caps images per request for vision models |
| `--served-model-name x` | The name clients pass as `model`, instead of the Hugging Face id |

Thinking models: if `content` comes back empty, the model spent `max_tokens` reasoning.
Raise `max_tokens`, or turn thinking off per request with
`extra_body={"chat_template_kwargs": {"enable_thinking": False}}` (Qwen3).

## Model storage

- OSC's shared caches are read-only. Anything else goes to `HF_CACHE` / `OLLAMA_CACHE`.
- Ollama always runs from `OLLAMA_CACHE`, because newer versions rewrite their model store on
  startup and fail on the read-only shared one. A model in the shared store gets its manifest
  copied and its weights symlinked, so nothing is downloaded.
- Ollama 0.40 no longer loads `llama3.2-vision` (architecture `mllama`), although it is in
  OSC's store. Use a newer vision model (`gemma3`, `qwen2.5vl`), or an older `OLLAMA_VERSION`.
- `/fs/scratch` deletes files that haven't been used in a while. Weights are easy to download
  again; check that they're still there before an important run.
- Gated models (Llama, Gemma) need `HF_TOKEN=hf_...` in `~/.bv_llm_host/secrets.env`
  (chmod 600) and accepted terms on huggingface.co.
- Downloads use the job's time. For big models run `scripts/download.sh <model>` first.

## Clusters without modules (Pitzer)

- vLLM: build a venv once, in a CPU batch job (`ensurepip` hangs on scratch, and login nodes
  cap CPU time): `uv venv $DIR && uv pip install --python $DIR vllm==<ver>`, and pin
  `transformers` to the 4.x release of that vLLM's era. Then set `VLLM_VENV=$DIR`. Choose a vLLM
  built for CUDA 12. On V100s the job adds `--dtype half` and turns off chunked prefill and
  prefix caching by itself.
- Ollama: copy the module's image to storage all clusters can read, then set `OLLAMA_SIF`:
  `cp /apps/ollama/<ver>/ollama_<ver>.sif /fs/scratch/<project>/...` (from Ascend or Cardinal).

## Troubleshooting

| Symptom | Look at |
|---|---|
| Job ends right away | `logs/<name>-<jobid>.out`: often out of GPU memory (lower `--max-model-len`, add GPUs) or an unknown flag |
| `url.sh --wait` waits forever | the job is still pending (`status.sh`) or downloading weights (the log) |
| Multi-GPU job hangs at startup | add `--disable-custom-all-reduce` to `ARGS` |
| `404` naming the model | send the id from `/v1/models` |
| Can't connect from a laptop | use `tunnel.sh`; compute nodes are only reachable from inside OSC |
