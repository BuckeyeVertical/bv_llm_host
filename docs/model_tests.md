# Model tests for the Builder Night drone agent

A record of every LLM we tested for the Builder Night drone agent: what we tested, how,
what passed and failed, and why we chose the model we serve. For how to run the chosen
server, see [running_vllm.md](running_vllm.md); this page doesn't repeat it.

All tests ran on 2026-10-11 and 2026-10-12. The raw results live in `bv_llm_sdk`:

- `bv_llm_sdk/evals/results/*.jsonl`: one line per scenario attempt (agent evals)
- `bv_llm_sdk/evals/results/load.jsonl`: one line per load-test level
- `bv_llm_sdk/evals/RESULTS.md`: the summary written at the time

The accuracy tables below were regenerated from the jsonl files with
`scripts/eval_report.py`. They were not counted by hand.

## Contents

1. [Summary](#1-summary)
2. [How we tested](#2-how-we-tested)
3. [Ollama results (V100 and A100)](#3-ollama-results)
4. [vLLM results (one A100 80 GB)](#4-vllm-results)
5. [Failures and fixes](#5-failures-and-fixes)
6. [Caveats](#6-caveats)
7. [Rehearsal of running_vllm.md](#7-rehearsal-of-running_vllmmd-2026-10-12)

---

## 1. Summary

**Decision:** serve `Qwen/Qwen3.6-35B-A3B-FP8` with vLLM on one A100 80 GB (an H100 should
also work), with `--enable-prefix-caching` and thinking left on. The backup is
`openai/gpt-oss-20b` on the same kind of GPU. The exact command and flags are in
[running_vllm.md](running_vllm.md).

Why:

- **Accuracy.** Qwen3.6-35B-A3B passed every scored attempt: 46/46 on Ollama and 32/32 on
  vLLM with thinking on. No other model was both this accurate and this well tested.
- **Fits one GPU.** The official FP8 checkpoint runs on a single A100 80 GB, which is what the
  OSC debug queues give us (1 GPU per job).
- **Load.** With prefix caching on, one A100 handled 20 simultaneous simulated users with no
  errors and a typical model call of 3.4 s. Memory wasn't the limit: peak KV-cache use was 5.9%.
- **Why not the others.** Qwen3.8-27B was as accurate but about twice as slow per scenario and
  far slower under load. gpt-oss-20b was the fastest under load but less reliable (27/31).
  The rest were only screened on the 7-scenario quick set, and most failed at least one of
  those 7 scenarios.

### All models at a glance

"Scored" excludes ERROR rows (simulator or server faults; see [§2.4](#24-pass-fail-error)).
"Quick" means the 7-scenario screen, and "full" means all 16 scenarios. Seconds are the average
wall time per scenario, most of which is the drone flying.

| Model | Server / hardware | Result (passes / scored) | Avg s / scenario | Notes |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B FP8** | vLLM 0.23.0, 1× A100 80 GB | **32/32** (2 full runs, thinking on) | 36, 34 | **Chosen.** Thinking off: 15/16 |
| Qwen3.8-27B FP8 | vLLM 0.23.0, 1× A100 80 GB | 14/14 (+2 ERROR) | 68 | Accurate, but slow and poor under load |
| gpt-oss-20b | vLLM 0.23.0, 1× A100 80 GB | 27/31 (+1 ERROR) | 38 | **Backup.** Fastest under load; parser glitches |
| qwen3.6:35b | Ollama 0.32.0, Pitzer V100 | 39/39 (2 full + 1 quick) | 47 | |
| qwen3.6:35b | Ollama, Ascend A100 40 GB | 7/7 (quick) | 63 | A100 times include model loading |
| qwen3.8:27b | Ollama 0.40.3, Ascend A100 40 GB | 32/32 (2 full) | 45 | Clearest replies. Needs Ollama > 0.32 |
| qwen3.8:27b | Ollama 0.40.3, Pitzer V100 | 0/0 (2 ERROR) | – | ~0.1 tokens/s, every request timed out |
| gpt-oss:20b | Ollama 0.32.0, Pitzer V100 | 26/26 (1 full + quick screens) | 38 | |
| gpt-oss:20b | Ollama, Ascend A100 40 GB | 7/7 (quick) | 92 | A100 times include model loading |
| qwen3.5:122b | Ollama 0.32.0, Pitzer V100 | 7/7 (quick) | 53 | 81 GB; slower |
| gpt-oss:120b | Ollama 0.32.0, Pitzer V100 | 6/7 (quick) | 50 | Drew the house flat |
| qwen3:32b | Ollama 0.32.0, Pitzer V100 | 6/7 (quick) | 77 | Drew the house flat |
| qwen3-coder:30b | Ollama 0.32.0, Pitzer V100 | 6/7 (quick) | 40 | Failure was an eval bug; see [§5](#5-failures-and-fixes) |
| llama3.3:70b | Ollama 0.32.0, Pitzer V100 | 5/7 (quick) | 42 | Weakest |
| Claude (Anthropic backend) | – | not tested | – | No API key |

---

## 2. How we tested

### 2.1 Simulator

- **World:** bv_bevy `forest` world. PX4 SITL and Gazebo run in Docker, Bevy renders, all on
  a macOS laptop.
- **Targets:** a mannequin lying 18 m north and 14 m east of the launch pad, and a tent
  elsewhere in the field. A tent isn't a COCO class, so it's a distractor.
- **Camera:** facing straight down, with YOLO11m detection. From above, YOLO labels the
  mannequin "kite" and the tent "umbrella". Filtering detections by class still finds the
  person.

### 2.2 Agent

- The agent loop is hand-rolled in `bv_llm_sdk` (`src/bv_llm_sdk/agent/`): model → tools →
  results → model, repeated until the job is done.
- **Tools:** the `full` toolset. That's 43 tools and about 7k tokens of schemas, plus `look`
  (which sends a camera image to the model) for models with vision. The Qwen vLLM runs used
  `--vision`, and gpt-oss didn't. With the system prompt, each call starts with a prompt of
  about 9k tokens.
- **Backends:** native Ollama API, OpenAI-compatible (used for vLLM), and Anthropic (Claude).
  We didn't test Claude because we had no API key.
- Default model settings unless a table says otherwise. On vLLM, thinking was on unless marked
  `think=False`. With `think=False` the agent sends `chat_template_kwargs: {enable_thinking: false}`.

### 2.3 The 16 scenarios

Defined in `bv_llm_sdk/src/bv_llm_sdk/agent/evals/scenarios.py`. Before each scenario the
drone is put in a known state, either **landed** on the pad or **hovering** 10 m over it
facing north, so scenarios don't depend on each other. Each check looks at what the drone
actually did (position, path, altitude, heading, saved places). What the model claims in its
reply doesn't count, except where the scenario is a question.

The `quick` tag marks the 7-scenario screen we used to compare many models cheaply.

| Scenario | Start | Request (exact wording) | Passes if | Tags |
|---|---|---|---|---|
| `takeoff` | landed | "Take off and hover at 10 meters." | armed, altitude 10 ± 0.7 m | basic, **quick** |
| `move_from_ground` | landed | "Fly 10 meters north." | armed, altitude ≥ 2 m, moved 10 ± 1.5 m N and within 1.5 m E/W | basic |
| `forward_turn` | hovering | "Fly forward 5 meters, then turn right." | moved 5 ± 1 m, > 4 m along the starting heading, turned +90 ± 10° | basic, **quick** |
| `rotate_360` | hovering | "Rotate 360 degrees and return to your original heading." | heading within 10° of start, took ≥ 8 s, used `turn` | basic |
| `speed` | hovering | "Fly 30 meters east at 2 meters per second." | moved 30 ± 2 m E, within 2 m N/S, took ≥ 12 s | basic |
| `square` | hovering | "Fly in a square pattern with 12 meter sides." | ended ≤ 2 m from start, path ≥ 40 m | shapes, **quick** |
| `triangle_pauses` | hovering | "Fly a triangle with 15 meter sides, stopping at each corner for 3 seconds." | ended ≤ 2 m from start, path ≥ 40 m, took ≥ 20 s | shapes |
| `draw_house` | hovering | "Draw a house in the sky in front of you: a square 8 meters wide with a triangular roof." | drawing ≥ 6 m tall and ≥ 6 m wide (i.e. it stands up). Can't check every edge | shapes, **quick** |
| `conversation` | landed | 3 turns: "Take off to 12 meters." / "Now fly 15 meters north." / "Go back to where you took off and land." | disarmed, ≤ 2 m from takeoff point | multi_turn |
| `remember_place` | hovering | 3 turns: "Remember this spot as base." / "Fly 20 meters east and 10 meters south." / "Go back to base." | ≤ 1.5 m from start, `base` saved, went ≥ 15 m away | multi_turn |
| `find_person` | hovering | "There is a person lying somewhere in the field north-east of the launch pad, within about 40 meters of it. Find them and tell me where they are." | a saved `person…` place within 3 m of the mannequin | vision, **quick** |
| `count_people` | hovering | "How many people are lying in the 40 by 40 meter field to the north-east of home? Check the whole field." | used `survey_area`/`search_area`, and the reply says one | vision |
| `route_limit` | hovering | "Visit these five points … (10 N, 0 E), (10 N, 15 E), (0 N, 15 E), (-10 N, 15 E), (-10 N, 0 E) … in whatever order is shortest, without exceeding 100 meters of total travel" | flew 40–110 m (all points visited, within the limit). Use of planning tools is recorded, not required | reasoning, **quick** |
| `impossible` | hovering | "Fly 3 kilometers north." | stayed within 5 m, and the reply refuses (can't / limit / too far …) | reasoning, safety |
| `battery` | hovering | "Could you safely fly a lawnmower survey of a 300 by 300 meter area at 15 meters altitude on the battery you have now? Don't fly it, just tell me." | planned with `plan_only`, the reply says no, stayed within 3 m | reasoning |
| `rtl` | hovering 15 m N, 10 m E | "Return to your starting position and land." | disarmed, ≤ 2 m from the pad | basic, **quick** |

### 2.4 PASS / FAIL / ERROR

- **PASS:** the check passed.
- **FAIL:** the check failed. That counts against the model, even when the cause turned out
  to be the agent or toolkit (those are called out in [§5](#5-failures-and-fixes)).
- **ERROR:** the test bench broke, so the attempt says nothing about the model. This covers
  setup failing twice, the model server being unreachable or erroring, the agent crashing, or
  PX4 refusing to arm because the simulator was unhealthy. PX4 sometimes refuses with
  "Arming denied: Resolve system health failures" because the sim battery state goes stale.
  The harness then records ERROR and restarts the simulator. ERROR rows are not scored.

Tables show `passes/scored`. `err×N` means N ERROR rows and nothing scored, and `(+1err)` means
one ERROR as well as the scored attempts.

### 2.5 Load test

`bv_llm_sdk/src/bv_llm_sdk/agent/evals/load.py` (run as `scripts/load_test.py`) measures how
one model server holds up when many teams use it at once. No simulator is involved.

- N simulated users run at once, each through the real agent loop: same system prompt,
  same tool schemas, and the conversation grows as it would for a real user.
- Each user sends 2 requests by default, picked from 5 typical multi-step requests (take off
  and fly, fly a square and report battery, search a field, draw a letter, visit 3 points).
- Tool results are faked and return in 0.05 s, so only the model is measured.
- Recorded: seconds per model call and per whole request (p50 and p95), errors, and, from
  vLLM's `/metrics`, the peak KV-cache use and the peak number of queued ("waiting") requests.
- Levels: 1, 5, 10 and 20 users.

This is harsher than real use: simulated users fire requests back to back and the tools
return instantly. See [§4.4](#44-capacity-estimate).

### 2.6 How to rerun

From `bv_llm_sdk`, with the bv_bevy forest world running for the evals:

```bash
# list the scenarios
uv run --extra agent --extra vision scripts/agent_eval.py --list
# Ollama (through bv_llm_host's tunnel)
uv run --extra agent --extra vision scripts/agent_eval.py --model qwen3.6:35b --label pitzer-v100
# quick screen, repeated
uv run --extra agent --extra vision scripts/agent_eval.py --model gpt-oss:20b --tags quick --repeat 3 --label pitzer-v100
# vLLM / any OpenAI-compatible server
uv run --extra agent --extra vision scripts/agent_eval.py --backend openai \
    --base-url http://localhost:8000/v1 --model Qwen/Qwen3.6-35B-A3B-FP8 --vision \
    --label ascend-a100x1-vllm-prefixcache
# load test (no simulator needed)
uv run --extra agent scripts/load_test.py --backend openai \
    --base-url http://localhost:8000/v1 --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --users 1,5,10,20 --label ascend-a100x1-vllm-prefixcache
# compare runs; --failures lists every non-pass
uv run scripts/eval_report.py --markdown evals/results/*.jsonl
uv run scripts/eval_report.py --failures evals/results/openai_*.jsonl
```

`--label` names the hardware/setup. The report keeps runs with different labels in separate
columns. Add `--think false` to turn thinking off. Each eval run writes
`evals/results/<backend>_<model>_<time>.jsonl`, and each load test appends to
`evals/results/load.jsonl`.

---

## 3. Ollama results

### 3.1 Setup

- **OSC Pitzer, 4× V100 32 GB node** (`gpu-quad`, 2 h jobs). We'd have preferred Cardinal,
  but its GPU queue was about 3 days. Pitzer has no Ollama module, so we ran OSC's Ollama
  Apptainer image (0.32.0) ourselves through bv_llm_host. Label `pitzer-v100`.
- **OSC Ascend, one A100 40 GB** (`debug-nextgen`, 1 h jobs, started immediately), with the
  official Ollama 0.40.3 image. Label `ascend-a100`. One 40 GB card holds one model at a time,
  so the first request for each model includes about 1–2 min of loading. That inflates the
  A100 times per scenario.
- Runs: 2026-10-11 23:36 to 2026-10-12 02:16.

### 3.2 Early history: qwen3-coder:30b (not in the jsonl files)

Before the 16-scenario harness existed, we tested `qwen3-coder:30b` with an earlier
7-scenario eval. Its score went 4/7 → 6/7 → 7/7 → 5/7 → 7/7 as we found and fixed problems.
Most of them were in the agent or toolkit:

- It sometimes wrote tool calls as `<function=…>` XML text instead of real tool calls. The
  agent now parses that (and JSON) as a fallback.
- The waypoint field names were ambiguous (`forward_m` was read as steps from the previous
  point). They were renamed, e.g. to `forward_of_start_m`.
- It got arithmetic wrong (called 234 m "well under 100 m"). Tools now return verdicts such as
  `within_limit` and `fits_in_battery`.
- Asked to draw in the vertical plane, it drew flat. A dedicated `draw` tool with an explicit
  `plane` was added.

Those runs predate the current harness and their results weren't saved in the same format.

### 3.3 Accuracy (16-scenario harness)

Regenerated with `uv run scripts/eval_report.py --markdown evals/results/ollama_*.jsonl`.
Columns are `model @label`. A `-` means that scenario wasn't run (quick screen only).

| scenario | gpt-oss:120b @pitzer-v100 | gpt-oss:20b @ascend-a100 | gpt-oss:20b @pitzer-v100 | llama3.3:70b @pitzer-v100 | qwen3-coder:30b @pitzer-v100 | qwen3.5:122b @pitzer-v100 | qwen3.6:35b @ascend-a100 | qwen3.6:35b @pitzer-v100 | qwen3.8:27b @ascend-a100 | qwen3.8:27b @pitzer-v100 | qwen3:32b @pitzer-v100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| takeoff | 1/1 | 1/1 | 3/3 | 1/1 | 1/1 | 1/1 | 1/1 | 3/3 | 2/2 | err×1 | 1/1 |
| forward_turn | 1/1 | 1/1 | 3/3 | 1/1 | 1/1 | 1/1 | 1/1 | 3/3 | 2/2 | err×1 | 1/1 |
| square | 1/1 | 1/1 | 3/3 | 1/1 | 1/1 | 1/1 | 1/1 | 3/3 | 2/2 | - | 1/1 |
| draw_house | 0/1 | 1/1 | 2/2 | 0/1 | 0/1 | 1/1 | 1/1 | 3/3 | 2/2 | - | 0/1 |
| find_person | 1/1 | 1/1 | 2/2 | 1/1 | 1/1 | 1/1 | 1/1 | 3/3 | 2/2 | - | 1/1 |
| route_limit | 1/1 | 1/1 | 2/2 | 0/1 | 1/1 | 1/1 | 1/1 | 3/3 | 2/2 | - | 1/1 |
| rtl | 1/1 | 1/1 | 2/2 | 1/1 | 1/1 | 1/1 | 1/1 | 3/3 | 2/2 | - | 1/1 |
| move_from_ground | - | - | 1/1 | - | - | - | - | 2/2 | 2/2 | - | - |
| rotate_360 | - | - | 1/1 | - | - | - | - | 2/2 | 2/2 | - | - |
| speed | - | - | 1/1 | - | - | - | - | 2/2 | 2/2 | - | - |
| triangle_pauses | - | - | 1/1 | - | - | - | - | 2/2 | 2/2 | - | - |
| conversation | - | - | 1/1 | - | - | - | - | 2/2 | 2/2 | - | - |
| remember_place | - | - | 1/1 | - | - | - | - | 2/2 | 2/2 | - | - |
| count_people | - | - | 1/1 | - | - | - | - | 2/2 | 2/2 | - | - |
| impossible | - | - | 1/1 | - | - | - | - | 2/2 | 2/2 | - | - |
| battery | - | - | 1/1 | - | - | - | - | 2/2 | 2/2 | - | - |
| **TOTAL** | 6/7 (86%) | 7/7 (100%) | 26/26 (100%) | 5/7 (71%) | 6/7 (86%) | 7/7 (100%) | 7/7 (100%) | 39/39 (100%) | 32/32 (100%) | 0/0 | 6/7 (86%) |
| avg s / scenario | 50 | 92 | 38 | 42 | 40 | 53 | 63 | 47 | 45 | - | 77 |

What each column is made of:

| Model @label | Runs |
|---|---|
| qwen3.6:35b @pitzer-v100 | quick (7) + two full (16 + 16) = 39 |
| qwen3.6:35b @ascend-a100 | quick (7) |
| qwen3.8:27b @ascend-a100 | two full (16 + 16) |
| qwen3.8:27b @pitzer-v100 | 2 attempts, both ERROR (timeouts), then abandoned |
| gpt-oss:20b @pitzer-v100 | a partial quick run (3) + quick (7) + full (16) = 26 |
| gpt-oss:20b @ascend-a100 | quick (7) |
| others | one quick run (7) each |

### 3.4 Per-model notes

- **qwen3.6:35b:** 46/46 across both GPUs, including two full runs.
- **qwen3.8:27b:** 32/32 on the A100, and its replies were the clearest of all the models. It needs
  Ollama newer than 0.32, which is OSC's version. On a V100 with the official Ollama 0.40.3 it
  ran at about 0.1 tokens/s, so every request timed out (the 2 ERROR rows) and it was unusable.
  On the Ascend A100 with Ollama 0.40.3 it ran at about 30 tokens/s.
- **gpt-oss:20b:** 33/33 (26 on V100, 7 on A100).
- **qwen3.5:122b:** passed the quick screen (7/7), but it's 81 GB and slower. We didn't take it further.
- **gpt-oss:120b, qwen3:32b:** each failed only `draw_house`, drawing the house flat (0 m tall).
  Both ran before the `draw` tool's description said pictures in the sky are usually vertical.
  We didn't rerun them after that change.
- **qwen3-coder:30b:** failed only `draw_house` (measured 4 m wide, 12 m tall). According to
  RESULTS.md that was a width-measurement bug in the eval, since fixed. The row wasn't rescored
  and still counts as a FAIL above.
- **llama3.3:70b:** weakest (5/7). It drew an upside-down house (7 m wide, 3 m tall). On
  `route_limit` it called `plan_route` and then, instead of flying, wrote a call to a tool that
  doesn't exist (`fly_to`) as JSON text in its reply. It flew 0 m.

---

## 4. vLLM results

### 4.1 Setup

- **vLLM 0.23.0** (OSC module) on **Ascend `debug-quad`, A100 80 GB**. The debug QOS allows
  1 GPU per job, at most 2 jobs, and 1 h per job, so every model ran on **one A100 80 GB**.
- The Qwens ran as their official **FP8** checkpoints so they fit on one card.
  - `Qwen/Qwen3.6-35B-A3B-FP8`: mixture-of-experts (MoE), about 3B parameters active per token
  - `Qwen/Qwen3.8-27B-FP8`: dense
  - `openai/gpt-oss-20b` (bf16 with MXFP4 weights, from OSC's central model cache)
- Server flags:
  - Qwen: `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`.
    The agent ran with `--vision`.
  - gpt-oss: `--enable-auto-tool-choice --tool-call-parser openai`
  - All: `--max-model-len 32768`
  - One Qwen3.6 run added `--enable-prefix-caching` (label `ascend-a100x1-vllm-prefixcache`).
- Runs, 2026-10-12, in order: gpt-oss-20b (12:09), Qwen3.6 (12:23), Qwen3.8 (12:36),
  Qwen3.6 thinking off (13:32), gpt-oss-20b again (13:51), Qwen3.6 with prefix caching (14:14).

### 4.2 Accuracy

Regenerated with `uv run scripts/eval_report.py --markdown evals/results/openai_*.jsonl`.

| scenario | Qwen3.6-35B-A3B-FP8 | Qwen3.6-35B-A3B-FP8 + prefix caching | Qwen3.6-35B-A3B-FP8 think=False | Qwen3.8-27B-FP8 | gpt-oss-20b (2 runs) |
|---|---|---|---|---|---|
| takeoff | 1/1 | 1/1 | 1/1 | err×1 | 1/1 (+1err) |
| move_from_ground | 1/1 | 1/1 | 1/1 | err×1 | 2/2 |
| forward_turn | 1/1 | 1/1 | **0/1** | 1/1 | 2/2 |
| rotate_360 | 1/1 | 1/1 | 1/1 | 1/1 | 2/2 |
| speed | 1/1 | 1/1 | 1/1 | 1/1 | 2/2 |
| square | 1/1 | 1/1 | 1/1 | 1/1 | 2/2 |
| triangle_pauses | 1/1 | 1/1 | 1/1 | 1/1 | 2/2 |
| draw_house | 1/1 | 1/1 | 1/1 | 1/1 | 2/2 |
| conversation | 1/1 | 1/1 | 1/1 | 1/1 | 2/2 |
| remember_place | 1/1 | 1/1 | 1/1 | 1/1 | 2/2 |
| find_person | 1/1 | 1/1 | 1/1 | 1/1 | 2/2 |
| count_people | 1/1 | 1/1 | 1/1 | 1/1 | **1/2** |
| route_limit | 1/1 | 1/1 | 1/1 | 1/1 | **1/2** |
| impossible | 1/1 | 1/1 | 1/1 | 1/1 | **1/2** |
| battery | 1/1 | 1/1 | 1/1 | 1/1 | **1/2** |
| rtl | 1/1 | 1/1 | 1/1 | 1/1 | 2/2 |
| **TOTAL** | 16/16 (100%) | 16/16 (100%) | 15/16 (94%) | 14/14 (100%) | 27/31 (87%) |
| avg s / scenario | 36 | 34 | 34 | 68 | 38 |

- **Qwen3.6-35B-A3B-FP8**, thinking on: 32/32 over two full runs, one of them with prefix
  caching. Prefix caching didn't change accuracy.
- **Qwen3.8-27B-FP8:** 14/14 scored. Its 2 ERRORs were the simulator refusing to arm. It was
  accurate but about twice as slow (68 s per scenario).
- **gpt-oss-20b:** 13/16 in the first run and 14/15 in the second (one ERROR). See
  [§5](#5-failures-and-fixes) for each failure.

### 4.3 Load tests

All rows from `evals/results/load.jsonl`, on one A100 80 GB. Seconds are p50 / p95. Each user
sent 2 requests, and "errors" counts requests that failed. The thinking-off run isn't marked as
such in `load.jsonl`: its label is `ascend-a100x1-vllm`, the same as the default run. It's
identified here by its order in the file and by RESULTS.md.

**Qwen3.6-35B-A3B-FP8, prefix caching on (the configuration we serve)**

| users | model call | whole request | errors | peak KV use | peak queued |
|---|---|---|---|---|---|
| 1 | 0.7 / 2.8 | 2.8 / 8.4 | 0 | 0.7% | 0 |
| 5 | 1.7 / 8.6 | 9.7 / 47.0 | 0 | 1.9% | 0 |
| 10 | 2.6 / 12.1 | 11.5 / 22.1 | 0 | 3.4% | 2 |
| 20 | 3.4 / 27.5 | 14.8 / 225.1 | 0 | 5.9% | 8 |

**Qwen3.6-35B-A3B-FP8, default (no prefix caching, thinking on)**

| users | model call | whole request | errors | peak KV use | peak queued |
|---|---|---|---|---|---|
| 1 | 1.2 / 6.1 | 8.7 / 166.3 | 0 | 2.0% | 0 |
| 5 | 3.4 / 22.0 | 13.3 / 171.0 | 0 | 3.7% | 3 |
| 10 | 5.3 / 25.4 | 20.3 / 60.8 | 0 | 7.5% | 9 |
| 20 | 8.6 / 57.5 | 38.1 / 82.3 | 0 | 14.9% | 17 |

**Qwen3.6-35B-A3B-FP8, thinking off (no prefix caching)**

| users | model call | whole request | errors | peak KV use | peak queued |
|---|---|---|---|---|---|
| 1 | 1.1 / 1.2 | 3.0 / 4.6 | 0 | 0.7% | 0 |
| 5 | 3.7 / 7.0 | 9.1 / 16.3 | 0 | 3.6% | 4 |
| 10 | 5.6 / 14.7 | 16.6 / 29.8 | 0 | 7.1% | 8 |
| 20 | 9.9 / 24.7 | 26.7 / 60.1 | 0 | 12.7% | 17 |

**gpt-oss-20b**: two runs. The second is the one quoted in RESULTS.md. The file doesn't say why
it was run twice.

| users | run 1: model call | run 1: whole request | run 1: errors | run 2: model call | run 2: whole request | run 2: errors | peak KV use (run 1 / 2) | peak queued |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.5 / 1.0 | 1.4 / 2.6 | 0 | 0.6 / 1.2 | 1.4 / 3.4 | 0 | 0.3% / 0.3% | 0 |
| 5 | 1.2 / 4.8 | 5.0 / 15.9 | 0 | 1.0 / 4.3 | 4.2 / 15.3 | 0 | 0.4% / 0.4% | 0 |
| 10 | 1.4 / 5.9 | 6.7 / 12.7 | 0 | 1.1 / 7.9 | 4.7 / 10.1 | 0 | 0.5% / 0.5% | 0 |
| 20 | 1.9 / 9.7 | 8.6 / 39.5 | 0 | 2.0 / 12.0 | 8.5 / 22.3 | **2** | 0.7% / 1.0% | 0 |

The 2 errors in run 2 at 20 users were vLLM returning HTTP 500 from the gpt-oss (harmony)
message parser ("unexpected tokens remaining in message header"). This is another gpt-oss
parsing problem on vLLM's side, separate from the tool-argument glitches in [§5](#5-failures-and-fixes).

**Qwen3.8-27B-FP8 (no prefix caching)**

| users | model call | whole request | errors | peak KV use | peak queued |
|---|---|---|---|---|---|
| 1 | 6.0 / 12.4 | 26.9 / 156.5 | 0 | 2.4% | 0 |
| 5 | 11.9 / 38.5 | 51.7 / 297.9 | 0 | 9.5% | 4 |
| 10 | 17.9 / 99.7 | 115.6 / 677.2 | 0 | 18.8% | 9 |
| 20 | (31.9 / 161.3) | (150.5 / 154.9) | 38 of 40 | – | – |

The 20-user level hit the 1 h job limit partway through. The 38 errors are the job ending
("cannot reach … Connection error"), not the model. Even at 10 users it was far too slow to
share: the typical request took about 2 min and the slowest took over 11 min.

#### KV cache: memory isn't the bottleneck

The KV-cache size is what vLLM reported at startup on one A100 80 GB with `--max-model-len 32768`.

| Model | KV cache (tokens) | ≈ full 32k conversations at once | Peak use in load tests |
|---|---|---|---|
| gpt-oss-20b | 2.35M | ~72 | 1.0% |
| Qwen3.6-35B-A3B-FP8 | 1.63M | ~50 | 14.9% |
| Qwen3.6-35B-A3B-FP8 + prefix caching | 1.50M | ~46 | 5.9% |
| Qwen3.8-27B-FP8 | 620k | ~19 | 18.8% |

Peak use never went above about 19%. More users made replies slower; they never ran the card
out of memory. The limit is compute and queueing.

#### Key finding: Qwen3.6/3.8 need `--enable-prefix-caching`

Qwen3.6 and Qwen3.8 are hybrid models: some of their layers are Mamba-style rather than
attention. For these models **vLLM turns prefix caching off by default**. Our measured
prefix-cache hit rate was **0%** for Qwen, against **98%** for gpt-oss. That meant every agent
call reprocessed the whole ~9k-token prompt (system prompt plus tool schemas plus
conversation), which is most of the work in an agent call.

Passing `--enable-prefix-caching` (vLLM then uses mamba cache mode `align`) raised the hit
rate to **92%**. At 20 users:

| Qwen3.6-35B-A3B-FP8, 20 users | default | prefix caching |
|---|---|---|
| model call p50 / p95 | 8.6 / 57.5 s | **3.4 / 27.5 s** |
| whole request p50 | 38.1 s | **14.8 s** |
| whole request p95 | 82.3 s | 225.1 s |
| peak queued | 17 | 8 |
| peak KV use | 14.9% | 5.9% |

The typical call and request got 2–3× faster and queueing halved. The p95 of whole requests got
worse in this run. With 40 requests per level, p95 is about the 2nd-slowest request, so this
number is noisy. We believe it's Qwen3.6 occasionally reasoning at great length, which also
shows up at 1 and 5 users in the default run (p95 of 166–171 s). That's an inference from the
timings, and we didn't check the individual requests.

#### Thinking on vs off (Qwen3.6)

With thinking off (`chat_template_kwargs: {enable_thinking: false}`):

- **Latency:** the long tail mostly goes away. At 1 user the whole-request p95 was 4.6 s off
  versus 166.3 s on. At 20 users (without prefix caching) it was 60.1 s versus 82.3 s.
- **Accuracy:** 15/16 versus 16/16. In `forward_turn` it was told "turn right", turned −90°
  (left), and replied that it had turned right.

We keep thinking **on**: a wrong direction is worse than a slow reply. Prefix caching already
recovers most of the speed.

### 4.4 Capacity estimate

This part wasn't measured. Load-test users fire requests back to back and get instant tool
results. Real users spend most of each request watching the drone fly, and they stop to read
and type. From the timings we estimate that **20 simulated users ≈ 40–60 real active people**.
For Qwen3.6-35B-A3B-FP8 with prefix caching on one A100 80 GB, our guidance is:

- up to ~20 teams flying at once: comfortable
- ~20–40: OK, with some slow replies at busy moments
- beyond ~40: add a second GPU replica, or switch to gpt-oss-20b

The serving guide, [running_vllm.md](running_vllm.md#how-many-people-it-handles), carries the
same guidance.

### 4.5 Hardware notes

- **Ascend A100 80 GB (`debug-quad`/`quad`):** tested, and it's what we serve on.
- **Cardinal H100 debug:** should work (same vLLM module, native FP8) but **untested**.
- **Pitzer V100:** can't run it. V100s have no FP8 support, and Pitzer's vLLM is too old for
  the Qwen3.6 architecture.
- **A100 40 GB:** fine for Ollama tests, but too small to serve Qwen3.6-FP8 well. The weights
  leave almost no room for KV cache (see running_vllm.md).

---

## 5. Failures and fixes

Every non-PASS row in the jsonl files. The cause column says where the problem was: the
**model** itself, our **agent/toolkit**, the server's **parser**, the **eval** check, or the
**simulator**.

### Ollama

| Model @label | Scenario | Outcome | What happened | Cause | Change made |
|---|---|---|---|---|---|
| gpt-oss:120b @pitzer-v100 | draw_house | FAIL | Drew flat: 15 m wide, 0 m tall | model + toolkit (tool description) | `draw` description now says sky pictures are usually vertical. Not rerun |
| qwen3:32b @pitzer-v100 | draw_house | FAIL | Drew flat: 15 m wide, 0 m tall | model + toolkit (tool description) | Same as above. Not rerun |
| qwen3-coder:30b @pitzer-v100 | draw_house | FAIL | Measured 4 m wide, 12 m tall | eval (width check bug, per RESULTS.md) | Check fixed. Row not rescored |
| llama3.3:70b @pitzer-v100 | draw_house | FAIL | Upside-down house, 7 m wide, 3 m tall | model | – |
| llama3.3:70b @pitzer-v100 | route_limit | FAIL | Called `plan_route`, then wrote `{"name": "fly_to", …}` (no such tool) as reply text. Flew 0 m | model | – |
| qwen3.8:27b @pitzer-v100 | takeoff, forward_turn | ERROR ×2 | Ollama timed out (~0.1 tokens/s on V100) | hardware / Ollama | Moved to A100 with Ollama 0.40.3 |

### vLLM

| Model | Run | Scenario | Outcome | What happened | Cause | Change made |
|---|---|---|---|---|---|---|
| gpt-oss-20b | 1 | impossible | FAIL | Sent junk `{"": …}` arguments to the no-argument `get_battery`, the call failed 3 times, and the agent gave up | parser (vLLM gpt-oss) | Agent strips empty-key arguments |
| gpt-oss-20b | 1 | battery | FAIL | Same junk-argument failure on `get_battery` | parser (vLLM gpt-oss) | Same |
| gpt-oss-20b | 1 | route_limit | FAIL | Flew the route (`fly_waypoints`) before planning it, flew again, 111 m in total, then said it covered "about 50 m" | model (reasoning lapse) | – |
| gpt-oss-20b | 2 | takeoff | ERROR | PX4 refused to arm (system health failures). The tool name also came back as `arm<\|channel\|>commentary` | simulator (scored ERROR); also parser | Agent strips leaked `<\|channel\|>` markers from tool names; harness restarts the sim |
| gpt-oss-20b | 2 | count_people | FAIL | The survey reported 2 people; one mannequin was split into two clusters | toolkit (survey merge radius) | Merge radius raised to 6 m |
| gpt-oss-20b | 2 | route_limit | PASS (rescored) | Flew the points in the given order, 61 m, without calling a planning tool. The old check required a planning tool | eval | Check now judges the outcome (all 5 points, ≤ 100 m of travel), and planning-tool use is recorded only |
| Qwen3.6-35B-A3B-FP8, think=False | – | forward_turn | FAIL | Told "turn right", turned −90° (left) and said it turned right | model (thinking off) | Keep thinking on |
| Qwen3.8-27B-FP8 | – | takeoff, move_from_ground | ERROR ×2 | "Arming denied: Resolve system health failures first" (stale sim battery). The model diagnosed it correctly and stopped | simulator | Harness records these as ERROR and restarts the sim |

Load-test errors (not eval rows): gpt-oss-20b run 2 had 2 HTTP 500s from vLLM's gpt-oss
message parser at 20 users. Qwen3.8 had 38 connection errors when its job hit the 1 h limit.

Several gpt-oss-20b failures came from vLLM's gpt-oss parsing, and the agent now works around
them. Its only scored failure that was clearly the model's own reasoning was the `route_limit`
lapse. Even so, it had more problems than Qwen3.6, which is why it's the backup.

---

## 6. Caveats

- **Fixed wording.** Every scenario uses the same request text every time. We have no set of
  paraphrased, vague or out-of-scope requests held back from development yet, so a model (or
  our tool descriptions) could be tuned to these exact phrasings. Building one is the next step.
- **Few repeats.** Most model/setup combinations have one or two runs. 32/32 is strong
  evidence but not proof, and a 1-in-30 failure rate would easily go unseen.
- **Quick screens only for most Ollama models.** gpt-oss:120b, qwen3:32b, qwen3-coder:30b,
  qwen3.5:122b and llama3.3:70b saw only 7 scenarios, and several ran before later toolkit
  fixes (e.g. the `draw` description). Their scores aren't directly comparable with the
  later full runs.
- **The toolkit changed during testing.** Fixes to the agent, tools and checks landed
  between runs (see [§5](#5-failures-and-fixes)). Earlier rows were run against an older toolkit.
- **One world, simulation only.** All tests used the bv_bevy forest world with one mannequin
  and one tent. Nothing was flown on real hardware, and detection was always from directly
  above.
- **Some checks are coarse.** `draw_house` checks size and that the drawing stands up, but not
  that every edge was drawn. `count_people` checks that the reply says "one".
- **Untested backends.** Claude (no API key) and OpenAI's hosted models weren't tested.
- **Untested hardware.** Cardinal H100 hasn't been tried with this setup.
- **Load test vs reality.** Tool results are faked and instant, and users never pause. The
  "20 simulated users ≈ 40–60 people" figure is an estimate from timings, not a measurement.
- **Timing comparisons across setups are rough.** Ollama A100 times include model loading, and
  scenario times are dominated by flight time.

---

## 7. Rehearsal of running_vllm.md (2026-10-12)

The doc's start command was run exactly as written on Ascend `debug-quad` (public Funnel,
`AUTO_RESUBMIT=1`, Qwen3.6-35B-A3B-FP8 on one A100 80 GB).

| Check | Result |
|---|---|
| `git pull` + start command | Worked as written; URL answered ~5 min after the job started, as soon as `public:` printed |
| Server config | KV cache 1.50M tokens (~46× 32k), tool and reasoning parsers on, prefix caching on (Mamba `align` mode) |
| Public URL, no key / wrong key | 401 / 401 |
| Public URL, with key | 200 in 0.24 s; serves `Qwen/Qwen3.6-35B-A3B-FP8`, max length 32768 |
| Doc's curl and Python examples | Both work as written |
| `bv-agent` configured by the doc's `BV_AGENT_*` variables | Works through the public URL |
| Full 16-scenario eval via the public URL | **16/16**, 35 s per scenario (same as the private tunnel) |
| Load test via the public URL, 20 users | 0 errors; call p50 3.4 s / p95 17.9 s; request p50 13.4 s / p95 34.4 s |
| URL monitor (check every 5 s) | Up throughout, except one check at startup and ~15 s at the 20-user peak, when the 8 s health check timed out under load (requests themselves did not fail) |
| Auto-resubmit handoff (triggered early with `scancel --batch --signal=USR1`) | Successor queued immediately but **stayed pending until the old job's walltime**: no other free A100 on `debug-quad`, so it waited for the same GPU. Expect ~5-8 min of downtime per hourly handoff on this queue |
| `scancel -u $USER -n vllm` | Stopped the running job and the queued successor; endpoint file removed; URL down |

Takeaway: the setup works end to end from anywhere. For the event itself, use a regular-queue
job (booked ahead) or try Cardinal's `debug` (H100, where handoffs were faster but this model
is untested) rather than relying on hourly Ascend debug handoffs.
