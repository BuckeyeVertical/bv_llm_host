#!/usr/bin/env python3
"""Run scripts/vlm_test.py over the models in vlm_models.json (up to --max-params-b)
and write report.md and results.csv (accuracy, speed, resource usage) to --out.

One model failing doesn't stop the rest. Rerunning resumes: models already tested
with the same settings are skipped (--force reruns them).
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import vlm_test

HERE = Path(__file__).resolve().parent
# vlm_test options forwarded to every run.
FORWARD = ["dataset", "dataset_dir", "data_root", "split", "limit", "seed", "max_side", "max_tokens",
           "latency_images", "concurrency", "request_timeout", "max_model_len",
           "gpu_memory_utilization", "startup_timeout", "hf_home", "hf_caches"]
# Those that don't affect the measurements.
NOT_MEASURED = {"data_root", "startup_timeout", "hf_home", "hf_caches"}

# (header, section, key, format)
ACCURACY = {
    "detection": [("F1@50", "accuracy", "f150", ".3f"), ("P@50", "accuracy", "precision50", ".3f"),
                  ("R@50", "accuracy", "recall50", ".3f"), ("F1@50:95", "accuracy", "f1_50_95", ".3f"),
                  ("Macro F1@50", "accuracy", "macro_f1_50", ".3f"),
                  ("Class-agnostic F1@50", "accuracy", "class_agnostic_f1_50", ".3f"),
                  ("Mean IoU", "accuracy", "mean_iou_matched", ".3f"),
                  ("Count MAE", "accuracy", "count_mae", ".2f")],
    "classification": [("Accuracy", "accuracy", "accuracy", ".3f"),
                       ("Macro F1", "accuracy", "macro_f1", ".3f")],
}
HEALTH = [("Request errors", "health", "request_errors", "d"),
          ("Parse failures", "health", "parse_failures", "d"), ("Truncated", "health", "truncated", "d")]
SPEED = [("Latency p50 s", "speed", "latency_p50_s", ".2f"), ("Latency p95 s", "speed", "latency_p95_s", ".2f"),
         ("TTFT p50 s", "speed", "ttft_p50_s", ".2f"), ("Decode tok/s", "speed", "decode_tok_per_s", ".0f"),
         ("Images/s", "speed", "throughput_img_per_s", ".2f"),
         ("Out tok/s", "speed", "throughput_out_tok_per_s", ".0f"),
         ("In tok/img", "speed", "mean_in_tokens", ".0f"), ("Out tok/img", "speed", "mean_out_tokens", ".0f")]
RESOURCES = [("GPUs", "info", "gpus", "d"), ("Weights GiB/GPU", "resources", "weights_gib_per_gpu", ".1f"),
             ("Peak GPU mem GiB", "resources", "gpu_mem_peak_gib", ".1f"),
             ("KV cache peak %", "resources", "kv_cache_peak_pct", ".1f"),
             ("GPU util %", "resources", "gpu_util_mean_pct", ".0f"),
             ("GPU power W", "resources", "gpu_power_mean_w", ".0f"),
             ("Wh/image", "resources", "gpu_energy_wh_per_image", ".4f"),
             ("CPU %", "resources", "cpu_mean_pct", ".0f"),
             ("Peak RAM GiB", "resources", "cpu_ram_peak_gib", ".1f"),
             ("Load s", "resources", "load_s", ".0f")]


def per_model_args(m):
    return {"model": m["model"], "gpus": m.get("gpus", 1),
            "box_format": m.get("box_format", vlm_test.BOX_FORMATS[0]),
            "extra_body": m.get("extra_body", {}), "vllm_args": m.get("vllm_args", []),
            "ignore": m.get("ignore", [])}


def load_result(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def same_settings(result, args, m):
    want = {**{k: getattr(args, k) for k in FORWARD if k not in NOT_MEASURED}, **per_model_args(m)}
    want.pop("ignore")
    return all(result["settings"].get(k) == v for k, v in want.items())


def run_model(args, m, out, gpus):
    """-> (status, note)"""
    need = m.get("gpus", 1)
    if need > gpus:
        return "skipped", f"needs {need} GPUs, job has {gpus}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").unlink(missing_ok=True)
    cmd = [sys.executable, "-u", str(HERE / "vlm_test.py"), "--out", str(out)]
    for k, v in per_model_args(m).items():
        cmd += [f"--{k.replace('_', '-')}", v if isinstance(v, str) else json.dumps(v)]
    for k in FORWARD:
        v = getattr(args, k)
        if v is not None:
            cmd += [f"--{k.replace('_', '-')}", str(v)]
    with open(out / "test.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=args.model_timeout)
        except subprocess.TimeoutExpired:
            return "failed", f"timed out after {args.model_timeout}s"
        finally:
            vlm_test.kill_tree(proc.pid, grace=90)
    if code == 0 and (out / "result.json").exists():
        return "ok", ""
    lines = (out / "test.log").read_text(errors="replace").strip().splitlines()
    return "failed", (lines[-1][:200] if lines else f"exit code {code}")


def cell(result, section, key, fmt):
    v = result.get(section, {}).get(key)
    return "-" if v is None else format(v, fmt)


def table(entries, columns):
    head = ["Model", "Params B"] + [c[0] for c in columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m, r in entries:
        lines.append("| " + " | ".join([m["model"], f"{m['params_b']:g}"]
                                       + [cell(r, *c[1:]) for c in columns]) + " |")
    return "\n".join(lines)


def write_report(args, out, models, status):
    entries = [(m, r) for m in models if status.get(m["model"], ("",))[0] == "ok"
               and (r := load_result(out / m["model"].replace("/", "__") / "result.json"))]
    dataset = args.dataset or args.dataset_dir
    doc = [f"# VLM benchmark: {dataset}", "",
           f"Updated {time.strftime('%Y-%m-%d %H:%M')}. {len(entries)} of {len(models)} models done.", ""]
    if entries:
        task = entries[0][1]["task"]
        main_key = "f150" if task == "detection" else "accuracy"
        entries.sort(key=lambda e: -e[1]["accuracy"][main_key])
        first = entries[0][1]
        doc += [f"- **Data:** {task}, `{args.split}` split, {first['n_images']} images (seed {args.seed}), "
                f"{len(first['classes'])} classes, longest side {args.max_side} px",
                f"- **Hardware / server:** {first['info'].get('gpu', '?')}, vLLM {first['info'].get('vllm', '?')}",
                "", "## Accuracy", "", table(entries, ACCURACY[task] + HEALTH), ""]
        if task == "detection":
            doc += ["F1/P/R are micro-averaged with class-aware matching at IoU 0.5. mAP is omitted because "
                    "VLMs give no confidence scores. Failed requests count as empty predictions.", ""]
        doc += ["## Speed", "", table(entries, SPEED), "",
                f"Latency, TTFT and decode: one request at a time (first {args.latency_images} images). "
                f"Images/s and out tok/s: the rest at concurrency {args.concurrency}.", "",
                "## Resource usage", "", table(entries, RESOURCES), "",
                "Measured during inference only. vLLM preallocates "
                f"{args.gpu_memory_utilization:.0%} of GPU memory, so peak memory is similar for every model; "
                "compare weights instead. GPU memory and power are summed over GPUs, utilization is averaged.", ""]
        header = ["model", "params_b"] + [c[0] for c in ACCURACY[task] + HEALTH + SPEED + RESOURCES]
        with open(out / "results.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for m, r in entries:
                w.writerow([m["model"], m["params_b"]] + [r.get(s, {}).get(k) for _, s, k, _ in
                                                         ACCURACY[task] + HEALTH + SPEED + RESOURCES])
    else:
        (out / "results.csv").unlink(missing_ok=True)
    other = [(m["model"], *status[m["model"]]) for m in models
             if m["model"] in status and status[m["model"]][0] != "ok"]
    if other:
        doc += ["## Not measured", "", "| Model | Status | Reason |", "|---|---|---|"]
        doc += [f"| {n} | {s} | {note.replace('|', '/')} |" for n, s, note in other] + [""]
    (out / "report.md").write_text("\n".join(doc))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--models-file", default=str(HERE / "vlm_models.json"))
    p.add_argument("--models", help="comma-separated subset of model ids")
    p.add_argument("--max-params-b", type=float, default=60)
    p.add_argument("--model-timeout", type=int, default=5400, help="seconds per model, download included")
    p.add_argument("--force", action="store_true", help="rerun finished models")
    p.add_argument("--download-only", action="store_true", help="fetch dataset and weights, test nothing")
    vlm_test.add_common_args(p)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    models = [m for m in json.loads(Path(args.models_file).read_text())["models"]
              if m["params_b"] <= args.max_params_b]
    if args.models:
        want = set(args.models.split(","))
        if missing := want - {m["model"] for m in models}:
            sys.exit(f"not in {args.models_file} (or over --max-params-b): {sorted(missing)}")
        models = [m for m in models if m["model"] in want]
    vlm_test.resolve_dataset(args)  # once, before any GPU time is spent
    if args.download_only:
        for m in models:
            try:
                home, _ = vlm_test.ensure_model(m["model"], args.hf_home, args.hf_caches.split(":"),
                                                m.get("ignore", []))
                print(f"{m['model']}: ready in {home}", flush=True)
            except Exception as e:
                print(f"{m['model']}: download failed: {e}", flush=True)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpus, status = len(vlm_test.visible_gpus(64)), {}
    print(f"{len(models)} models, {gpus} GPUs -> {out}", flush=True)
    for i, m in enumerate(models, 1):
        mdir = out / m["model"].replace("/", "__")
        old = load_result(mdir / "result.json")
        if old and not args.force and same_settings(old, args, m):
            status[m["model"]] = ("ok", "")
            print(f"[{i}/{len(models)}] {m['model']}: already done", flush=True)
        else:
            print(f"[{i}/{len(models)}] {m['model']} ({m['params_b']}B) ...", flush=True)
            start = time.time()
            status[m["model"]] = run_model(args, m, mdir, gpus)
            print(f"    {' '.join(status[m['model']])} ({time.time() - start:.0f}s)", flush=True)
        write_report(args, out, models, status)
    print(f"report: {out / 'report.md'}")
    if not any(s == "ok" for s, _ in status.values()):
        sys.exit("no model succeeded")


if __name__ == "__main__":
    main()
