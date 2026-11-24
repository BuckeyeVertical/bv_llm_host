#!/usr/bin/env python3
"""Test one VLM on a Roboflow dataset: accuracy, speed, resource usage.

Starts `vllm serve` (or uses --base-url), asks the model to label every image, and
writes result.json and predictions.jsonl to --out. Needs only OSC's vllm module.

  module load vllm/0.23.0
  python scripts/vlm_test.py --model Qwen/Qwen3-VL-8B-Instruct \\
      --dataset workspace/project/1 --out results/vlm/test

Roboflow downloads need ROBOFLOW_API_KEY; --dataset-dir uses an export already on
disk (COCO for detection, folder-per-class for classification).
"""
import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import psutil
import requests
from PIL import Image

BOX_FORMATS = ("norm1000_xyxy", "norm1000_yxyx", "abs_xyxy")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOUS = [0.5 + 0.05 * i for i in range(10)]


# ---------------------------------------------------------------- dataset

def download_roboflow(spec, root):
    """Download workspace/project/version, returning the extracted directory."""
    parts = spec.strip("/").split("/")
    if len(parts) != 3:
        sys.exit(f"--dataset must be workspace/project/version, got {spec!r}")
    final = Path(root) / "__".join(parts)
    if final.is_dir():
        return final
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit(f"{final} not downloaded yet and ROBOFLOW_API_KEY is not set")
    api, auth = "https://api.roboflow.com", {"Authorization": f"Bearer {key}"}

    def get(path):
        r = requests.get(f"{api}/{path}", headers=auth, timeout=60)
        if not r.ok:
            sys.exit(f"Roboflow {path}: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    ptype = get("/".join(parts[:2]))["project"]["type"]
    if ptype in ("object-detection", "instance-segmentation"):
        fmt = "coco"
    elif ptype == "single-label-classification" or ptype == "classification":
        fmt = "folder"
    else:
        sys.exit(f"unsupported Roboflow project type: {ptype}")
    # Roboflow builds the export on first request; poll until the link appears.
    for _ in range(120):
        link = (get("/".join(parts) + f"/{fmt}").get("export") or {}).get("link")
        if link:
            break
        time.sleep(5)
    else:
        sys.exit("timed out waiting for the Roboflow export")
    print(f"downloading {spec} ({ptype}) -> {final}", flush=True)
    part = final.with_name(f"{final.name}.part{os.getpid()}")
    part.mkdir(parents=True)
    try:
        with requests.get(link, stream=True, timeout=120) as r:
            if not r.ok:  # not raise_for_status: its message contains the signed link
                sys.exit(f"dataset download failed: HTTP {r.status_code}")
            with open(part / "export.zip", "wb") as f:
                shutil.copyfileobj(r.raw, f)
        with zipfile.ZipFile(part / "export.zip") as z:
            z.extractall(part / "data")
        (part / "data").rename(final)
    finally:
        shutil.rmtree(part, ignore_errors=True)
    return final


def load_dataset(root, split):
    """-> (task, classes, samples). Boxes are [x1,y1,x2,y2] as fractions of the image."""
    folder = Path(root) / split
    if not folder.is_dir():
        found = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
        sys.exit(f"no {split!r} split in {root} (found: {found})")
    coco_file = folder / "_annotations.coco.json"
    if not coco_file.exists():
        classes = sorted(p.name for p in folder.iterdir() if p.is_dir())
        samples = [{"path": p, "id": f"{c}/{p.name}", "label": c} for c in classes
                   for p in sorted((folder / c).iterdir()) if p.suffix.lower() in IMAGE_EXT]
        if not samples:
            sys.exit(f"no COCO annotations or class folders in {folder}")
        return "classification", classes, samples
    coco = json.loads(coco_file.read_text())
    used = {a["category_id"] for a in coco["annotations"]}
    supers = {c.get("supercategory") for c in coco["categories"]}
    # Roboflow adds a placeholder parent category ("none" supercategory, no boxes).
    cats = {c["id"]: c["name"] for c in coco["categories"]
            if c["id"] in used or not (c.get("supercategory") == "none" and c["name"] in supers)}
    boxes = {}
    for a in coco["annotations"]:
        boxes.setdefault(a["image_id"], []).append(a)
    samples = []
    for im in coco["images"]:
        gts = []
        for a in boxes.get(im["id"], []):
            x, y, w, h = map(float, a["bbox"])
            if w > 0 and h > 0:
                gts.append({"label": cats[a["category_id"]],
                            "box": [x / im["width"], y / im["height"],
                                    (x + w) / im["width"], (y + h) / im["height"]]})
        samples.append({"path": folder / im["file_name"], "id": im["file_name"], "gts": gts})
    return "detection", sorted(set(cats.values())), samples


def encode_image(path, max_side):
    """JPEG data URL resized to max_side with sides rounded to multiples of 28, so
    Qwen2.5-VL's own resize is a no-op and its pixel coordinates match (w, h)."""
    img = Image.open(path).convert("RGB")
    scale = min(1.0, max_side / max(img.size))
    w, h = (max(56, int(s * scale / 28 + 0.5) * 28) for s in img.size)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), w, h


# ---------------------------------------------------------------- prompt / parse

def build_prompt(task, classes, box_format):
    names = ", ".join(json.dumps(c) for c in classes)
    if task == "classification":
        return (f"Classify this image as exactly one of these classes: {names}.\n"
                'Answer with JSON only: {"label": "<class>"}')
    key, coords = {
        "norm1000_xyxy": ("bbox_2d", "[x1, y1, x2, y2], integers normalized to 0-1000"),
        "norm1000_yxyx": ("box_2d", "[y1, x1, y2, x2], integers normalized to 0-1000"),
        "abs_xyxy": ("bbox_2d", "[x1, y1, x2, y2] in absolute pixels"),
    }[box_format]
    return (f"Detect every instance of these classes in the image: {names}.\n"
            "Answer with a JSON array only, one object per instance: "
            f'[{{"{key}": {coords}, "label": "<class>"}}]. '
            "Use the class names exactly as given. If there are none, answer [].")


def norm_label(s):
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()


def strip_thinking(text):
    return text.rsplit("</think>", 1)[-1].strip()


def parse_objects(text):
    """-> (list of dicts, ok). Recovers the complete objects of a truncated array."""
    text = strip_thinking(text)
    m = re.search(r"```(?:json)?\s*(.*?)(?:```|$)", text, re.S)
    if m:
        text = m.group(1).strip()
    for candidate in (text, *re.findall(r"\[.*\]|\{.*\}", text, re.S)):
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list) and all(isinstance(x, dict) for x in v)]
            data = [data] if not lists else lists[0]
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return data, True
    objs = []
    for chunk in re.findall(r"\{[^{}]*\}", text):
        try:
            objs.append(json.loads(chunk))
        except ValueError:
            pass
    return objs, bool(objs)


def parse_detections(text, box_format, w, h):
    objs, ok = parse_objects(text)
    out, bad = [], 0
    for o in objs:
        box = next((o[k] for k in ("bbox_2d", "box_2d", "bbox", "box") if k in o), None)
        try:
            a, b, c, d = (float(v) for v in box)
        except (TypeError, ValueError):
            bad += 1
            continue
        if box_format == "norm1000_yxyx":
            a, b, c, d = b, a, d, c
        sx, sy = (w, h) if box_format == "abs_xyxy" else (1000, 1000)
        clip = lambda v: min(1.0, max(0.0, v))
        box = [clip(a / sx), clip(b / sy), clip(c / sx), clip(d / sy)]
        if box[2] <= box[0] or box[3] <= box[1]:
            bad += 1
            continue
        out.append({"label": str(o.get("label", o.get("class", ""))), "box": box})
    return out, ok and not (objs and bad == len(objs)), bad


def parse_class(text, classes):
    objs, ok = parse_objects(text)
    raw = objs[0].get("label", "") if ok and objs else strip_thinking(text)
    by_norm = {norm_label(c): c for c in classes}
    return by_norm.get(norm_label(str(raw).strip("\"'. ")))


# ---------------------------------------------------------------- metrics

def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(preds, gts, thr, use_class=True):
    """Greedy one-to-one matching by IoU -> [(pred index, gt index, iou)]."""
    pairs = sorted(((iou(p["box"], g["box"]), i, j) for i, p in enumerate(preds)
                    for j, g in enumerate(gts)
                    if not use_class or norm_label(p["label"]) == norm_label(g["label"])),
                   reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i), used_g.add(j)
            out.append((i, j, v))
    return out


def prf(tp, n_pred, n_gt):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"precision": p, "recall": r, "f1": 2 * tp / (n_pred + n_gt) if n_pred + n_gt else 0.0}


def detection_metrics(rows, classes):
    """VLMs give no confidence scores, so this is F1 at fixed IoU rather than mAP."""
    n_pred = sum(len(r["preds"]) for r in rows)
    n_gt = sum(len(r["gts"]) for r in rows)
    tp = {t: sum(len(match(r["preds"], r["gts"], t)) for r in rows) for t in IOUS}
    matched = [m[2] for r in rows for m in match(r["preds"], r["gts"], 0.5)]
    loc = sum(len(match(r["preds"], r["gts"], 0.5, use_class=False)) for r in rows)
    known = {norm_label(c) for c in classes}
    per_class = {}
    for c in classes:
        sub = lambda items: [x for x in items if norm_label(x["label"]) == norm_label(c)]
        n_p, n_g = (sum(len(sub(r[k])) for r in rows) for k in ("preds", "gts"))
        t = sum(len(match(sub(r["preds"]), sub(r["gts"]), 0.5)) for r in rows)
        per_class[c] = {**prf(t, n_p, n_g), "n_gt": n_g, "n_pred": n_p}
    present = [v["f1"] for v in per_class.values() if v["n_gt"]]
    f1s = [prf(tp[t], n_pred, n_gt)["f1"] for t in IOUS]
    return {**{f"{k}50": v for k, v in prf(tp[IOUS[0]], n_pred, n_gt).items()},
            "f1_50_95": sum(f1s) / len(f1s),
            "macro_f1_50": sum(present) / len(present) if present else 0.0,
            "class_agnostic_f1_50": prf(loc, n_pred, n_gt)["f1"],
            "mean_iou_matched": sum(matched) / len(matched) if matched else None,
            "count_mae": sum(abs(len(r["preds"]) - len(r["gts"])) for r in rows) / len(rows),
            "unknown_label_preds": sum(norm_label(p["label"]) not in known
                                       for r in rows for p in r["preds"]),
            "n_gt": n_gt, "n_pred": n_pred, "per_class": per_class}


def classification_metrics(rows, classes):
    per_class = {}
    for c in classes:
        tp = sum(r["pred"] == c and r["label"] == c for r in rows)
        n_gt = sum(r["label"] == c for r in rows)
        per_class[c] = {**prf(tp, sum(r["pred"] == c for r in rows), n_gt), "n_gt": n_gt}
    present = [v["f1"] for v in per_class.values() if v["n_gt"]]
    return {"accuracy": sum(r["pred"] == r["label"] for r in rows) / len(rows),
            "macro_f1": sum(present) / len(present) if present else 0.0, "per_class": per_class}


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))] if values else None


# ---------------------------------------------------------------- server / resources

def kill_tree(pid, grace=30):
    """SIGTERM then SIGKILL a process and all its descendants."""
    try:
        parent = psutil.Process(pid)
        procs = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for sig in ("terminate", "kill"):
        for p in procs:
            try:
                getattr(p, sig)()
            except psutil.NoSuchProcess:
                pass
        _, procs = psutil.wait_procs(procs, timeout=grace)
        if not procs:
            return


def cached_snapshot(hf_home, model):
    """Newest complete local snapshot dir of model, or None (caches can be partial)."""
    base = Path(hf_home) / "hub" / f"models--{model.replace('/', '--')}" / "snapshots"
    if not base.is_dir():
        return None
    for snap in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not (snap / "config.json").exists():
            continue
        index = snap / "model.safetensors.index.json"
        if index.exists():
            if all((snap / f).exists() for f in set(json.loads(index.read_text())["weight_map"].values())):
                return snap
        elif any((snap / f).exists() for f in ("model.safetensors", "consolidated.safetensors")):
            return snap
    return None


def ensure_model(model, hf_home, caches, ignore=()):
    """Use a complete copy from a read-only cache, else download to hf_home.
    -> (HF_HOME to serve from, snapshot dir)"""
    for home in filter(None, caches):
        snap = cached_snapshot(home, model)
        if snap:
            return str(home), snap
    # Set before the import, which reads it: the module's HF_HOME is read-only.
    os.environ["HF_HOME"] = str(hf_home)
    from huggingface_hub import snapshot_download
    # Also verifies and completes an interrupted download; a no-op when complete.
    print(f"checking/downloading {model} -> {hf_home}", flush=True)
    snap = snapshot_download(model, cache_dir=str(Path(hf_home) / "hub"),
                             ignore_patterns=["*.pth", "*.bin", "*.gguf", "original/*", *ignore])
    return str(hf_home), Path(snap)


def visible_gpus(n):
    ids = os.environ.get("CUDA_VISIBLE_DEVICES")
    if ids is None:
        try:
            ids = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                          text=True).replace("\n", ",")
        except (OSError, subprocess.CalledProcessError):
            ids = ""
    return [i.strip() for i in ids.split(",") if i.strip()][:n]


class Server:
    def __init__(self, model, args, hf_home, gpus, max_wait, log_path, snapshot):
        self.log_path = log_path
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        env = {**os.environ, "HF_HOME": hf_home, "HF_HUB_OFFLINE": "1",
               "CUDA_VISIBLE_DEVICES": ",".join(gpus)}
        # HF_HUB_CACHE would override HF_HOME; a key would make /v1 require auth.
        for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "VLLM_API_KEY", "ROBOFLOW_API_KEY"):
            env.pop(k, None)
        cmd = ["vllm", "serve", str(snapshot), "--served-model-name", model,
               "--host", "127.0.0.1", "--port", str(port),
               "--limit-mm-per-prompt", '{"image": 1}', *args]
        if len(gpus) > 1:
            cmd += ["--tensor-parallel-size", str(len(gpus))]
        self.cmd = cmd
        print("+", " ".join(cmd), flush=True)
        start = time.time()
        with open(log_path, "w") as log:
            self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"vllm exited with {self.proc.returncode}; see {log_path}")
                if time.time() - start > max_wait:
                    raise RuntimeError(f"vllm not ready after {max_wait}s; see {log_path}")
                try:
                    if requests.get(self.base + "/health", timeout=5).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
        except BaseException:
            self.stop()
            raise
        self.load_s = time.time() - start

    def log_stats(self):
        text = Path(self.log_path).read_text(errors="replace")
        weights = re.findall(r"Model loading took ([\d.]+) ?GiB", text)
        kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
        return {"weights_gib_per_gpu": float(weights[-1]) if weights else None,
                "kv_cache_tokens": int(kv[-1].replace(",", "")) if kv else None}

    def stop(self):
        kill_tree(self.proc.pid)


class Monitor(threading.Thread):
    """Samples GPU, server process and KV-cache usage once a second."""

    def __init__(self, gpus, pid, base):
        super().__init__(daemon=True)
        self.gpus, self.pid, self.base = gpus, pid, base
        self.samples, self.done = [], threading.Event()
        self.procs = {}

    def gpu_sample(self):
        if not self.gpus:
            return {}
        out = subprocess.check_output(
            ["nvidia-smi", "-i", ",".join(self.gpus), "--format=csv,noheader,nounits",
             "--query-gpu=memory.used,utilization.gpu,power.draw"], text=True, timeout=10)
        rows = [[float(v) for v in line.split(",")] for line in out.strip().splitlines()]
        return {"mem": sum(r[0] for r in rows) / 1024, "util": sum(r[1] for r in rows) / len(rows),
                "power": sum(r[2] for r in rows)}

    def proc_sample(self):
        if not self.pid:
            return {}
        parent = psutil.Process(self.pid)
        rss = cpu = 0.0
        for p in [parent, *parent.children(recursive=True)]:
            try:
                p = self.procs.setdefault(p.pid, p)  # cpu_percent needs the same object
                rss += p.memory_info().rss
                cpu += p.cpu_percent()
            except psutil.NoSuchProcess:
                pass
        return {"rss": rss / 2**30, "cpu": cpu}

    def kv_sample(self):
        text = requests.get(self.base + "/metrics", timeout=5).text
        vals = re.findall(r"^vllm:kv_cache_usage_perc(?:\{.*\})? (\S+)", text, re.M)
        return {"kv": max(map(float, vals)) * 100} if vals else {}

    def run(self):
        while not self.done.is_set():
            s = {"t": time.time()}
            for fn in (self.gpu_sample, self.proc_sample, self.kv_sample):
                try:
                    s.update(fn())
                except (OSError, ValueError, subprocess.SubprocessError, psutil.Error,
                        requests.RequestException):
                    pass
            self.samples.append(s)
            self.done.wait(1)

    def summary(self):
        self.done.set()
        self.join(timeout=20)
        col = lambda k: [s[k] for s in self.samples if k in s]
        mean = lambda v: sum(v) / len(v) if v else None
        peak = lambda v: max(v) if v else None
        pw = [(s["t"], s["power"]) for s in self.samples if "power" in s]
        joules = sum((t1 - t0) * (p0 + p1) / 2 for (t0, p0), (t1, p1) in zip(pw, pw[1:]))
        return {"gpu_mem_peak_gib": peak(col("mem")), "gpu_util_mean_pct": mean(col("util")),
                "gpu_power_mean_w": mean(col("power")), "gpu_power_peak_w": peak(col("power")),
                "gpu_energy_wh": joules / 3600 if len(pw) > 1 else None,
                "kv_cache_peak_pct": peak(col("kv")), "cpu_ram_peak_gib": peak(col("rss")),
                "cpu_mean_pct": mean(col("cpu")[1:])}


# ---------------------------------------------------------------- inference

async def ask(client, model, sample, prompt, args):
    row = {"id": sample["id"], "error": None, "text": ""}
    try:
        url, row["w"], row["h"] = await asyncio.to_thread(encode_image, sample["path"], args.max_side)
        start, ttft, usage, finish, parts = time.perf_counter(), None, None, None, []
        stream = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=args.max_tokens, stream=True,
            stream_options={"include_usage": True}, extra_body=args.extra_body or None,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": prompt}]}])
        async for chunk in stream:
            usage = chunk.usage or usage
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                finish = chunk.choices[0].finish_reason or finish
                if delta:
                    ttft = ttft or time.perf_counter() - start
                    parts.append(delta)
        row.update(latency_s=time.perf_counter() - start, ttft_s=ttft, finish=finish,
                   text="".join(parts), in_tokens=usage.prompt_tokens if usage else 0,
                   out_tokens=usage.completion_tokens if usage else 0)
    except Exception as e:  # one bad image must not end the run; it is counted below
        row["error"] = f"{type(e).__name__}: {e}"[:500]
    return row


async def run_inference(base, model, samples, prompt, args):
    """First --latency-images one at a time (clean latency), the rest concurrently
    (throughput). Every image is sent once, so caches never flatter a phase."""
    from openai import AsyncOpenAI
    async with AsyncOpenAI(base_url=base + "/v1", api_key=os.environ.get("OPENAI_API_KEY", "none"),
                           timeout=args.request_timeout, max_retries=0) as client:
        n = min(args.latency_images, len(samples))
        seq = [await ask(client, model, s, prompt, args) for s in samples[:n]]
        sem = asyncio.Semaphore(args.concurrency)

        async def limited(s):
            async with sem:
                return await ask(client, model, s, prompt, args)

        start = time.perf_counter()
        conc = await asyncio.gather(*(limited(s) for s in samples[n:]))
        return seq, list(conc), time.perf_counter() - start


async def warm_up(base, model, args):
    from openai import AsyncOpenAI
    buf = io.BytesIO()
    Image.new("RGB", (448, 448), "gray").save(buf, "JPEG")
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    async with AsyncOpenAI(base_url=base + "/v1", api_key=os.environ.get("OPENAI_API_KEY", "none"),
                           timeout=args.request_timeout) as client:
        await client.chat.completions.create(
            model=model, max_tokens=16, extra_body=args.extra_body or None,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "Describe."}]}])


def speed_metrics(seq, conc, wall):
    ok_seq = [r for r in seq if not r["error"]]
    ok_conc = [r for r in conc if not r["error"]]
    lat = [r["latency_s"] for r in ok_seq]
    ttft = [r["ttft_s"] for r in ok_seq if r["ttft_s"]]
    decode = [(r["out_tokens"] - 1) / (r["latency_s"] - r["ttft_s"]) for r in ok_seq
              if r["ttft_s"] and r["out_tokens"] > 1 and r["latency_s"] > r["ttft_s"]]
    everything = ok_seq + ok_conc
    mean = lambda v: sum(v) / len(v) if v else None
    return {"latency_mean_s": mean(lat), "latency_p50_s": percentile(lat, 0.5),
            "latency_p95_s": percentile(lat, 0.95),
            "ttft_p50_s": statistics.median(ttft) if ttft else None,
            "decode_tok_per_s": statistics.median(decode) if decode else None,
            "throughput_img_per_s": len(ok_conc) / wall if ok_conc else None,
            "throughput_out_tok_per_s": sum(r["out_tokens"] for r in ok_conc) / wall if ok_conc else None,
            "mean_in_tokens": mean([r["in_tokens"] for r in everything]),
            "mean_out_tokens": mean([r["out_tokens"] for r in everything]),
            "n_latency": len(ok_seq), "n_throughput": len(ok_conc)}


def add_common_args(p):
    """Options shared with vlm_benchmark.py, which forwards them."""
    user, acct = os.environ.get("USER", "user"), os.environ.get("OSC_ACCOUNT", "PAS3560")
    scratch = f"/fs/scratch/{acct}/{user}"
    g = p.add_argument_group("dataset and run")
    g.add_argument("--dataset", help="Roboflow workspace/project/version")
    g.add_argument("--dataset-dir", help="local Roboflow export to use instead")
    g.add_argument("--data-root", default=f"{scratch}/roboflow", help="where datasets download to")
    g.add_argument("--split", default="test")
    g.add_argument("--limit", type=int, default=200, help="images to sample, 0 = all")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-side", type=int, default=1120, help="longest image side sent to the model")
    g.add_argument("--max-tokens", type=int, default=2048)
    g.add_argument("--latency-images", type=int, default=20)
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--request-timeout", type=float, default=300)
    g.add_argument("--max-model-len", type=int, default=8192)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    g.add_argument("--startup-timeout", type=int, default=1800)
    g.add_argument("--hf-home", default=os.environ.get("VLM_HF_HOME") or f"{scratch}/hf",
                   help="where missing weights download to")
    g.add_argument("--hf-caches", default=os.environ.get(
        "VLM_HF_CACHES", "/fs/ess/pub_data/vllm/.cache/huggingface:/fs/scratch/PAS3560/allenthomas/hf"),
        help="colon-separated read-only caches to check first")


def resolve_dataset(args):
    if bool(args.dataset) == bool(args.dataset_dir):
        sys.exit("give exactly one of --dataset or --dataset-dir")
    return Path(args.dataset_dir) if args.dataset_dir else download_roboflow(args.dataset, args.data_root)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--box-format", choices=BOX_FORMATS, default=BOX_FORMATS[0])
    p.add_argument("--extra-body", type=json.loads, default={}, help="JSON added to each request")
    p.add_argument("--vllm-args", type=json.loads, default=[], help="JSON list of extra vllm flags")
    p.add_argument("--ignore", type=json.loads, default=[], help="JSON list of HF files to skip")
    p.add_argument("--base-url", help="use a running server (no /v1) instead of starting one")
    add_common_args(p)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # run the finally blocks

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").unlink(missing_ok=True)  # never leave a stale success behind
    task, classes, samples = load_dataset(resolve_dataset(args), args.split)
    samples.sort(key=lambda s: s["id"])
    random.Random(args.seed).shuffle(samples)
    samples = samples[:args.limit] if args.limit else samples
    prompt = build_prompt(task, classes, args.box_format)
    print(f"{task}: {len(samples)} images, {len(classes)} classes", flush=True)

    server = monitor = None
    try:
        gpus, info = [], {}
        if args.base_url:
            base = args.base_url.rstrip("/").removesuffix("/v1")
        else:
            gpus = visible_gpus(args.gpus)
            if len(gpus) < args.gpus:
                sys.exit(f"needs {args.gpus} GPUs, {len(gpus)} visible")
            home, snap = ensure_model(args.model, args.hf_home, args.hf_caches.split(":"), args.ignore)
            vllm_args = ["--max-model-len", str(args.max_model_len),
                         "--gpu-memory-utilization", str(args.gpu_memory_utilization), *args.vllm_args]
            server = Server(args.model, vllm_args, home, gpus, args.startup_timeout,
                            out / "vllm.log", snap)
            base = server.base
            from importlib.metadata import version
            info = {"revision": Path(snap).name, "vllm": version("vllm"), "command": server.cmd,
                    "gpu": subprocess.check_output(["nvidia-smi", "-i", ",".join(gpus),
                                                    "--query-gpu=name", "--format=csv,noheader"],
                                                   text=True).splitlines()[0]}
        asyncio.run(warm_up(base, args.model, args))
        monitor = Monitor(gpus, server.proc.pid if server else None, base)
        monitor.start()
        seq, conc, wall = asyncio.run(run_inference(base, args.model, samples, prompt, args))
        resources = monitor.summary()
        if server:
            resources.update(load_s=server.load_s, **server.log_stats())
        if resources["gpu_energy_wh"] is not None:
            resources["gpu_energy_wh_per_image"] = resources["gpu_energy_wh"] / len(samples)
    finally:
        if monitor:
            monitor.done.set()
        if server:
            server.stop()

    rows = seq + conc
    parse_failures = 0
    for r, s in zip(rows, samples):
        if task == "detection":
            preds, ok, r["bad_boxes"] = ([], True, 0) if r["error"] else \
                parse_detections(r["text"], args.box_format, r["w"], r["h"])
            r.update(preds=preds, gts=s["gts"])
        else:
            r.update(pred=None if r["error"] else parse_class(r["text"], classes), label=s["label"])
            ok = r["error"] or r["pred"] is not None
        r["parse_ok"] = bool(ok)
        parse_failures += not ok
    with open(out / "predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    errors = sum(bool(r["error"]) for r in rows)
    if errors == len(rows):
        sys.exit(f"every request failed, e.g. {rows[0]['error']}")
    result = {
        "model": args.model, "task": task, "n_images": len(rows), "classes": classes,
        # Failed requests stay in the accuracy numbers, as empty predictions.
        "accuracy": (detection_metrics if task == "detection" else classification_metrics)(rows, classes),
        "speed": speed_metrics(seq, conc, wall), "resources": resources,
        "health": {"request_errors": errors, "parse_failures": parse_failures,
                   "truncated": sum(r.get("finish") == "length" for r in rows)},
        "info": {**info, "gpus": len(gpus), "prompt": prompt, "job": os.environ.get("SLURM_JOB_ID"),
                 "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "settings": vars(args)}
    tmp = out / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    tmp.rename(out / "result.json")
    print(json.dumps({k: result[k] for k in ("accuracy", "speed", "resources", "health")}, indent=2))


if __name__ == "__main__":
    main()
