"""
Per-model GPU benchmark (no cameras, no service): FP16 vs FP32 and batch size.

Runs each live YOLO model on real frames from a video with batch sizes 1/2/4/8 and
reports per-call latency, per-image cost and the split between Ultralytics
preprocess / inference / postprocess. Stop the ML service first (shared GPU).

    python bench_models.py --video motion_clips/x.mp4 --models coco,pose --batches 1,2,4,8 --half both
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from load_env import load_ml_env  # noqa: E402

load_ml_env()  # read ml_services/.env BEFORE inference_engine reads its env at import time

from inference_engine import (  # noqa: E402
    get_yolo_coco_model,
    get_yolo_custom_model,
    get_yolo_smoke_model,
    get_yolo_weapon_model,
    resolve_ml_device,
)
from pose_estimator import get_pose_model  # noqa: E402


def load_frames(video: str, n: int, w: int, h: int) -> list:
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (w, h)))
    cap.release()
    if not frames:
        raise SystemExit(f"no frames read from {video}")
    return frames


def p(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def bench(model, frames, *, batch: int, half: bool, imgsz: int, device, iters: int) -> dict:
    import torch

    kw = dict(device=device, imgsz=imgsz, half=half, conf=0.3, verbose=False, batch=batch)
    sync = torch.cuda.synchronize if torch.cuda.is_available() and device != "cpu" else (lambda: None)
    model.predict(frames[:batch], **kw)  # warm-up for this shape/precision
    model.predict(frames[:batch], **kw)
    wall, pre, inf, post = [], [], [], []
    for i in range(iters):
        start = (i * batch) % max(1, len(frames) - batch)
        imgs = frames[start : start + batch]
        t0 = time.perf_counter()
        res = model.predict(imgs, **kw)
        sync()
        if len(res) != batch:
            raise RuntimeError(f"expected {batch} results, got {len(res)}")
        wall.append(time.perf_counter() - t0)
        sp = getattr(res[0], "speed", {}) or {}
        # Ultralytics speed is per image (ms) for the whole batch
        pre.append(float(sp.get("preprocess") or 0) * batch / 1000)
        inf.append(float(sp.get("inference") or 0) * batch / 1000)
        post.append(float(sp.get("postprocess") or 0) * batch / 1000)
    med = statistics.median(wall)
    return {
        "batch": batch,
        "half": half,
        "imgsz": imgsz,
        "call_ms_p50": round(med * 1000, 1),
        "call_ms_p95": round(p(wall, 0.95) * 1000, 1),
        "per_image_ms": round(med * 1000 / batch, 1),
        "images_per_sec": round(batch / med, 2),
        "pre_ms": round(statistics.median(pre) * 1000, 1),
        "infer_ms": round(statistics.median(inf) * 1000, 1),
        "post_ms": round(statistics.median(post) * 1000, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--models", default="coco,pose,custom,smoke,weapon")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--half", default="both", choices=["both", "true", "false"])
    ap.add_argument("--imgsz", type=int, default=0, help="0 = ML_YOLO_IMGSZ (pose uses its own)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out")
    args = ap.parse_args()

    import os

    from pose_estimator import POSE_IMGSZ

    device = resolve_ml_device()
    frames = load_frames(args.video, 64, args.width, args.height)
    loaders = {
        "coco": get_yolo_coco_model,
        "pose": get_pose_model,
        "custom": get_yolo_custom_model,
        "smoke": get_yolo_smoke_model,
        "weapon": get_yolo_weapon_model,
    }
    halves = {"both": [True, False], "true": [True], "false": [False]}[args.half]
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    default_imgsz = args.imgsz or int(os.getenv("ML_YOLO_IMGSZ", "960"))
    rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        model = loaders[name]()
        if model is None:
            print(f"[bench] {name}: not available, skipped")
            continue
        imgsz = POSE_IMGSZ if name == "pose" else default_imgsz
        for half in halves:
            for b in batches:
                try:
                    r = bench(model, frames, batch=b, half=half, imgsz=imgsz, device=device, iters=args.iters)
                except Exception as exc:  # e.g. CUDA OOM at large batch
                    r = {"batch": b, "half": half, "imgsz": imgsz, "error": str(exc)[:200]}
                r["model"] = name
                rows.append(r)
                print(json.dumps(r))
    if args.out:
        Path(args.out).write_text(json.dumps({"device": str(device), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
