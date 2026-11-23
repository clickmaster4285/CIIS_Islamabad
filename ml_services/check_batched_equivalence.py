"""
Stage 3 check: batched GPU prefetch gives the same detections as the legacy per-frame path.

Needs the GPU + real weights (global Python, ML service stopped):
    python check_batched_equivalence.py --video motion_clips/<clip>.mp4

For each model it compares, frame by frame, legacy `_predict` (batch 1, per-camera class
filter) against `_gpu_prefetch` (one batched call, union of classes, filtered per camera).
Boxes are matched by class + IoU; small score/box drift from batched kernels is allowed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _boxes(result) -> list[tuple[int, float, np.ndarray]]:
    b = getattr(result, "boxes", None)
    if b is None or not len(b):
        return []
    return [
        (int(c), float(s), np.array(x, dtype=float))
        for c, s, x in zip(b.cls.int().tolist(), b.conf.tolist(), b.xyxy.tolist())
    ]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compare(legacy, batched, *, iou_min=0.9, score_tol=0.05) -> tuple[int, int, int]:
    """(matched, only_legacy, only_batched). Borderline-score boxes may flip; counted separately."""
    a, b = _boxes(legacy), _boxes(batched)
    used = set()
    matched = 0
    only_a = 0
    for c, s, x in a:
        best, best_j = 0.0, -1
        for j, (c2, s2, x2) in enumerate(b):
            if j in used or c2 != c:
                continue
            v = _iou(x, x2)
            if v > best:
                best, best_j = v, j
        if best >= iou_min and abs(b[best_j][1] - s) <= score_tol:
            used.add(best_j)
            matched += 1
        else:
            only_a += 1
    return matched, only_a, len(b) - len(used)


class _Stream:
    def __init__(self, frame):
        self.frame = frame
        self.connected = True

    def get_latest_meta(self):
        return self.frame, 1, 0.0

    def get_frame(self):
        return self.frame

    def stop(self):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    from load_env import load_ml_env

    load_ml_env()
    import live_stream

    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        if len(frames) == 0 or cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 == 0:
            frames.append(cv2.resize(f, (1280, 720)))
    cap.release()

    mgr = live_stream.LiveStreamManager()
    mgr._clips.start = lambda: None
    mgr.start()
    mgr._running = False  # models loaded; stop loops, we drive calls directly
    time = __import__("time")
    time.sleep(1.0)

    purposes = [
        ["general_objects", "custom_objects", "smoke_fire", "weapon", "face_recognition", "attendance", "anpr"],
        ["face_recognition", "attendance"],
        ["general_objects"],
        ["anpr", "smoke_fire"],
    ]
    totals: dict[str, list[int]] = {}
    for start in range(0, len(frames), args.batch):
        chunk = frames[start : start + args.batch]
        claimed = []
        for i, fr in enumerate(chunk):
            key = f"eq-{start + i}"
            mgr._purposes[key] = purposes[(start + i) % len(purposes)]
            sess = live_stream._CameraSession(key, _Stream(fr), "rtsp://eq")
            mgr._sessions[key] = sess
            claimed.append({"session": sess, "frame": fr, "seq": 1, "frame_ts": 0.0, "prev_seq": 0})
        mgr._gpu_prefetch(claimed)
        for item in claimed:
            key = item["session"].ip
            pipe = mgr._purpose_pipeline(key)
            infer_frame, _, _ = mgr._prepare_frame(item["frame"])
            pre = item["pre"]
            checks = []
            if "coco" in pre:
                ids = pipe.get("coco_classes")
                cf = live_stream.ALLOWED_COCO_CLASS_IDS if ids is None else frozenset(int(c) for c in ids)
                checks.append(("coco", mgr._predict(mgr._coco_model, infer_frame, classes=cf, tag="coco")[0]))
            if "pose" in pre:
                checks.append(
                    (
                        "pose",
                        mgr._predict(
                            mgr._pose_model,
                            infer_frame,
                            min_conf=live_stream.POSE_CONF,
                            imgsz=live_stream.POSE_IMGSZ,
                            tag="pose",
                        )[0],
                    )
                )
            if "custom" in pre:
                ids = live_stream.custom_only_class_ids(mgr._custom_model)
                checks.append(("custom", mgr._predict(mgr._custom_model, infer_frame, classes=ids or None, tag="custom")[0]))
            if "smoke" in pre:
                checks.append(
                    ("smoke", mgr._predict(mgr._smoke_model, infer_frame, min_conf=live_stream.SMOKE_FIRE_MIN_CONF, tag="smoke")[0])
                )
            if "weapon" in pre:
                checks.append(
                    ("weapon", mgr._predict(mgr._weapon_model, infer_frame, min_conf=live_stream.WEAPON_MIN_CONF, tag="weapon")[0])
                )
            # Coco with people must have pose prefetched (pose needs persons)
            coco_has_person = "coco" in pre and 0 in (pre["coco"].boxes.cls.int().tolist() if len(pre["coco"].boxes) else [])
            if coco_has_person and mgr._pose_model is not None:
                assert "pose" in pre, f"{key}: people found but pose not prefetched"
            for tag, legacy in checks:
                m, a, b = compare(legacy, pre[tag])
                t = totals.setdefault(tag, [0, 0, 0])
                t[0] += m
                t[1] += a
                t[2] += b
            # Class filter respected per camera
            if "coco" in pre and len(pre["coco"].boxes):
                ids = pipe.get("coco_classes")
                if ids is not None:
                    got = set(pre["coco"].boxes.cls.int().tolist())
                    assert got <= set(ids), f"{key}: classes {got - set(ids)} outside filter"
        mgr._specialist_count.clear()

    ok = True
    for tag, (m, a, b) in totals.items():
        total = m + a + b
        rate = m / total if total else 1.0
        print(f"{tag:7} matched {m}  only_legacy {a}  only_batched {b}  match_rate {rate:.3f}")
        if rate < 0.97:
            ok = False
    print("EQUIVALENT" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
