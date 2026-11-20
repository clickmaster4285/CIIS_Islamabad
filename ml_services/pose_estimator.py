"""YOLO Pose for people: 17 body keypoints attached to each person detection.

Pose only exists for people. Other objects keep the normal detector. The person
detector stays the source of truth for boxes and IDs; pose boxes are matched to
those person boxes by overlap and their keypoints are copied onto the detection:

    det["keypoints"] = [[x, y, conf], ...]  # 17 COCO points, original-frame pixels
    det["pose_score"] = 0.87                # pose model's box confidence

COCO order: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
11-12 hips, 13-14 knees, 15-16 ankles (left, right).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_POSE_WEIGHTS = BASE_DIR / "models" / "pose" / "yolo26s-pose.pt"

KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
PERSON_CLASSES = frozenset({"person"})


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


POSE_ENABLED = os.getenv("ML_POSE_ENABLED", "true").strip().lower() in ("true", "1", "yes")
POSE_CONF = _env_float("ML_POSE_CONF", 0.30)
# Pose needs less detail than object detection: smaller image = much less GPU time
POSE_IMGSZ = max(320, int(_env_float("ML_POSE_IMGSZ", 640)))
# Pose box ↔ person box overlap needed to treat them as the same person
POSE_MATCH_IOU = _env_float("ML_POSE_MATCH_IOU", 0.45)

_model = None
_model_lock = threading.Lock()
_load_failed = False


def resolve_pose_weights() -> Path | None:
    raw = os.getenv("ML_POSE_WEIGHTS", "").strip()
    path = Path(raw) if raw else DEFAULT_POSE_WEIGHTS
    if not path.is_absolute():
        path = BASE_DIR / path
    return path if path.is_file() else None


def get_pose_model():
    """Load the pose model once (None when disabled or weights missing)."""
    global _model, _load_failed
    if not POSE_ENABLED or _load_failed:
        return None
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None or _load_failed:
            return _model
        weights = resolve_pose_weights()
        if weights is None:
            print("[pose] No pose weights (ML_POSE_WEIGHTS) — pose disabled.")
            _load_failed = True
            return None
        try:
            from ultralytics import YOLO

            model = YOLO(str(weights))
            if getattr(model, "task", "") != "pose":
                print(f"[pose] {weights.name} is not a pose model (task={model.task}) — pose disabled.")
                _load_failed = True
                return None
            _model = model
            print(f"[pose] Loaded {weights.name}")
        except Exception as exc:
            print(f"[pose] Load failed: {exc}")
            _load_failed = True
    return _model


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def is_person(det: dict[str, Any]) -> bool:
    return str(det.get("class_name") or det.get("label") or "").strip().lower() in PERSON_CLASSES


def parse_pose_result(result: Any, *, sx: float = 1.0, sy: float = 1.0) -> list[dict[str, Any]]:
    """Ultralytics pose result → [{"bbox": [...], "score": f, "keypoints": [[x, y, c] * 17]}] in original pixels."""
    boxes = getattr(result, "boxes", None)
    kpts = getattr(result, "keypoints", None)
    if boxes is None or kpts is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    if len(xyxy) == 0:
        return []
    scores = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
    pts = kpts.data.cpu().numpy() if hasattr(kpts.data, "cpu") else np.asarray(kpts.data)
    out = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
        points = []
        for p in pts[i]:
            conf = float(p[2]) if len(p) > 2 else 1.0
            points.append([round(float(p[0]) * sx, 1), round(float(p[1]) * sy, 1), round(conf, 3)])
        out.append({"bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy], "score": float(scores[i]), "keypoints": points})
    return out


def attach_poses(detections: list[dict[str, Any]], poses: list[dict[str, Any]], *, min_iou: float = POSE_MATCH_IOU) -> int:
    """Copy keypoints onto matching person detections (best overlap first, each pose used once)."""
    people = [d for d in detections if is_person(d) and len(d.get("bbox") or []) >= 4]
    pairs = []
    for di, det in enumerate(people):
        box = [float(v) for v in det["bbox"][:4]]
        for pi, pose in enumerate(poses):
            score = _iou(box, pose["bbox"])
            if score >= min_iou:
                pairs.append((score, di, pi))
    used_d: set[int] = set()
    used_p: set[int] = set()
    attached = 0
    for _score, di, pi in sorted(pairs, reverse=True):
        if di in used_d or pi in used_p:
            continue
        used_d.add(di)
        used_p.add(pi)
        people[di]["keypoints"] = poses[pi]["keypoints"]
        people[di]["pose_score"] = round(poses[pi]["score"], 3)
        attached += 1
    return attached
