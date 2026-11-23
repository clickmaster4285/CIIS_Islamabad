"""
Stage 2.3 — recognise faces once per ByteTrack id instead of on every frame.

Live inference used to run YuNet + SFace on every person box of every frame (two crops
when the head crop finds no face). With ~20 people in view that is most of the frame time.

The gate keeps the last face result per (camera, track id):
  * known person   -> reused until ML_FACE_TRACK_REVERIFY_SEC (default 10 s)
  * unknown / none -> retried at most every ML_FACE_UNKNOWN_RETRY_SEC, so a person who
                      turns towards the camera is still recognised within that time
  * the box jumped (IoU with the track's previous box < ML_FACE_TRACK_MIN_IOU, or the
    centre moved more than ML_FACE_TRACK_MAX_JUMP box sizes) -> recognised again at once,
    because ByteTrack may have switched the id onto another person
Boxes without a track id are recognised every time (old behaviour).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable

from face_recognizer import is_unknown_label
from live_metrics import get_live_metrics

_FACE_META_KEYS = ("face_embedding", "face_match_score", "face_match_method")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _bbox4(det: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = det.get("bbox")
    try:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError, IndexError):
        return None


def is_face_target(det: dict[str, Any]) -> bool:
    name = str(det.get("class_name") or "").strip().lower()
    try:
        cls_id = int(det.get("class_id", -1))
    except (TypeError, ValueError):
        cls_id = -1
    return str(det.get("model") or "coco") == "coco" and (cls_id in (0, 80) or name in ("person", "face"))


class FaceTrackGate:
    def __init__(
        self,
        *,
        reverify_sec: float | None = None,
        unknown_retry_sec: float | None = None,
        forget_sec: float = 60.0,
        min_iou: float | None = None,
        max_jump: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.reverify_sec = (
            reverify_sec if reverify_sec is not None else _env_float("ML_FACE_TRACK_REVERIFY_SEC", 10.0)
        )
        self.unknown_retry_sec = (
            unknown_retry_sec if unknown_retry_sec is not None else _env_float("ML_FACE_UNKNOWN_RETRY_SEC", 1.0)
        )
        self.forget_sec = forget_sec
        self.min_iou = min_iou if min_iou is not None else _env_float("ML_FACE_TRACK_MIN_IOU", 0.3)
        self.max_jump = max_jump if max_jump is not None else _env_float("ML_FACE_TRACK_MAX_JUMP", 1.0)
        self._clock = clock
        self._state: dict[tuple[str, int], dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._last_prune = 0.0

    def _fresh(self, state: dict[str, Any] | None, now: float) -> bool:
        if not state:
            return False
        ttl = self.reverify_sec if state["known"] else self.unknown_retry_sec
        return (now - state["at"]) < ttl

    def _jumped(self, prev: tuple | None, box: tuple | None) -> bool:
        """True when the track's box moved implausibly far since the last frame (id switch)."""
        if prev is None or box is None:
            return False
        if _box_iou(prev, box) < self.min_iou:
            return True
        if self.max_jump <= 0:
            return False
        size = max(1.0, prev[2] - prev[0], prev[3] - prev[1])
        dx = (box[0] + box[2] - prev[0] - prev[2]) / 2.0
        dy = (box[1] + box[3] - prev[1] - prev[3]) / 2.0
        return (dx * dx + dy * dy) ** 0.5 > self.max_jump * size

    def label(self, camera_key: str, frame, detections: list[dict[str, Any]], face_db) -> list[dict[str, Any]]:
        """Fill label + face_* fields on person/face boxes (in place). Returns detections."""
        if face_db is None or not detections:
            return detections
        metrics = get_live_metrics()
        now = self._clock()
        for det in detections:
            if not is_face_target(det):
                continue
            box = _bbox4(det)
            if box is None:
                continue
            tid = det.get("track_id")
            try:
                key = (camera_key, int(tid)) if tid is not None else None
            except (TypeError, ValueError):
                key = None
            if key is not None:
                with self._lock:
                    state = self._state.get(key)
                    prev_box = state.get("box") if state is not None else None
                    if state is not None:
                        state["seen"] = now
                        state["box"] = box
                if self._fresh(state, now) and not self._jumped(prev_box, box):
                    self._apply(det, state["meta"], cached=True)
                    metrics.record_stage("face.track_gate_hit", 0.0)
                    continue
            t0 = time.perf_counter()
            meta = face_db.label_detection_detail(
                frame, int(det.get("class_id", 0)), str(det.get("class_name") or "person"), tuple(det["bbox"][:4])
            )
            metrics.record_stage("face.call_from_coco_parse", time.perf_counter() - t0)
            self._apply(det, meta, cached=False)
            if key is not None:
                with self._lock:
                    self._state[key] = {
                        "meta": meta,
                        "known": not is_unknown_label(str(meta.get("label") or "")),
                        "at": now,
                        "seen": now,
                        "box": box,
                    }
        self._prune(now)
        return detections

    @staticmethod
    def _apply(det: dict[str, Any], meta: dict[str, Any], *, cached: bool) -> None:
        det["label"] = meta.get("label") or det.get("class_name") or "person"
        for k in _FACE_META_KEYS:
            det.pop(k, None)
        if meta.get("face_embedding"):
            det["face_embedding"] = meta["face_embedding"]
        if meta.get("face_match_score") is not None:
            det["face_match_score"] = meta["face_match_score"]
        method = meta.get("face_match_method") or ""
        if cached and method:
            method = "track_cache"
        if method:
            det["face_match_method"] = method

    def _prune(self, now: float) -> None:
        if now - self._last_prune < 5.0:
            return
        self._last_prune = now
        cutoff = now - self.forget_sec
        with self._lock:
            for key in [k for k, s in self._state.items() if s["seen"] < cutoff]:
                del self._state[key]

    def reset(self, camera_key: str) -> None:
        with self._lock:
            for key in [k for k in self._state if k[0] == camera_key]:
                del self._state[key]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._state)
