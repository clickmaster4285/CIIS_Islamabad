"""Per-camera, per-class ByteTrack so every object gets a stable ID.

Track ids are unique per camera across all classes: every class slot of a camera draws
its ids from one per-camera counter (no hashing of the class name, no shared global
counter). ultralytics' STrack keeps a single class-level counter that every BYTETracker
creation resets, which made different cameras (and classes) hand out the same ids.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml


def _byte_tracker_args(*, track_buffer: int = 90) -> IterableSimpleNamespace:
    cfg = dict(YAML.load(check_yaml("bytetrack.yaml")))
    cfg["track_buffer"] = max(30, int(track_buffer))
    return IterableSimpleNamespace(**cfg)


def _class_key(det: dict[str, Any]) -> str:
    name = str(det.get("class_name") or det.get("label") or "object").strip().lower() or "object"
    try:
        cls_id = int(det.get("class_id")) if det.get("class_id") is not None else None
    except (TypeError, ValueError):
        cls_id = None
    if cls_id is not None:
        return f"{cls_id}:{name}"
    return name


def _bbox4(det: dict[str, Any]) -> list[float]:
    bbox = det.get("bbox") or [0, 0, 0, 0]
    if len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _CameraIdCounter:
    """Track-id source for one camera, shared by all its class slots."""

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value


def _slot_tracker(args: IterableSimpleNamespace, counter: _CameraIdCounter) -> BYTETracker:
    """BYTETracker whose tracks take ids from `counter`, never from the global STrack counter."""

    class _SlotSTrack(STrack):
        def next_id(self) -> int:  # type: ignore[override]  (STrack calls self.next_id())
            return counter.next()

    class _SlotBYTETracker(BYTETracker):
        track_class = _SlotSTrack

        @staticmethod
        def reset_id() -> None:  # never reset the shared global counter
            return None

    return _SlotBYTETracker(args=args)


class _DetResults:
    """Minimal Results-like wrapper for BYTETracker.update()."""

    __slots__ = ("xyxy", "conf", "cls", "_n")

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self._n = int(xyxy.shape[0]) if xyxy is not None else 0

    @property
    def xywh(self) -> np.ndarray:
        if self._n == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x1 = self.xyxy[:, 0]
        y1 = self.xyxy[:, 1]
        x2 = self.xyxy[:, 2]
        y2 = self.xyxy[:, 3]
        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        cx = x1 + w * 0.5
        cy = y1 + h * 0.5
        return np.stack([cx, cy, w, h], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, mask):
        return _DetResults(self.xyxy[mask], self.conf[mask], self.cls[mask])


def _empty_results() -> _DetResults:
    return _DetResults(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )


class CameraByteTrackerPool:
    """One BYTETracker per camera + object class (person, car, bag, …)."""

    def __init__(self, *, track_buffer: int = 90):
        self._track_buffer = track_buffer
        self._trackers: dict[tuple[str, str], BYTETracker] = {}
        self._counters: dict[str, _CameraIdCounter] = {}
        # Guards the dicts only; one camera is never updated by two threads at once.
        self._lock = threading.Lock()

    def reset(self, camera_key: str = "") -> None:
        with self._lock:
            if camera_key:
                key = (camera_key or "").strip()
                for k in [k for k in self._trackers if k[0] == key]:
                    self._trackers.pop(k, None)
                self._counters.pop(key, None)
            else:
                self._trackers.clear()
                self._counters.clear()

    @staticmethod
    def _cam(camera_key: str) -> str:
        return (camera_key or "").strip() or "_default"

    def _counter(self, cam: str) -> _CameraIdCounter:
        with self._lock:
            counter = self._counters.get(cam)
            if counter is None:
                counter = _CameraIdCounter()
                self._counters[cam] = counter
            return counter

    def _get(self, camera_key: str, class_key: str) -> BYTETracker:
        cam = self._cam(camera_key)
        slot = (cam, class_key)
        with self._lock:
            tracker = self._trackers.get(slot)
        if tracker is None:
            tracker = _slot_tracker(_byte_tracker_args(track_buffer=self._track_buffer), self._counter(cam))
            with self._lock:
                tracker = self._trackers.setdefault(slot, tracker)
        return tracker

    def _update_class(
        self,
        camera_key: str,
        class_key: str,
        items: list[tuple[int, dict[str, Any]]],
        frame: np.ndarray | None,
    ) -> dict[int, int]:
        """Return map of original detection index → track id (unique per camera)."""
        assigned: dict[int, int] = {}
        if not items:
            return assigned

        xyxy = np.asarray([_bbox4(det) for _, det in items], dtype=np.float32)
        conf = np.asarray([float(det.get("confidence") or 0.0) for _, det in items], dtype=np.float32)
        cls = np.zeros((len(items),), dtype=np.float32)
        for i, (_, det) in enumerate(items):
            try:
                cls[i] = float(int(det.get("class_id", 0)))
            except (TypeError, ValueError):
                cls[i] = 0.0

        tracker = self._get(camera_key, class_key)
        tracks = tracker.update(_DetResults(xyxy, conf, cls), frame)

        if tracks is not None and len(tracks) > 0:
            for row in np.asarray(tracks, dtype=np.float32):
                if row.shape[0] < 5:
                    continue
                track_id = int(row[4])
                idx = None
                if row.shape[0] >= 8:
                    raw_idx = int(row[7])
                    if 0 <= raw_idx < len(items):
                        idx = raw_idx
                if idx is None:
                    tb = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                    best_i, best_iou = -1, 0.25
                    for j, (orig, det) in enumerate(items):
                        if orig in assigned:
                            continue
                        score = _iou(_bbox4(det), tb)
                        if score > best_iou:
                            best_iou, best_i = score, j
                    if best_i >= 0:
                        idx = best_i
                if idx is None:
                    continue
                assigned[items[idx][0]] = track_id

        # Boxes ByteTrack did not confirm (low score / first frame): a fresh id from the same
        # per-camera counter, so it can never collide with a real track id on this camera.
        counter = self._counter(self._cam(camera_key))
        for orig, _det in items:
            if orig not in assigned:
                assigned[orig] = counter.next()
        return assigned

    def assign_track_ids(
        self,
        camera_key: str,
        detections: list[dict[str, Any]],
        frame: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Attach a ByteTrack id onto every detection, grouped by object class.

        Every existing class slot of this camera is updated each frame (with no boxes when
        the class is absent) so lost tracks age out instead of freezing.
        """
        cam = self._cam(camera_key)
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for i, det in enumerate(detections or []):
            grouped.setdefault(_class_key(det), []).append((i, det))

        with self._lock:
            idle = [k[1] for k in self._trackers if k[0] == cam and k[1] not in grouped]
        for class_key in idle:
            try:
                self._get(camera_key, class_key).update(_empty_results(), frame)
            except Exception:
                pass

        if not detections:
            return detections

        by_idx: dict[int, int] = {}
        for class_key, items in grouped.items():
            by_idx.update(self._update_class(camera_key, class_key, items, frame))

        out: list[dict[str, Any]] = []
        for i, det in enumerate(detections):
            enriched = dict(det)
            tid = by_idx.get(i)
            if tid is not None:
                enriched["track_id"] = tid
            else:
                enriched.pop("track_id", None)
            out.append(enriched)
        return out
