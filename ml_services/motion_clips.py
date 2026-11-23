"""Record a video clip whenever objects move on a live camera.

Inference (≈1 fps per camera under full load) only tells us WHEN objects start and stop
moving; the video itself is sampled straight from each camera's latest-frame buffer at
ML_MOTION_CLIP_FPS. Each camera keeps ML_MOTION_PREROLL_SEC of frames; a motion "start"
opens a recording that begins with that pre-roll and runs while anything on the camera
moves (+ post-roll), at least ML_MOTION_CLIP_MIN_SEC and at most ML_MOTION_CLIP_MAX_SEC.
Objects still moving at the cap continue in a new part.

Finished clips are written to ML_MOTION_CLIP_DIR as <id>.mp4 + <id>.jpg (poster, when the
motion was confirmed) + <id>.end.jpg (last frame) + <id>.json (written last = ready).
The backend pulls them via /live/motion/clips and deletes them here once stored.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from clip_pose import PoseBatch, dense_path, nearest_frame, run_pose, seed_pose, smooth_path, track_from_seed
from motion_path_draw import draw_motion_path, draw_skeleton

CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")

# camera key → stream object with get_latest() -> (frame | None, seq)
FrameSources = Callable[[], dict[str, Any]]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ClipConfig:
    enabled: bool = True
    fps: float = 8.0
    preroll_sec: float = 5.0
    postroll_sec: float = 2.0
    min_sec: float = 10.0
    max_sec: float = 30.0
    # Wait after a clip ends before a new one may start on that camera (0 = none)
    cooldown_sec: float = 0.0
    # Real-movement filter: a finished clip is kept only if some object moved at least
    # this far from A to B (% of the frame diagonal, same measure as the backend's
    # distance_pct). Small moves (people shifting in a chair) are left out of the log
    # and a clip with no real move is not written at all. 0 = keep everything.
    keep_min_pct: float = 5.0
    keep_person_pct: float = 3.0
    # Or came this much closer to / farther from the camera (box size ratio)
    keep_scale: float = 1.5
    # People with pose: keep when the body (hips) moved this many torso lengths A → B,
    # or the posture changed (sat down / stood up). Used instead of keep_person_pct.
    keep_body_shift: float = 1.0
    keep_posture_change: bool = True
    # Always kept, however small the move
    keep_always_classes: frozenset[str] = frozenset(
        {"weapon", "gun", "pistol", "rifle", "firearm", "knife_weapon", "heavy-weapon"}
    )
    # Danger objects: seeing one starts a clip at once (moving or not) and keeps it
    # recording while it is in view. Cooldown and the movement filter do not apply.
    danger_classes: frozenset[str] = frozenset(
        {"weapon", "gun", "pistol", "rifle", "firearm", "knife_weapon", "heavy-weapon", "fire", "smoke", "flame"}
    )
    # A danger object that stays where it was recorded is not recorded again until it
    # moves or has been out of view this long
    danger_rearm_sec: float = 600.0
    max_width: int = 1280
    jpeg_quality: int = 80
    crf: int = 28
    # Oldest clips are dropped when the backend has not collected them and this fills up
    max_pending_mb: float = 2048.0
    out_dir: str = ""

    @classmethod
    def from_env(cls) -> ClipConfig:
        # One folder per ML process: gpu0 and gpu1 run from the same directory, and a shared
        # folder made each node list, prune and delete the other node's clips.
        from live_metrics import ml_node_id

        node = ml_node_id()
        base = Path(__file__).resolve().parent / "motion_clips"
        default_dir = str(base / node if node else base)
        return cls(
            enabled=os.getenv("ML_MOTION_CLIPS_ENABLED", "true").strip().lower() in ("true", "1", "yes"),
            fps=min(25.0, max(1.0, _env_float("ML_MOTION_CLIP_FPS", 8.0))),
            preroll_sec=max(0.0, _env_float("ML_MOTION_PREROLL_SEC", 5.0)),
            postroll_sec=max(0.0, _env_float("ML_MOTION_POSTROLL_SEC", 2.0)),
            min_sec=max(1.0, _env_float("ML_MOTION_CLIP_MIN_SEC", 10.0)),
            max_sec=max(2.0, _env_float("ML_MOTION_CLIP_MAX_SEC", 30.0)),
            cooldown_sec=max(0.0, _env_float("ML_MOTION_CLIP_COOLDOWN_SEC", 0.0)),
            keep_min_pct=max(0.0, _env_float("ML_MOTION_KEEP_MIN_PCT", 5.0)),
            keep_person_pct=max(0.0, _env_float("ML_MOTION_KEEP_PERSON_PCT", 3.0)),
            keep_scale=max(1.0, _env_float("ML_MOTION_KEEP_SCALE", 1.5)),
            keep_body_shift=max(0.0, _env_float("ML_MOTION_KEEP_BODY_SHIFT", 1.0)),
            keep_posture_change=os.getenv("ML_MOTION_KEEP_POSTURE_CHANGE", "true").strip().lower()
            in ("true", "1", "yes"),
            keep_always_classes=frozenset(
                c.strip().lower()
                for c in os.getenv(
                    "ML_MOTION_KEEP_ALWAYS_CLASSES",
                    "weapon,gun,pistol,rifle,firearm,knife_weapon,heavy-weapon",
                ).split(",")
                if c.strip()
            ),
            danger_classes=frozenset(
                c.strip().lower()
                for c in os.getenv(
                    "ML_DANGER_CLIP_CLASSES",
                    "weapon,gun,pistol,rifle,firearm,knife_weapon,heavy-weapon,fire,smoke,flame",
                ).split(",")
                if c.strip()
            ),
            danger_rearm_sec=max(0.0, _env_float("ML_DANGER_REARM_SEC", 600.0)),
            max_width=max(320, int(_env_float("ML_MOTION_CLIP_WIDTH", 1280))),
            jpeg_quality=min(95, max(40, int(_env_float("ML_MOTION_CLIP_JPEG_QUALITY", 80)))),
            crf=min(40, max(18, int(_env_float("ML_MOTION_CLIP_CRF", 28)))),
            max_pending_mb=max(100.0, _env_float("ML_MOTION_CLIP_MAX_PENDING_MB", 2048.0)),
            out_dir=os.getenv("ML_MOTION_CLIP_DIR", "").strip() or default_dir,
        )


PERSON_CLASSES = frozenset({"person", "face"})
VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck", "bicycle", "vehicle"})
VEHICLE_BOX_COLOR = (0, 165, 255)  # orange (BGR)


def _box_center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _box_size(b: list[float]) -> float:
    return max(1.0, ((b[2] - b[0]) * (b[3] - b[1])) ** 0.5)


def _box_iou(a: list[float], b: list[float]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _rescale_det(det: dict[str, Any], k: float) -> dict[str, Any]:
    """Pose / box result from a downscaled image back to clip pixels."""
    out = dict(det)
    if det.get("bbox") is not None:
        out["bbox"] = [float(v) * k for v in det["bbox"][:4]]
    if det.get("keypoints"):
        out["keypoints"] = [[float(p[0]) * k, float(p[1]) * k, *p[2:]] for p in det["keypoints"]]
    return out


def move_pct(from_bbox: list[float], to_bbox: list[float], width: int, height: int) -> float | None:
    """Net move A → B as % of the frame diagonal (None = can't tell)."""
    diag = (width * width + height * height) ** 0.5 if width and height else 0.0
    if len(from_bbox) < 4 or len(to_bbox) < 4 or diag <= 0:
        return None
    (fx, fy), (tx, ty) = _box_center(from_bbox), _box_center(to_bbox)
    return 100.0 * ((tx - fx) ** 2 + (ty - fy) ** 2) ** 0.5 / diag


@dataclass
class _Movement:
    track_id: int
    global_object_id: str
    class_name: str
    started_at: float
    from_bbox: list[float]
    to_bbox: list[float]
    ended_at: float | None = None
    still_moving: bool = True
    # Danger object (weapon / fire / smoke): "still_moving" here means "still in view"
    danger: bool = False
    last_seen: float = 0.0
    # Lasting motion key ("g:GP12" or "t:<track>"); falls back to track_id
    key: Any = None
    # Pose (people): net hip move in torso lengths, posture before / after
    body_shift: float | None = None
    posture_from: str = ""
    posture_to: str = ""
    # Movement path [x, y, t] (source px) from motion_state: drawn on the clip frames
    path: list = field(default_factory=list)
    # Clip frames in which YOLO pose found this person (frame-by-frame tracking)
    pose_frames: int = 0

    def __post_init__(self):
        if self.key is None:
            self.key = self.track_id


@dataclass
class _Recording:
    camera_key: str
    started_at: float
    part: int
    frames: list[tuple[float, bytes]] = field(default_factory=list)
    movements: dict[int, _Movement] = field(default_factory=dict)
    poster_at: float = 0.0
    last_motion_at: float = 0.0
    source_size: tuple[int, int] = (0, 0)
    clip_size: tuple[int, int] = (0, 0)

    def moving(self) -> list[_Movement]:
        return [m for m in self.movements.values() if m.still_moving]


@dataclass
class _Camera:
    ring: deque[tuple[float, bytes]] = field(default_factory=deque)
    recording: _Recording | None = None
    cooldown_until: float = 0.0
    last_seq: int = -1
    new_frames: deque[float] = field(default_factory=lambda: deque(maxlen=400))
    source_size: tuple[int, int] = (0, 0)
    clip_size: tuple[int, int] = (0, 0)
    # danger class → (box where it was recorded, last time seen)
    danger_spots: dict[str, tuple[list[float], float]] = field(default_factory=dict)
    last_ring_at: float = 0.0


class MotionClipRecorder:
    """Sampler thread: motion events + camera frames → recordings. Encoder thread: MP4 files."""

    def __init__(
        self,
        config: ClipConfig | None = None,
        *,
        ffmpeg_path: Callable[[], str | None] | None = None,
        frame_sources: FrameSources | None = None,
        pose_batch: PoseBatch | None = None,
        vehicle_batch: PoseBatch | None = None,
    ):
        self.config = config or ClipConfig.from_env()
        # YOLO pose run on every clip frame after the clip ends (frame-by-frame skeleton + path)
        self._pose_batch = pose_batch
        self._clip_pose = os.getenv("ML_CLIP_POSE_ENABLED", "true").strip().lower() in ("true", "1", "yes")
        # YOLO vehicle detection on every clip frame (vehicles have no pose)
        self._vehicle_batch = vehicle_batch
        self._clip_vehicles = os.getenv("ML_CLIP_VEHICLE_TRACK_ENABLED", "true").strip().lower() in ("true", "1", "yes")
        # Idle pre-roll sample rate (frames kept while nothing records); default half the clip fps.
        try:
            pre_fps = float(os.getenv("ML_MOTION_PREROLL_FPS", "") or self.config.fps / 2.0)
        except (TypeError, ValueError):
            pre_fps = self.config.fps / 2.0
        self._preroll_interval = 1.0 / min(self.config.fps, max(0.5, pre_fps))
        # Clip pose / vehicle reprocessing: long clips are subsampled to at most this many
        # frames, and every frame is downscaled to this width for the detector.
        self._reprocess_max_frames = max(16, int(_env_float("ML_CLIP_REPROCESS_MAX_FRAMES", 160)))
        self._reprocess_width = max(320, int(_env_float("ML_CLIP_REPROCESS_WIDTH", 960)))
        self._ffmpeg_path = ffmpeg_path or (lambda: None)
        self._frame_sources = frame_sources or (lambda: {})
        self._dir = Path(self.config.out_dir)
        self._cameras: dict[str, _Camera] = {}
        self._lock = threading.Lock()
        self._events: queue.Queue = queue.Queue(maxsize=512)
        self._encode: queue.Queue = queue.Queue(maxsize=32)
        self._threads: list[threading.Thread] = []
        self._running = False
        self.stats = {
            "events_dropped": 0,
            "clips_written": 0,
            "clips_failed": 0,
            "clips_pruned": 0,
            "clips_skipped_small": 0,
            "clips_pose_tracked": 0,
            "clips_vehicle_tracked": 0,
        }

    # ── lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        if not self.config.enabled or self._running:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            self.sweep_orphans()
        except Exception as exc:
            print(f"[motion-clip] Orphan sweep failed: {exc}")
        self._running = True
        for target, name in ((self._sample_loop, "motion-clip-sampler"), (self._encode_loop, "motion-clip-encode")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)
        print(
            f"[motion-clip] Recording to {self._dir} ({self.config.fps} fps, pre-roll {self.config.preroll_sec}s, "
            f"{self.config.min_sec}-{self.config.max_sec}s per clip)"
        )

    def stop(self) -> None:
        self._running = False

    def reset(self, camera_key: str) -> None:
        with self._lock:
            self._cameras.pop(camera_key, None)

    # ── input from the inference loop (must stay cheap) ───────

    def submit(self, camera_key: str, detections: list[dict[str, Any]], events: list[dict[str, Any]], now: float) -> None:
        """Motion events + current boxes for one inferred frame."""
        if not self._running:
            return
        moving = [d for d in detections if d.get("motion_state") == "moving" and d.get("track_id") is not None]
        danger = [d for d in detections if self.danger_class(d)]
        if not events and not moving and not danger:
            return
        try:
            self._events.put_nowait((camera_key, moving, events, now, danger))
        except queue.Full:
            self.stats["events_dropped"] += 1

    # ── sampler thread ────────────────────────────────────────

    def _sample_loop(self) -> None:
        interval = 1.0 / self.config.fps
        next_at = time.monotonic()
        while self._running:
            self._drain_events()
            now = time.time()
            try:
                sources = self._frame_sources()
            except Exception:
                sources = {}
            for key, stream in sources.items():
                try:
                    frame, seq = stream.get_latest()
                except Exception:
                    continue
                if frame is None:
                    continue
                with self._lock:
                    cam = self._cameras.setdefault(key, _Camera())
                    if seq == cam.last_seq:
                        continue
                    # No recording on this camera: the pre-roll ring only needs
                    # preroll_fps (resize + JPEG per camera was the sampler's main cost).
                    # Gaps are padded with repeated frames when a recording starts.
                    if cam.recording is None and now - cam.last_ring_at < self._preroll_interval * 0.95:
                        continue
                    cam.last_seq = seq
                    cam.new_frames.append(now)
                try:
                    self.add_frame(key, frame, now)
                except Exception as exc:
                    print(f"[motion-clip] Frame error [{key}]: {exc}")
            self.check_all(time.time())
            next_at += interval
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_at = time.monotonic()  # fell behind: don't try to catch up

    def _drain_events(self) -> None:
        while True:
            try:
                camera_key, moving, events, now, danger = self._events.get_nowait()
            except queue.Empty:
                return
            self.on_motion(camera_key, moving, events, now, danger)

    def danger_class(self, det: dict[str, Any]) -> str:
        """Danger class name of a detection, or "" if it is not one."""
        cls = str(det.get("class_name") or det.get("label") or "").strip().lower()
        if cls in self.config.danger_classes:
            return cls
        tag = str(det.get("model") or det.get("model_tag") or "").strip().lower()
        if tag == "weapon" and "weapon" in self.config.danger_classes:
            return cls or "weapon"
        return ""

    def _danger_key(self, cls: str) -> int:
        # One entry per danger class per clip; negative so it never clashes with track ids
        return -1 - sorted(self.config.danger_classes | {cls}).index(cls)

    def _new_recording(self, cam: _Camera, camera_key: str, now: float) -> _Recording:
        # Motion began ~start_sec before it was confirmed; the ring covers the pre-roll
        rec = _Recording(
            camera_key=camera_key,
            started_at=now - self.config.preroll_sec,
            part=1,
            poster_at=now,
            last_motion_at=now,
        )
        cam.recording = rec
        return rec

    def on_motion(
        self,
        camera_key: str,
        moving_dets: list[dict[str, Any]],
        events: list[dict[str, Any]],
        now: float,
        danger_dets: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            cam = self._cameras.setdefault(camera_key, _Camera())
            rec = cam.recording
            for ev in events:
                rec = self._apply_event(cam, camera_key, ev, now)
            for det in danger_dets or []:
                cls = self.danger_class(det)
                if not cls:
                    continue
                bbox = [float(v) for v in (det.get("bbox") or [])[:4]]
                key = self._danger_key(cls)
                mv = rec.movements.get(key) if rec is not None else None
                active = mv is not None and mv.still_moving
                # Same danger object still at the spot it was recorded at → already has its clip.
                # It is recorded again only after it moves or has been gone danger_rearm_sec.
                spot = cam.danger_spots.get(cls)
                same_spot = (
                    spot is not None
                    and now - spot[1] <= self.config.danger_rearm_sec
                    and (not bbox or not spot[0] or _box_iou(spot[0], bbox) >= 0.3)
                )
                if det.get("stale") and not active:
                    # Box reused from an earlier specialist run (weapon / fire models run every
                    # Nth inference): it may keep a running danger clip going, never start one.
                    continue
                cam.danger_spots[cls] = (spot[0] if same_spot else bbox, now)
                if same_spot and not active:
                    continue
                if rec is None:
                    rec = self._new_recording(cam, camera_key, now)  # no cooldown for danger
                if not active:
                    rec.movements[key] = _Movement(
                        track_id=key, global_object_id="", class_name=cls, started_at=now,
                        from_bbox=bbox, to_bbox=bbox, danger=True, last_seen=now,
                    )
                else:
                    mv.last_seen = now
                    if bbox:
                        mv.to_bbox = bbox
                rec.last_motion_at = now
            if rec is None:
                return
            by_key = {(d.get("motion_key") or int(d["track_id"])): d for d in moving_dets}
            for mv in rec.moving():
                if mv.danger:
                    continue
                det = by_key.get(mv.key)
                if det and det.get("bbox"):
                    mv.to_bbox = [float(v) for v in det["bbox"][:4]]
                if det and det.get("motion_path"):
                    mv.path = det["motion_path"]
            if rec.moving():
                rec.last_motion_at = now

    def add_frame(self, camera_key: str, frame: np.ndarray, now: float) -> None:
        cfg = self.config
        src_h, src_w = frame.shape[:2]
        scale = min(1.0, cfg.max_width / float(src_w)) if src_w else 1.0
        if scale < 1.0:
            img = cv2.resize(frame, (int(src_w * scale) // 2 * 2, int(src_h * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
        else:
            img = frame[: src_h // 2 * 2, : src_w // 2 * 2]  # libx264 / yuv420p need even sizes
        ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        if not ok:
            return
        data = jpeg.tobytes()
        with self._lock:
            cam = self._cameras.setdefault(camera_key, _Camera())
            cam.source_size = (src_w, src_h)
            cam.clip_size = (img.shape[1], img.shape[0])
            cam.ring.append((now, data))
            cam.last_ring_at = now
            while cam.ring and now - cam.ring[0][0] > cfg.preroll_sec:
                cam.ring.popleft()
            rec = cam.recording
            if rec is not None:
                if not rec.frames:
                    # First frame of a new recording: seed it with the pre-roll
                    rec.frames = self._padded([f for f in cam.ring if f[0] >= rec.started_at])
                else:
                    rec.frames.append((now, data))
                rec.source_size, rec.clip_size = cam.source_size, cam.clip_size
            self._check_camera(camera_key, cam, now)

    def _padded(self, frames: list[tuple[float, bytes]]) -> list[tuple[float, bytes]]:
        """Repeat frames across gaps so the pre-roll (sampled slower) plays at the clip fps."""
        step = 1.0 / self.config.fps
        out: list[tuple[float, bytes]] = []
        for ts, data in frames:
            if out:
                prev_ts, prev_data = out[-1]
                t = prev_ts + step
                while ts - t > step * 0.5:
                    out.append((t, prev_data))
                    t += step
            out.append((ts, data))
        return out

    def _apply_event(self, cam: _Camera, camera_key: str, ev: dict[str, Any], now: float) -> _Recording | None:
        """Caller holds self._lock."""
        rec = cam.recording
        tid = int(ev.get("track_id") or 0)
        key = ev.get("motion_key") or tid
        kind = ev.get("event")
        if kind == "start":
            if rec is None:
                if now < cam.cooldown_until:
                    return None
                rec = self._new_recording(cam, camera_key, now)
            if key not in rec.movements or not rec.movements[key].still_moving:
                rec.movements[key] = _Movement(
                    track_id=tid,
                    key=key,
                    global_object_id=str(ev.get("global_object_id") or ""),
                    class_name=str(ev.get("class_name") or ""),
                    started_at=float(ev.get("at") or now),
                    from_bbox=list(ev.get("from_bbox") or []),
                    to_bbox=list(ev.get("to_bbox") or []),
                    posture_from=str(ev.get("posture_from") or ""),
                )
            rec.last_motion_at = now
        elif rec is not None and kind in ("stop", "lost") and key in rec.movements:
            mv = rec.movements[key]
            mv.track_id = tid
            mv.body_shift = ev.get("body_shift")
            mv.posture_to = str(ev.get("posture_to") or "")
            mv.path = ev.get("path") or mv.path
            mv.still_moving = False
            mv.ended_at = float(ev.get("at") or now)
            mv.to_bbox = list(ev.get("to_bbox") or mv.to_bbox)
            if ev.get("global_object_id"):
                mv.global_object_id = str(ev["global_object_id"])
        return rec

    # ── finishing recordings ──────────────────────────────────

    def check_all(self, now: float) -> None:
        with self._lock:
            for key, cam in self._cameras.items():
                self._check_camera(key, cam, now)

    def _check_camera(self, camera_key: str, cam: _Camera, now: float) -> None:
        """Caller holds self._lock."""
        rec = cam.recording
        if rec is None or not rec.frames:
            return
        cfg = self.config
        for mv in rec.moving():
            # Danger object out of view for the post-roll: its part of the clip is over
            if mv.danger and now - mv.last_seen > cfg.postroll_sec:
                mv.still_moving = False
                mv.ended_at = mv.last_seen
        # Measured from the first frame actually recorded (the ring may be shorter than the pre-roll)
        length = now - rec.frames[0][0]
        capped = length >= cfg.max_sec
        quiet = not rec.moving() and now - rec.last_motion_at >= cfg.postroll_sec and length >= cfg.min_sec
        if not (capped or quiet):
            return
        cam.recording = None
        cam.cooldown_until = now + cfg.cooldown_sec
        for mv in rec.moving():
            # A danger object gets one clip per sighting; it is not continued into more parts
            # (a weapon left on a desk would otherwise record forever). If it moves, the
            # moved spot starts a new clip.
            if mv.danger:
                mv.still_moving = False
                mv.ended_at = mv.last_seen
        still = rec.moving()
        if capped and still:
            # Same movement continues in the next part (no cooldown between parts)
            nxt = _Recording(camera_key=camera_key, started_at=now, part=rec.part + 1, poster_at=now, last_motion_at=now)
            for mv in still:
                nxt.movements[mv.key] = _Movement(
                    track_id=mv.track_id,
                    key=mv.key,
                    global_object_id=mv.global_object_id,
                    class_name=mv.class_name,
                    started_at=now,
                    from_bbox=list(mv.to_bbox),
                    to_bbox=list(mv.to_bbox),
                    danger=mv.danger,
                    last_seen=mv.last_seen,
                )
            cam.recording = nxt
            cam.cooldown_until = 0.0
        real = self.real_movements(rec)
        if not real:
            self.stats["clips_skipped_small"] += 1
            return
        rec.movements = {mv.key: mv for mv in real}
        try:
            self._encode.put_nowait(rec)
        except queue.Full:
            self.stats["clips_failed"] += 1
            print(f"[motion-clip] Encode queue full — dropped clip [{camera_key}]")

    def is_real_move(self, mv: _Movement, width: int, height: int) -> bool:
        cfg = self.config
        cls = (mv.class_name or "").strip().lower()
        if mv.danger or cls in cfg.keep_always_classes:
            return True
        if cls in PERSON_CLASSES and mv.body_shift is not None:
            # Pose available: judge by the body, not the box (hands / leaning do not count)
            if cfg.keep_posture_change and mv.posture_from and mv.posture_to and mv.posture_from != mv.posture_to:
                return True
            return float(mv.body_shift) >= cfg.keep_body_shift
        min_pct = cfg.keep_person_pct if cls in PERSON_CLASSES else cfg.keep_min_pct
        if min_pct <= 0:
            return True
        pct = move_pct(mv.from_bbox, mv.to_bbox, width, height)
        if pct is None:
            return True  # can't measure → keep rather than lose evidence
        if pct >= min_pct:
            return True
        # Walked toward / away from the camera: little shift on screen, big size change
        grow = _box_size(mv.to_bbox) / _box_size(mv.from_bbox)
        return grow >= cfg.keep_scale or grow <= 1.0 / cfg.keep_scale

    def real_movements(self, rec: _Recording) -> list[_Movement]:
        """Movements in a finished recording that pass the real-movement filter."""
        w, h = rec.source_size
        return [mv for mv in rec.movements.values() if self.is_real_move(mv, w, h)]

    # ── encode thread: MP4 + poster + metadata ────────────────

    def _encode_loop(self) -> None:
        while self._running:
            try:
                rec = self._encode.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.write_clip(rec)
                self._prune()
            except Exception as exc:
                self.stats["clips_failed"] += 1
                print(f"[motion-clip] Write failed [{rec.camera_key}]: {exc}")

    def _with_paths(self, rec: _Recording, frames: list[tuple[float, bytes]]) -> list[tuple[float, bytes]]:
        """Draw movement on the clip frames, tracked frame by frame.

        People: YOLO pose on EVERY frame → skeleton per frame + dense hip path.
        Vehicles: YOLO vehicle detection on EVERY frame → box per frame + dense wheel path.
        Others (or model unavailable): the path from live inference.
        Paths grow with time: A at the start, arrow at the current position.
        """
        moves = [mv for mv in rec.movements.values() if not mv.danger]
        if not moves or not rec.source_size[0] or not rec.clip_size[0]:
            return frames
        scale = rec.clip_size[0] / float(rec.source_size[0])
        thickness = max(2, int(round(rec.clip_size[1] / 240)))
        times = [ts for ts, _data in frames]
        # Detector input: at most _reprocess_max_frames evenly spaced frames, downscaled to
        # _reprocess_width, decoded chunk by chunk (a 30 s clip used to be decoded into
        # memory in full, ~240 full-size images, and run through the GPU frame by frame).
        n = len(frames)
        step = max(1, -(-n // self._reprocess_max_frames))
        sample_idx = list(range(0, n, step))
        if sample_idx[-1] != n - 1:
            sample_idx.append(n - 1)
        sample_times = [times[i] for i in sample_idx]
        # nearest sampled frame at or before each clip frame (for drawing every frame)
        sample_of = [0] * n
        si = 0
        for fi in range(n):
            while si + 1 < len(sample_idx) and sample_idx[si + 1] <= fi:
                si += 1
            sample_of[fi] = si

        tracks: list[tuple[str, list[dict[str, Any] | None]]] = []  # ("pose" | "box", track per sample)
        drawn_paths: list[tuple[list, float]] = []  # (path, scale)
        handled: set[int] = set()  # id() of movements already drawn

        groups = (
            ("pose", PERSON_CLASSES, self._pose_batch if self._clip_pose else None),
            ("box", VEHICLE_CLASSES, self._vehicle_batch if self._clip_vehicles else None),
        )
        for kind, classes, batch_fn in groups:
            group = [mv for mv in moves if (mv.class_name or "").lower() in classes and len(mv.from_bbox) >= 4]
            if not group or batch_fn is None:
                continue
            try:
                found = self._detect_sampled(frames, sample_idx, batch_fn)
                if found is None:
                    continue
                taken: list[set[int]] = [set() for _ in sample_idx]
                tracked_any = False
                for mv in group:
                    track = self._track_mover(mv, found, sample_times, scale, taken)
                    path = smooth_path(dense_path(track, mv.started_at)) if track else []
                    if len(path) >= 2:
                        tracks.append((kind, track))
                        drawn_paths.append((path, 1.0))
                        handled.add(id(mv))
                        tracked_any = True
                        mv.pose_frames = sum(1 for fi in range(n) if track[sample_of[fi]] is not None)
                        # Dense path in source pixels replaces the sparse live one
                        mv.path = [[round(x / scale, 1), round(y / scale, 1), t] for x, y, t in path]
                if tracked_any:
                    self.stats["clips_pose_tracked" if kind == "pose" else "clips_vehicle_tracked"] += 1
            except Exception as exc:
                print(f"[motion-clip] Frame-by-frame {kind} tracking failed ({exc}) — using live path")
        for mv in moves:
            if id(mv) not in handled and len(mv.path or []) >= 2:
                drawn_paths.append((mv.path, scale))
        if not drawn_paths and not tracks:
            return frames

        out = []
        for fi, (ts, data) in enumerate(frames):
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # one frame at a time
            if img is None:
                out.append((ts, data))
                continue
            for path, sc in drawn_paths:
                draw_motion_path(img, path, thickness, scale=sc, until=ts)
            for kind, track in tracks:
                pos = sample_of[fi]
                item = track[pos] if pos < len(track) else None
                if item is None:
                    continue
                if kind == "pose" and item["pose"].get("keypoints"):
                    draw_skeleton(img, item["pose"]["keypoints"], max(1, thickness - 1))
                else:
                    x1, y1, x2, y2 = (int(v) for v in item["pose"]["bbox"][:4])
                    cv2.rectangle(img, (x1, y1), (x2, y2), VEHICLE_BOX_COLOR, max(1, thickness - 1), cv2.LINE_AA)
            ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
            out.append((ts, jpeg.tobytes() if ok else data))
        return out

    def _detect_sampled(
        self, frames: list[tuple[float, bytes]], sample_idx: list[int], batch_fn: PoseBatch, chunk: int = 8
    ) -> list[list[dict[str, Any]]] | None:
        """Run batch_fn over the sampled frames (downscaled, decoded per chunk).

        Results come back in clip-frame pixels. None when a frame cannot be decoded.
        """
        found: list[list[dict[str, Any]]] = []
        for c in range(0, len(sample_idx), chunk):
            imgs: list[np.ndarray] = []
            factors: list[float] = []
            for i in sample_idx[c : c + chunk]:
                img = cv2.imdecode(np.frombuffer(frames[i][1], np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    return None
                h, w = img.shape[:2]
                f = 1.0
                if w > self._reprocess_width:
                    f = self._reprocess_width / float(w)
                    img = cv2.resize(img, (self._reprocess_width, max(2, int(h * f))), interpolation=cv2.INTER_AREA)
                imgs.append(img)
                factors.append(f)
            res = run_pose(imgs, batch_fn, batch_size=chunk)
            for dets, f in zip(res, factors):
                found.append(dets if f == 1.0 else [_rescale_det(d, 1.0 / f) for d in dets])
        return found

    @staticmethod
    def _track_mover(mv: _Movement, poses: list, times: list[float], scale: float, taken: list[set[int]]):
        """Find the moving person in the clip frames and follow them through every frame.

        Seeded at the END position B (where the person is now — the most reliable box) and
        followed backward and forward. Live tracking sees a camera only every few seconds and
        can put the start box A on the wrong person; frame-by-frame pose corrects that.
        Falls back to the start box when nobody is found at B.
        """
        for box, t in ((mv.to_bbox, mv.ended_at), (mv.from_bbox, mv.started_at)):
            if len(box or []) < 4:
                continue
            fi = nearest_frame(times, t)
            for f in (fi, fi - 1, fi + 1, fi - 2, fi + 2):
                if 0 <= f < len(poses):
                    j = seed_pose(poses[f], [v * scale for v in box[:4]], exclude=taken[f])
                    if j is not None:
                        return track_from_seed(poses, times, f, j, taken=taken)
        return None

    def write_clip(self, rec: _Recording) -> None:
        frames = rec.frames
        if len(frames) < 2:
            return
        frames = self._with_paths(rec, frames)
        ffmpeg = self._ffmpeg_path()
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found")
        duration = max(0.5, frames[-1][0] - frames[0][0])
        fps = max(1.0, (len(frames) - 1) / duration)
        safe_key = re.sub(r"[^A-Za-z0-9_-]", "_", rec.camera_key)
        clip_id = f"{safe_key}-{int(frames[0][0] * 1000)}-{uuid.uuid4().hex[:6]}"
        mp4 = self._dir / f"{clip_id}.mp4"
        tmp = self._dir / f"{clip_id}.part.mp4"
        cmd = [
            ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "image2pipe", "-framerate", f"{fps:.3f}", "-c:v", "mjpeg", "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.config.crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(tmp),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for _ts, data in frames:
                proc.stdin.write(data)
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "replace")
            code = proc.wait(timeout=120)
        except Exception:
            proc.kill()
            tmp.unlink(missing_ok=True)
            raise
        finally:
            proc.stderr.close()
        if code != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg exit {code}: {err.strip()[:300]}")
        tmp.replace(mp4)

        poster = min(frames, key=lambda f: abs(f[0] - rec.poster_at))[1]
        (self._dir / f"{clip_id}.jpg").write_bytes(poster)
        (self._dir / f"{clip_id}.end.jpg").write_bytes(frames[-1][1])

        movements = [
            {
                "track_id": None if mv.danger else mv.track_id,
                "danger": mv.danger,
                "global_object_id": mv.global_object_id,
                "class_name": mv.class_name,
                "started_at": mv.started_at,
                "ended_at": mv.ended_at,
                "still_moving": mv.still_moving,
                "from_bbox": [round(v, 1) for v in mv.from_bbox],
                "to_bbox": [round(v, 1) for v in mv.to_bbox],
                "body_shift": mv.body_shift,
                "posture_from": mv.posture_from,
                "posture_to": mv.posture_to,
                "pose_frames": mv.pose_frames,
                "path": [[round(p[0], 1), round(p[1], 1), round(p[2], 2)] for p in (mv.path or [])][:400],
            }
            for mv in rec.movements.values()
        ]
        meta = {
            "clip_id": clip_id,
            "camera_key": rec.camera_key,
            "part": rec.part,
            "started_at": frames[0][0],
            "ended_at": frames[-1][0],
            "duration_sec": round(duration, 2),
            "frame_count": len(frames),
            "fps": round(fps, 2),
            "clip_width": rec.clip_size[0],
            "clip_height": rec.clip_size[1],
            "source_width": rec.source_size[0],
            "source_height": rec.source_size[1],
            "size_bytes": mp4.stat().st_size,
            "trigger": "danger" if any(m.danger for m in rec.movements.values()) else "motion",
            "movements": movements,
        }
        tmp_meta = self._dir / f"{clip_id}.json.tmp"
        tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
        tmp_meta.replace(self._dir / f"{clip_id}.json")
        self.stats["clips_written"] += 1
        objs = ", ".join(m["global_object_id"] or m["class_name"] for m in movements)
        print(f"[motion-clip] {rec.camera_key} saved {duration:.1f}s clip ({len(frames)} frames) — {objs}")

    _ORPHAN_SUFFIXES = (".part.mp4", ".json.tmp", ".end.jpg", ".mp4", ".jpg")

    def sweep_orphans(self, max_age_sec: float = 600.0, now: float | None = None) -> int:
        """Delete clip files whose <id>.json never appeared (crash mid-write), once old enough."""
        if not self._dir.is_dir():
            return 0
        now = time.time() if now is None else now
        removed = 0
        for path in self._dir.iterdir():
            if not path.is_file():
                continue
            name = path.name
            suffix = next((s for s in self._ORPHAN_SUFFIXES if name.endswith(s)), None)
            if suffix is None:
                continue
            clip_id = name[: -len(suffix)]
            if suffix != ".json.tmp" and (self._dir / f"{clip_id}.json").is_file():
                continue
            try:
                if now - path.stat().st_mtime < max_age_sec:
                    continue
                path.unlink()
                removed += 1
            except OSError:
                continue
        if removed:
            self.stats["orphans_removed"] = self.stats.get("orphans_removed", 0) + removed
        return removed

    def _prune(self) -> None:
        self.sweep_orphans()
        limit = self.config.max_pending_mb * 1024 * 1024
        clips = sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in self._dir.glob("*") if p.is_file())
        while clips and total > limit:
            oldest = clips.pop(0)
            total -= self.delete_clip(oldest.stem)
            self.stats["clips_pruned"] += 1

    # ── pull API helpers ──────────────────────────────────────

    def _path(self, clip_id: str, suffix: str) -> Path | None:
        if not CLIP_ID_RE.match(clip_id or ""):
            return None
        return self._dir / f"{clip_id}{suffix}"

    def list_ready(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self._dir.is_dir():
            return []
        out = []
        for meta in sorted(self._dir.glob("*.json"), key=lambda p: p.stat().st_mtime)[: max(1, limit)]:
            try:
                out.append(json.loads(meta.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def file_path(self, clip_id: str, kind: str) -> Path | None:
        suffix = {"video": ".mp4", "poster": ".jpg", "end": ".end.jpg"}.get(kind)
        path = self._path(clip_id, suffix) if suffix else None
        return path if path is not None and path.is_file() else None

    def delete_clip(self, clip_id: str) -> int:
        freed = 0
        for suffix in (".json", ".mp4", ".jpg", ".end.jpg"):
            path = self._path(clip_id, suffix)
            if path is not None and path.is_file():
                freed += path.stat().st_size
                path.unlink(missing_ok=True)
        return freed

    def status(self) -> dict[str, Any]:
        with self._lock:
            recording = {
                key: {
                    "since": round(cam.recording.started_at, 2),
                    "part": cam.recording.part,
                    "frames": len(cam.recording.frames),
                    "moving": [m.global_object_id or m.class_name for m in cam.recording.moving()],
                }
                for key, cam in self._cameras.items()
                if cam.recording is not None
            }
            buffered = {key: len(cam.ring) for key, cam in self._cameras.items()}
            t = time.time()
            sampled = {
                key: round(sum(1 for x in cam.new_frames if t - x <= 30) / 30, 2) for key, cam in self._cameras.items()
            }
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "fps": self.config.fps,
            "dir": str(self._dir),
            "pending_clips": len(list(self._dir.glob("*.json"))) if self._dir.is_dir() else 0,
            "preroll_frames": buffered,
            "new_frames_per_sec": sampled,
            "recording": recording,
            "event_queue": self._events.qsize(),
            **self.stats,
        }
