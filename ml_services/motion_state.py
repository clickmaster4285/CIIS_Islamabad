"""Per-object moving / stationary state on live cameras (boxes, plus body pose for people).

An object is "moving" once it has left its resting position (anchor) for
ML_MOTION_START_SEC, and "stationary" again after ML_MOTION_STOP_SEC without leaving a
small still-zone.

* Objects: measured on the box centre (shift as a fraction of box size).
* People with pose keypoints: measured on the body point (hip centre, else shoulders)
  as a fraction of torso length. Hands, heads and leaning in a chair do not move the
  hips, so they no longer count as movement. Each person also gets a posture:
  "sitting" / "standing" (from thigh angle) and "walking" while moving.

Objects are followed by their lasting global ID (GP12, GV3…) when ML has one, so a
movement survives the local tracker losing the object and giving it a new track ID.
Start / stop events feed the clip recorder and /live/motion/status.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any

MOVING = "moving"
STATIONARY = "stationary"
PERSON_CLASSES = frozenset({"person"})
PATH_MAX_POINTS = 300
PATH_KEEP_SEC = 5.0

SITTING = "sitting"
STANDING = "standing"
WALKING = "walking"
UNKNOWN = ""

# COCO keypoint indices
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE = 5, 6, 11, 12, 13, 14


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class MotionConfig:
    enabled: bool = True
    # Center shift (fraction of box size) or size change that counts as leaving the anchor
    move_frac: float = 0.15
    scale_frac: float = 0.30
    # Must stay away from the anchor this long before it is "moving" (filters jitter)
    start_sec: float = 1.0
    # Still-zone while moving; no exit from it for stop_sec → "stationary"
    still_frac: float = 0.05
    stop_sec: float = 3.0
    # Forget a track not seen for this long (ends its motion as "lost"). Never less than
    # track_ttl_frames × the camera's real time between inferences.
    track_ttl_sec: float = 10.0
    track_ttl_frames: float = 3.0
    # People with pose: body point (hips) shift as a fraction of torso length
    pose_move_frac: float = 0.5
    pose_still_frac: float = 0.15
    pose_kpt_conf: float = 0.3
    log_events: bool = True
    # Parts of another object (face, plate) or shapes that never stand still (smoke, fire)
    exclude_classes: frozenset[str] = frozenset({"face", "license_plate", "smoke", "fire", "flame"})

    @classmethod
    def from_env(cls) -> MotionConfig:
        return cls(
            enabled=os.getenv("ML_MOTION_ENABLED", "true").strip().lower() in ("true", "1", "yes"),
            move_frac=max(0.01, _env_float("ML_MOTION_MOVE_FRAC", 0.15)),
            scale_frac=max(0.05, _env_float("ML_MOTION_SCALE_FRAC", 0.30)),
            start_sec=max(0.0, _env_float("ML_MOTION_START_SEC", 1.0)),
            still_frac=max(0.005, _env_float("ML_MOTION_STILL_FRAC", 0.05)),
            stop_sec=max(0.5, _env_float("ML_MOTION_STOP_SEC", 3.0)),
            track_ttl_sec=max(1.0, _env_float("ML_MOTION_TRACK_TTL_SEC", 10.0)),
            track_ttl_frames=max(1.0, _env_float("ML_MOTION_TRACK_TTL_FRAMES", 3.0)),
            pose_move_frac=max(0.05, _env_float("ML_MOTION_POSE_MOVE_FRAC", 0.5)),
            pose_still_frac=max(0.02, _env_float("ML_MOTION_POSE_STILL_FRAC", 0.15)),
            pose_kpt_conf=min(0.95, max(0.05, _env_float("ML_MOTION_POSE_KPT_CONF", 0.3))),
            log_events=os.getenv("ML_MOTION_LOG", "true").strip().lower() in ("true", "1", "yes"),
            exclude_classes=frozenset(
                c.strip().lower()
                for c in os.getenv("ML_MOTION_EXCLUDE_CLASSES", "face,license_plate,smoke,fire,flame").split(",")
                if c.strip()
            ),
        )


def _bbox4(det: dict[str, Any]) -> list[float] | None:
    bbox = det.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _size(b: list[float]) -> float:
    return max(1.0, math.sqrt((b[2] - b[0]) * (b[3] - b[1])))


def _ground(b: list[float]) -> tuple[float, float]:
    """Bottom centre of the box: where a vehicle's wheels / an object's base touch the floor."""
    return (b[0] + b[2]) / 2.0, b[3]


def _shift(ref: list[float], cur: list[float], *, ground: bool = False) -> tuple[float, float]:
    """(point shift / ref size, |size change| / ref size).

    ground=True (vehicles, objects): measured at the box's bottom centre. When parked
    objects overlap, the top of a box jumps as it gets hidden / uncovered, but the point
    on the floor stays put — so overlap no longer looks like movement.
    """
    point = _ground if ground else _center
    (rx, ry), (cx, cy) = point(ref), point(cur)
    size = _size(ref)
    return math.hypot(cx - rx, cy - ry) / size, abs(_size(cur) - size) / size


# ── pose helpers ─────────────────────────────────────────────


def _kp(kps: list, i: int, min_conf: float) -> tuple[float, float] | None:
    try:
        x, y, c = kps[i][0], kps[i][1], (kps[i][2] if len(kps[i]) > 2 else 1.0)
    except (IndexError, TypeError):
        return None
    return (float(x), float(y)) if float(c) >= min_conf else None


def _mid(a, b):
    if a and b:
        return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    return a or b


def body_point(kps: list | None, min_conf: float = 0.3) -> tuple[float, float] | None:
    """Hip centre (moves only when the whole body moves); shoulders if hips are hidden."""
    if not kps or len(kps) < 17:
        return None
    hips = _mid(_kp(kps, L_HIP, min_conf), _kp(kps, R_HIP, min_conf))
    if hips:
        return hips
    return _mid(_kp(kps, L_SHOULDER, min_conf), _kp(kps, R_SHOULDER, min_conf))


def torso_length(kps: list | None, bbox: list[float], min_conf: float = 0.3) -> float:
    """Shoulder-centre to hip-centre distance (body scale); 30% of box height if not visible."""
    if kps and len(kps) >= 17:
        sh = _mid(_kp(kps, L_SHOULDER, min_conf), _kp(kps, R_SHOULDER, min_conf))
        hp = _mid(_kp(kps, L_HIP, min_conf), _kp(kps, R_HIP, min_conf))
        if sh and hp:
            d = math.hypot(sh[0] - hp[0], sh[1] - hp[1])
            if d > 4:
                return d
    return max(4.0, 0.3 * (bbox[3] - bbox[1]))


def posture(kps: list | None, min_conf: float = 0.3) -> str:
    """"standing" when thighs point down, "sitting" when they are near horizontal, else ""."""
    if not kps or len(kps) < 17:
        return UNKNOWN
    sh = _mid(_kp(kps, L_SHOULDER, min_conf), _kp(kps, R_SHOULDER, min_conf))
    hp = _mid(_kp(kps, L_HIP, min_conf), _kp(kps, R_HIP, min_conf))
    torso = math.hypot(sh[0] - hp[0], sh[1] - hp[1]) if sh and hp else 0.0
    ratios, lengths = [], []
    for hip_i, knee_i in ((L_HIP, L_KNEE), (R_HIP, R_KNEE)):
        hip, knee = _kp(kps, hip_i, min_conf), _kp(kps, knee_i, min_conf)
        if hip and knee:
            length = math.hypot(knee[0] - hip[0], knee[1] - hip[1])
            if length > 4:
                ratios.append((knee[1] - hip[1]) / length)  # 1 = straight down, 0 = horizontal
                lengths.append(length)
    if not ratios or torso <= 4:
        return UNKNOWN
    r = sum(ratios) / len(ratios)
    # Thigh length vs torso: a standing thigh is about as long as the torso. A seated thigh
    # pointing at a high camera looks vertical but short (foreshortened) → not "standing".
    thigh = (sum(lengths) / len(lengths)) / torso
    if r >= 0.75 and thigh >= 0.6:
        return STANDING
    if r <= 0.5 or thigh < 0.45:
        return SITTING
    return UNKNOWN


@dataclass
class _Track:
    key: str
    track_id: int
    class_name: str
    anchor: list[float]
    last_bbox: list[float]
    last_seen: float
    state: str = STATIONARY
    state_since: float = 0.0
    global_object_id: str = ""
    # Candidate start: first time the object left the anchor (cleared if it returns)
    away_since: float | None = None
    # While moving: still-zone reference and when the object last left it
    still_ref: list[float] = field(default_factory=list)
    last_move_at: float = 0.0
    motion_from: list[float] = field(default_factory=list)
    # Pose (people): body point at rest / still-zone / movement start, and body scale
    body_anchor: tuple[float, float] | None = None
    body_still: tuple[float, float] | None = None
    body_from: tuple[float, float] | None = None
    body_last: tuple[float, float] | None = None
    torso: float = 0.0
    posture: str = UNKNOWN  # last known sitting / standing
    posture_from: str = UNKNOWN
    # Path while moving: [x, y, t] of the body point (people) or box centre; kept for
    # PATH_KEEP_SEC after the stop so the trail stays on screen briefly
    path: list[list[float]] = field(default_factory=list)
    path_until: float = 0.0


class CameraMotionTracker:
    """Motion state for every object on one camera. Not thread-safe; owner serializes calls."""

    def __init__(self, camera_key: str, config: MotionConfig):
        self.camera_key = camera_key
        self.config = config
        self._tracks: dict[str, _Track] = {}
        self._update_times: deque[float] = deque(maxlen=200)

    def _ttl(self) -> float:
        """Forget-after time: never shorter than a few real frame gaps on this camera."""
        times = list(self._update_times)[-20:]
        gaps = [b - a for a, b in zip(times, times[1:]) if b > a]
        gap = median(gaps) if gaps else 0.0
        return max(self.config.track_ttl_sec, self.config.track_ttl_frames * gap)

    def _track_for(self, det: dict[str, Any], tid: int, bbox: list[float], now: float) -> _Track:
        gid = str(det.get("global_object_id") or "").strip()
        tkey, gkey = f"t:{tid}", (f"g:{gid}" if gid else "")
        tr = self._tracks.get(gkey) if gkey else None
        if tr is None:
            tr = self._tracks.get(tkey)
            if tr is not None and gkey:
                # Track learned its lasting ID: follow it by that ID from now on
                self._tracks.pop(tkey, None)
                tr.key = gkey
                self._tracks[gkey] = tr
        if tr is None:
            key = gkey or tkey
            tr = _Track(
                key=key,
                track_id=tid,
                class_name=str(det.get("class_name") or det.get("label") or "object").strip().lower(),
                anchor=bbox,
                last_bbox=bbox,
                last_seen=now,
                state_since=now,
            )
            self._tracks[key] = tr
        tr.track_id = tid  # a broken track keeps the movement, under its new local ID
        if gid:
            tr.global_object_id = gid
        return tr

    def update(self, detections: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
        """Annotate detections with motion_state / motion_since / posture; return start/stop/lost events."""
        cfg = self.config
        self._update_times.append(now)
        events: list[dict[str, Any]] = []
        seen: set[str] = set()
        for det in detections:
            tid = det.get("track_id")
            bbox = _bbox4(det)
            if tid is None or bbox is None:
                continue
            if str(det.get("class_name") or det.get("label") or "").strip().lower() in cfg.exclude_classes:
                continue
            tr = self._track_for(det, int(tid), bbox, now)
            if tr.key in seen:
                continue  # two detections claim one lasting ID this frame: keep the first
            seen.add(tr.key)
            tr.last_bbox = bbox
            tr.last_seen = now

            kps = det.get("keypoints")
            body = body_point(kps, cfg.pose_kpt_conf)
            pose_now = posture(kps, cfg.pose_kpt_conf)
            posture_before = tr.posture
            if pose_now:
                tr.posture = pose_now
            if body is not None:
                tr.torso = torso_length(kps, bbox, cfg.pose_kpt_conf)
                tr.body_last = body
                if tr.body_anchor is None:
                    tr.body_anchor = body
            use_pose = body is not None and tr.body_anchor is not None

            if tr.state == STATIONARY:
                if use_pose:
                    away = _dist(tr.body_anchor, body) / tr.torso > cfg.pose_move_frac
                elif tr.body_anchor is not None:
                    away = None  # person tracked by pose, pose missed this frame: no decision
                else:
                    shift, scale = _shift(tr.anchor, bbox, ground=tr.class_name not in PERSON_CLASSES)
                    away = shift > cfg.move_frac or scale > cfg.scale_frac
                if away:
                    if tr.away_since is None:
                        tr.away_since = now
                        tr.posture_from = posture_before  # posture while still at rest
                    if now - tr.away_since >= cfg.start_sec:
                        events.append(self._start(tr, now))
                elif away is False:
                    tr.away_since = None
            else:
                if use_pose and tr.body_still is not None:
                    moved = _dist(tr.body_still, body) / tr.torso > cfg.pose_still_frac
                else:
                    shift, scale = _shift(tr.still_ref, bbox, ground=tr.class_name not in PERSON_CLASSES)
                    moved = shift > cfg.still_frac or scale > cfg.still_frac * 2
                if moved:
                    tr.still_ref = bbox
                    tr.body_still = body if use_pose else tr.body_still
                    tr.last_move_at = now
                elif now - tr.last_move_at >= cfg.stop_sec:
                    events.append(self._stop(tr, now, kind="stop"))

            if tr.state == MOVING:
                _add_path_point(tr, _path_point(tr, bbox, body), now)
            if tr.path and (tr.state == MOVING or now <= tr.path_until):
                det["motion_path"] = [list(p) for p in tr.path]

            det["motion_state"] = tr.state
            det["motion_since"] = tr.state_since
            det["motion_key"] = tr.key
            if body is not None or tr.posture:
                det["posture"] = WALKING if tr.state == MOVING and body is not None else tr.posture

        ttl = self._ttl()
        for key in [k for k, tr in self._tracks.items() if k not in seen and now - tr.last_seen > ttl]:
            tr = self._tracks.pop(key)
            if tr.state == MOVING:
                events.append(self._stop(tr, tr.last_seen, kind="lost"))
        return events

    def _start(self, tr: _Track, now: float) -> dict[str, Any]:
        # Movement began when the object first left the anchor, not when it was confirmed
        started = tr.away_since if tr.away_since is not None else now
        tr.state = MOVING
        tr.state_since = started
        tr.motion_from = tr.anchor
        tr.still_ref = tr.last_bbox
        tr.body_from = tr.body_anchor
        tr.body_still = tr.body_last
        tr.last_move_at = now
        tr.away_since = None
        start_pt = tr.body_from if tr.body_from is not None else (
            _center(tr.anchor) if tr.class_name in PERSON_CLASSES else _ground(tr.anchor)
        )
        tr.path = [[round(start_pt[0], 1), round(start_pt[1], 1), round(started, 2)]]
        return self._event(tr, "start", started, from_bbox=tr.motion_from, to_bbox=tr.last_bbox)

    def _stop(self, tr: _Track, now: float, *, kind: str) -> dict[str, Any]:
        started = tr.state_since
        tr.state = STATIONARY
        tr.state_since = now
        event = self._event(tr, kind, now, from_bbox=tr.motion_from, to_bbox=tr.last_bbox)
        event["motion_started_at"] = started
        event["duration_sec"] = round(max(0.0, now - started), 2)
        if tr.body_from is not None and tr.body_last is not None and tr.torso > 0:
            # Net body move A → B in torso lengths (pose); None when pose was not available
            event["body_shift"] = round(_dist(tr.body_from, tr.body_last) / tr.torso, 2)
        event["posture_to"] = tr.posture
        event["path"] = [list(p) for p in tr.path]
        tr.path_until = now + PATH_KEEP_SEC
        tr.anchor = tr.last_bbox
        tr.body_anchor = tr.body_last
        tr.away_since = None
        return event

    def _event(self, tr: _Track, kind: str, at: float, *, from_bbox: list[float], to_bbox: list[float]) -> dict[str, Any]:
        (fx, fy), (tx, ty) = _center(from_bbox), _center(to_bbox)
        return {
            "camera_key": self.camera_key,
            "event": kind,
            "at": at,
            "track_id": tr.track_id,
            "motion_key": tr.key,
            "global_object_id": tr.global_object_id,
            "class_name": tr.class_name,
            "from_bbox": [round(v, 1) for v in from_bbox],
            "to_bbox": [round(v, 1) for v in to_bbox],
            "distance_px": round(math.hypot(tx - fx, ty - fy), 1),
            "posture_from": tr.posture_from,
        }

    def moving_count(self) -> int:
        return sum(1 for tr in self._tracks.values() if tr.state == MOVING)

    def track_count(self) -> int:
        return len(self._tracks)

    def updates_per_sec(self, now: float, window: float = 30.0) -> float:
        """How often inference fed this camera recently (motion can only be seen this often)."""
        recent = [t for t in self._update_times if now - t <= window]
        return round(len(recent) / window, 2)


def _path_point(tr: _Track, bbox: list[float], body: tuple[float, float] | None) -> tuple[float, float]:
    """Body point for people with pose (so arm swings don't wiggle the line), else box centre."""
    if body is not None:
        return body
    if tr.body_anchor is not None and tr.body_last is not None:
        return tr.body_last
    return _center(bbox) if tr.class_name in PERSON_CLASSES else _ground(bbox)


def _add_path_point(tr: _Track, pt: tuple[float, float], now: float) -> None:
    last = tr.path[-1] if tr.path else None
    if last is not None and abs(last[0] - pt[0]) < 1 and abs(last[1] - pt[1]) < 1:
        last[2] = round(now, 2)
        return
    tr.path.append([round(pt[0], 1), round(pt[1], 1), round(now, 2)])
    if len(tr.path) > PATH_MAX_POINTS:
        del tr.path[1 : len(tr.path) - PATH_MAX_POINTS + 1]  # keep the start point (A)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


class MotionRegistry:
    """All cameras' motion trackers plus a short history of events for status/debugging."""

    def __init__(self, config: MotionConfig | None = None, *, history: int = 200):
        self.config = config or MotionConfig.from_env()
        self._cameras: dict[str, CameraMotionTracker] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=history)
        self._lock = threading.Lock()

    def update(self, camera_key: str, detections: list[dict[str, Any]], now: float | None = None) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []
        now = time.time() if now is None else now
        with self._lock:
            tracker = self._cameras.get(camera_key)
            if tracker is None:
                tracker = self._cameras[camera_key] = CameraMotionTracker(camera_key, self.config)
            events = tracker.update(detections, now)
            self._events.extend(events)
        if self.config.log_events:
            for ev in events:
                extra = f" after {ev['duration_sec']}s" if "duration_sec" in ev else ""
                obj = ev["global_object_id"] or f"track {ev['track_id']}"
                body = f" body {ev['body_shift']}x torso" if ev.get("body_shift") is not None else ""
                pose = ""
                if ev.get("posture_from") or ev.get("posture_to"):
                    pose = f" [{ev.get('posture_from') or '?'}→{ev.get('posture_to') or '?'}]"
                print(
                    f"[motion] {ev['camera_key']} {ev['class_name']} {obj} {ev['event']}{extra}"
                    f" moved {ev['distance_px']}px{body}{pose}"
                )
        return events

    def reset(self, camera_key: str = "") -> None:
        with self._lock:
            if camera_key:
                self._cameras.pop(camera_key, None)
            else:
                self._cameras.clear()

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            return {
                "enabled": self.config.enabled,
                "config": {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in self.config.__dict__.items()},
                "cameras": {
                    key: {
                        "tracks": t.track_count(),
                        "moving": t.moving_count(),
                        "inferences_per_sec": t.updates_per_sec(now),
                        "forget_after_sec": round(t._ttl(), 1),
                    }
                    for key, t in sorted(self._cameras.items())
                },
                "recent_events": list(self._events)[-50:],
            }
