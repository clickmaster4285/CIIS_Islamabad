"""Collect motion clips recorded by the ML nodes and link each movement to its tracked object.

The ML service records a clip when objects move (pre-roll + motion, 10-30 s) and keeps it
until we collect it: list → download video + poster → store → delete on the ML node.
"""

from __future__ import annotations

from ml.client import ml_auth_headers
import logging
import math
import re
import threading
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from typing import Any

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import close_old_connections, transaction
from django.db.models import Case, Exists, IntegerField, OuterRef, Q, Sum, When
from django.utils import timezone

from config.db import release_db

from .models import GlobalObject, MotionClip, ObjectCameraTrack, ObjectMovement, ObjectVisit
from .services import _class_group, _iou

logger = logging.getLogger(__name__)

_stop = threading.Event()
_thread: threading.Thread | None = None

# How far around a movement we look for the object's track / visit (polls are 30-40 s apart)
_LINK_SLACK = timedelta(minutes=2)
_STATIONARY_LINK_IOU = 0.3

# Same id format the ML node issues (and enforces on its own endpoints). Anything else is
# never used in a file name or URL.
_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,96}$")
# A clip that fails this many times in a row is dropped on the ML node so it cannot hold
# the head of the queue forever.
_MAX_CLIP_FAILURES = 3
_clip_failures: dict[str, int] = {}


class InvalidClip(ValueError):
    """ML clip metadata that cannot be stored."""


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _ts(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value), tz=dt_timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _center(b: list) -> tuple[float, float]:
    return (float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0


def _size(b: list) -> float:
    return math.sqrt(max(1.0, (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))))


def describe_move(from_bbox: list, to_bbox: list, width: int, height: int) -> tuple[float, float, str]:
    """(distance px, distance % of frame diagonal, direction) between two boxes."""
    if len(from_bbox) < 4 or len(to_bbox) < 4:
        return 0.0, 0.0, ""
    (fx, fy), (tx, ty) = _center(from_bbox), _center(to_bbox)
    dx, dy = tx - fx, ty - fy
    dist = math.hypot(dx, dy)
    diag = math.hypot(width, height) if width and height else 0.0
    pct = round(100.0 * dist / diag, 1) if diag else 0.0
    grow = _size(to_bbox) / _size(from_bbox)
    ref = _size(from_bbox)
    if dist < 0.15 * ref:
        # Barely shifted on screen: it came toward / went away from the camera
        if grow >= 1.3:
            return round(dist, 1), pct, "closer"
        if grow <= 1 / 1.3:
            return round(dist, 1), pct, "farther"
        return round(dist, 1), pct, ""
    # Screen y grows downward; 8 compass-style directions
    angle = math.degrees(math.atan2(-dy, dx)) % 360
    names = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return round(dist, 1), pct, names[int(((angle + 22.5) % 360) // 45)]


def _camera_for_key(camera_key: str):
    from cameras.models import Camera

    key = (camera_key or "").strip().lower()
    if key.startswith("cam-") and key[4:].isdigit():
        return Camera.objects.filter(pk=int(key[4:])).first()
    return None


def link_movement(camera, mv: dict[str, Any], started: datetime, ended: datetime) -> tuple[GlobalObject | None, ObjectVisit | None]:
    """Find the tracked object and visit this movement belongs to (None when the backend never saw it)."""
    if camera is None:
        return None, None
    lo, hi = started - _LINK_SLACK, ended + _LINK_SLACK
    obj: GlobalObject | None = None

    # 1) Same ByteTrack id on this camera around that time
    try:
        track_id = int(mv.get("track_id"))
    except (TypeError, ValueError):
        track_id = None
    if track_id is not None:
        track = (
            ObjectCameraTrack.objects.select_related("global_object")
            .filter(camera=camera, local_track_id=track_id, started_at__lte=hi)
            .filter(Q(ended_at__isnull=True) | Q(ended_at__gte=lo))
            .order_by("-started_at")
            .first()
        )
        if track is not None:
            obj = track.global_object

    # 2) ML object code, only if that object was on this camera then (codes restart with ML)
    ml_gid = str(mv.get("global_object_id") or "").strip()
    if obj is None and ml_gid:
        obj = (
            GlobalObject.objects.filter(
                code=ml_gid,
                visits__camera=camera,
                visits__entry_at__lte=hi,
            )
            .filter(Q(visits__exit_at__isnull=True) | Q(visits__exit_at__gte=lo))
            .distinct()
            .first()
        )

    visits = ObjectVisit.objects.filter(camera=camera, entry_at__lte=hi).filter(
        Q(exit_at__isnull=True) | Q(exit_at__gte=lo)
    )

    # 3) Whatever of this class rested where the movement began
    from_bbox = mv.get("from_bbox") or []
    if obj is None and len(from_bbox) >= 4:
        group = _class_group(str(mv.get("class_name") or ""))
        best, best_iou = None, _STATIONARY_LINK_IOU
        for visit in visits.select_related("global_object").order_by("-entry_at")[:50]:
            if _class_group(visit.global_object.class_name or "") != group:
                continue
            score = _iou(visit.bbox or [], from_bbox)
            if score > best_iou:
                best, best_iou = visit, score
        if best is not None:
            return best.global_object, best

    if obj is None:
        return None, None
    visit = visits.filter(global_object=obj).order_by("-entry_at").first()
    return obj, visit


def _num(value: Any, kind=float, default=0, *, lo=None, hi=None):
    """Parse a number from ML metadata, or raise InvalidClip."""
    if value is None or value == "":
        return default
    try:
        out = kind(value)
    except (TypeError, ValueError, OverflowError):
        raise InvalidClip(f"bad number {value!r}")
    if isinstance(out, float) and not math.isfinite(out):
        raise InvalidClip(f"bad number {value!r}")
    if (lo is not None and out < lo) or (hi is not None and out > hi):
        raise InvalidClip(f"number out of range {value!r}")
    return out


def _bbox(value: Any) -> list[float]:
    if not value:
        return []
    if not isinstance(value, (list, tuple)):
        raise InvalidClip("bbox is not a list")
    return [_num(v) for v in list(value)[:4]]


def _parse_clip_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Validate everything up front, so no file is written for metadata we would reject."""
    if not isinstance(meta, dict):
        raise InvalidClip("metadata is not an object")
    clip_id = str(meta.get("clip_id") or "").strip()
    if not _CLIP_ID_RE.match(clip_id):
        raise InvalidClip(f"bad clip id {clip_id!r}")
    started = _ts(meta.get("started_at")) or timezone.now()
    ended = _ts(meta.get("ended_at")) or started
    movements = meta.get("movements") or []
    if not isinstance(movements, list):
        raise InvalidClip("movements is not a list")
    moves = []
    for mv in movements:
        if not isinstance(mv, dict):
            raise InvalidClip("movement is not an object")
        try:
            track_id = int(mv.get("track_id"))
        except (TypeError, ValueError, OverflowError):
            track_id = None
        moves.append({
            "raw": mv,
            "started": _ts(mv.get("started_at")) or started,
            "ended": _ts(mv.get("ended_at")),
            "from_bbox": _bbox(mv.get("from_bbox")),
            "to_bbox": _bbox(mv.get("to_bbox")),
            "track_id": track_id if track_id is not None and 0 <= track_id < 2**31 else None,
            "body_shift": _float_or_none(mv.get("body_shift")),
        })
    return {
        "clip_id": clip_id,
        "camera_key": str(meta.get("camera_key") or "")[:64],
        "part": max(1, _num(meta.get("part"), int, 1, lo=0, hi=32767)),
        "started": started,
        "ended": ended,
        "duration": _num(meta.get("duration_sec"), float, 0.0, lo=0),
        "src_w": _num(meta.get("source_width"), int, 0, lo=0, hi=100_000),
        "src_h": _num(meta.get("source_height"), int, 0, lo=0, hi=100_000),
        "clip_w": _num(meta.get("clip_width"), int, 0, lo=0, hi=100_000),
        "clip_h": _num(meta.get("clip_height"), int, 0, lo=0, hi=100_000),
        "fps": _num(meta.get("fps"), float, 0.0, lo=0, hi=1000),
        "trigger": "danger" if meta.get("trigger") == "danger" else "motion",
        "moves": moves,
    }


def store_clip(
    meta: dict[str, Any], video: bytes, poster: bytes | None, end_frame: bytes | None = None
) -> MotionClip | None:
    """Save one ML clip with its movements. Returns None when it is already stored.

    Raises InvalidClip for metadata that cannot be stored. Files written before a failure
    (or a transaction rollback) are deleted again, so a failed clip leaves nothing behind.
    """
    m = _parse_clip_meta(meta)
    clip_id = m["clip_id"]
    if MotionClip.objects.filter(ml_clip_id=clip_id).exists():
        return None
    camera = _camera_for_key(m["camera_key"])
    src_w, src_h = m["src_w"], m["src_h"]
    clip = MotionClip(
        ml_clip_id=clip_id,
        camera=camera,
        camera_key=m["camera_key"],
        part=m["part"],
        started_at=m["started"],
        ended_at=m["ended"],
        duration_seconds=m["duration"],
        source_width=src_w,
        source_height=src_h,
        clip_width=m["clip_w"],
        clip_height=m["clip_h"],
        fps=m["fps"],
        size_bytes=len(video),
        trigger=m["trigger"],
    )
    try:
        with transaction.atomic():
            clip.video.save(f"{clip_id}.mp4", ContentFile(video), save=False)
            if poster:
                clip.poster.save(f"{clip_id}.jpg", ContentFile(poster), save=False)
            if end_frame:
                clip.end_frame.save(f"{clip_id}.end.jpg", ContentFile(end_frame), save=False)
            clip.save()

            for mv in m["moves"]:
                raw = mv["raw"]
                mv_start, mv_end = mv["started"], mv["ended"]
                obj, visit = link_movement(camera, raw, mv_start, mv_end or m["ended"])
                from_bbox, to_bbox = mv["from_bbox"], mv["to_bbox"]
                dist, pct, direction = describe_move(from_bbox, to_bbox, src_w, src_h)
                ObjectMovement.objects.create(
                    clip=clip,
                    global_object=obj,
                    visit=visit,
                    camera=camera,
                    class_name=str(raw.get("class_name") or "")[:80],
                    ml_object_id=str(raw.get("global_object_id") or "")[:32],
                    local_track_id=mv["track_id"],
                    started_at=mv_start,
                    ended_at=mv_end,
                    still_moving=bool(raw.get("still_moving")),
                    duration_seconds=max(0.0, ((mv_end or m["ended"]) - mv_start).total_seconds()),
                    from_bbox=from_bbox,
                    to_bbox=to_bbox,
                    distance_px=dist,
                    distance_pct=pct,
                    direction=direction,
                    body_shift=mv["body_shift"],
                    posture_from=str(raw.get("posture_from") or "")[:16],
                    posture_to=str(raw.get("posture_to") or "")[:16],
                )
    except BaseException:
        # Rolled back: the row is gone, so the files must go too.
        _delete_files(clip)
        raise
    return clip


_DANGER_TITLES = {"fire": "Fire Detected", "smoke": "Smoke Detected", "weapon": "Weapon Detected"}
# A poll-based alert for the same camera + kind this close to the clip = already raised
_DANGER_DEDUP = timedelta(minutes=2)


def raise_danger_alerts(clip: MotionClip, meta: dict[str, Any], poster: bytes | None) -> int:
    """Weapon / fire / smoke seen by ML (even between backend polls) → one alert per kind.

    Skipped when the live poll already raised the same alert on this camera around the
    clip. The alert gets the clip's poster (frame at first sighting) and its video.
    """
    from cameras.detection_utils import _classify_alert_key, same_spot_alert_exists
    from cameras.models import ClipStatus, DetectionEvent

    camera = clip.camera
    if camera is None:
        return 0
    best: dict[str, dict[str, Any]] = {}
    for mv in meta.get("movements") or []:
        if not mv.get("danger"):
            continue
        key = _classify_alert_key({"class_name": mv.get("class_name") or ""})
        if key in _DANGER_TITLES and key not in best:
            best[key] = mv
    created = 0
    for key, mv in best.items():
        class_name = key
        if DetectionEvent.objects.filter(
            camera=camera,
            is_alert=True,
            class_name=class_name,
            created_at__gte=clip.started_at - _DANGER_DEDUP,
            created_at__lte=clip.ended_at + _DANGER_DEDUP,
        ).exists():
            continue
        if same_spot_alert_exists(camera, class_name, mv.get("from_bbox") or []):
            continue
        zone = (camera.zone or "").strip() or "—"
        cam_name = (camera.name or camera.code or f"cam-{camera.pk}").strip()
        event = DetectionEvent(
            camera=camera,
            class_name=class_name,
            label=f"{_DANGER_TITLES[key]} @ {cam_name} / {zone}"[:120],
            employee_name="",
            personal_number="",
            confidence=0.0,
            bbox=[float(v) for v in (mv.get("from_bbox") or [])][:4],
            infer_frame_width=clip.source_width or None,
            infer_frame_height=clip.source_height or None,
            is_alert=True,
            clip_status=ClipStatus.READY if poster else ClipStatus.SKIPPED,
            track_event="alert",
        )
        if poster:
            event.clip.save(f"{clip.ml_clip_id}-{key}.jpg", ContentFile(poster), save=False)
        event.video.name = clip.video.name  # same file as the motion clip
        event.save()
        created += 1
        logger.warning("[motion-clips] %s alert from ML clip %s camera=%s", key, clip.ml_clip_id, cam_name)
    if created:
        try:
            from realtime.sio_app import emit_invalidate

            emit_invalidate(["cameras", "detections", "alerts"], throttle_sec=0.5)
        except Exception:
            logger.debug("Alert realtime emit failed", exc_info=True)
    return created


def _record_failure(base: str, clip_id: str, reason: str, *, poison: bool = False) -> None:
    """Count a failed clip; drop it on the ML node after repeated failures (or at once when
    its metadata can never be stored)."""
    import requests

    key = f"{base}|{clip_id}"
    count = _clip_failures.get(key, 0) + 1
    _clip_failures[key] = count
    if not poison and count < _MAX_CLIP_FAILURES:
        logger.warning("[motion-clips] %s: %s (attempt %s/%s)", clip_id, reason, count, _MAX_CLIP_FAILURES)
        return
    logger.warning("[motion-clips] %s: %s - giving up, deleting it on %s", clip_id, reason, base)
    _clip_failures.pop(key, None)
    if _CLIP_ID_RE.match(clip_id):
        try:
            requests.delete(f"{base}/live/motion/clips/{clip_id}", timeout=10, headers=ml_auth_headers())
        except Exception as exc:
            logger.warning("[motion-clips] %s: delete on ML node failed: %s", clip_id, exc)


def _pull_one(base: str, meta: dict[str, Any]) -> bool:
    """Collect one clip. Returns True when it was newly stored."""
    import requests

    clip_id = str(meta.get("clip_id") or "").strip()
    stored = False
    if not MotionClip.objects.filter(ml_clip_id=clip_id).exists():
        _parse_clip_meta(meta)  # reject bad metadata before downloading anything
        video = requests.get(f"{base}/live/motion/clips/{clip_id}/video", timeout=60, headers=ml_auth_headers())
        if video.status_code != 200 or not video.content:
            raise IOError(f"video download failed ({video.status_code})")
        images = {}
        for kind in ("poster", "end"):
            img = requests.get(f"{base}/live/motion/clips/{clip_id}/{kind}", timeout=30, headers=ml_auth_headers())
            images[kind] = img.content if img.status_code == 200 else None
        clip = store_clip(meta, video.content, images["poster"], images["end"])
        if clip:
            stored = True
            if meta.get("trigger") == "danger":
                try:
                    raise_danger_alerts(clip, meta, images["poster"])
                except Exception:
                    logger.exception("[motion-clips] %s: danger alert failed", clip_id)
    # Stored (now or before) → the ML node may drop its copy
    requests.delete(f"{base}/live/motion/clips/{clip_id}", timeout=10, headers=ml_auth_headers())
    return stored


def pull_from_node(base_url: str, *, limit: int = 20) -> int:
    """Collect finished clips from one ML node. Returns how many were stored.

    Each clip is handled on its own: one bad clip is logged and retried a few times, then
    deleted on the ML node, and never stops the clips after it from being collected."""
    import requests

    base = base_url.rstrip("/")
    res = requests.get(f"{base}/live/motion/clips", params={"limit": limit}, timeout=10, headers=ml_auth_headers())
    if res.status_code == 404:
        return 0  # ML node without motion clips (older code)
    res.raise_for_status()
    stored = 0
    for meta in res.json().get("clips") or []:
        clip_id = str((meta or {}).get("clip_id") or "").strip() if isinstance(meta, dict) else ""
        if not _CLIP_ID_RE.match(clip_id):
            logger.warning("[motion-clips] %s: skipping clip with invalid id %r", base, clip_id[:120])
            continue
        try:
            if _pull_one(base, meta):
                stored += 1
            _clip_failures.pop(f"{base}|{clip_id}", None)
        except InvalidClip as exc:
            _record_failure(base, clip_id, f"invalid metadata: {exc}", poison=True)
        except Exception as exc:
            _record_failure(base, clip_id, f"{type(exc).__name__}: {exc}")
    return stored


def pull_motion_clips() -> int:
    from ml.client import known_ml_base_urls

    stored = 0
    for base in known_ml_base_urls():
        try:
            stored += pull_from_node(base)
        except Exception as exc:
            logger.warning("[motion-clips] Pull from %s failed: %s", base, exc)
    return stored


def _delete_files(clip: MotionClip) -> None:
    for f in (clip.video, clip.poster, clip.end_frame):
        if f:
            try:
                f.delete(save=False)
            except OSError:
                pass


def _delete_clip(clip: MotionClip) -> None:
    _delete_files(clip)
    clip.delete()


def _prunable(qs):
    """Clips whose video is not also the evidence video of a DetectionEvent (danger alerts
    point their video at the clip file); those are kept with the alert."""
    from cameras.models import DetectionEvent

    return qs.filter(Q(video="") | ~Exists(DetectionEvent.objects.filter(video=OuterRef("video"))))


def prune_old_clips(
    *,
    days: int | None = None,
    danger_days: int | None = None,
    max_gb: float | None = None,
    batch: int = 1000,
) -> int:
    """Delete clips (files + rows) by age and keep the total under a disk cap.

    Normal clips: older than MOTION_CLIP_RETENTION_DAYS. Danger clips (weapon / fire /
    smoke): older than MOTION_CLIP_DANGER_RETENTION_DAYS. Then, while all clips together
    are over MOTION_CLIP_MAX_GB, the oldest go first — normal clips before danger clips.
    0 = no limit for each setting.
    """
    keep = int(days if days is not None else getattr(settings, "MOTION_CLIP_RETENTION_DAYS", 7))
    keep_danger = int(
        danger_days if danger_days is not None else getattr(settings, "MOTION_CLIP_DANGER_RETENTION_DAYS", 90)
    )
    cap_gb = float(max_gb if max_gb is not None else getattr(settings, "MOTION_CLIP_MAX_GB", 200))
    now = timezone.now()
    removed = 0

    for trigger_q, days_kept in ((~Q(trigger="danger"), keep), (Q(trigger="danger"), keep_danger)):
        if days_kept <= 0:
            continue
        cutoff = now - timedelta(days=days_kept)
        for clip in _prunable(MotionClip.objects.filter(trigger_q, started_at__lt=cutoff)).order_by("started_at")[:batch]:
            _delete_clip(clip)
            removed += 1

    if cap_gb > 0:
        cap = int(cap_gb * 1024**3)
        total = MotionClip.objects.aggregate(t=Sum("size_bytes"))["t"] or 0
        if total > cap:
            order = _prunable(MotionClip.objects.all()).annotate(
                is_danger=Case(When(trigger="danger", then=1), default=0, output_field=IntegerField())
            ).order_by("is_danger", "started_at")
            for clip in order[:batch]:
                if total <= cap:
                    break
                total -= clip.size_bytes
                _delete_clip(clip)
                removed += 1
    return removed


def _worker_loop() -> None:
    interval = max(3.0, float(getattr(settings, "MOTION_CLIP_PULL_INTERVAL_SEC", 10)))
    last_prune = 0.0
    while not _stop.is_set():
        close_old_connections()
        try:
            stored = pull_motion_clips()
            if stored:
                logger.info("[motion-clips] Stored %s new clip(s)", stored)
            now = timezone.now().timestamp()
            if now - last_prune > 3600:
                last_prune = now
                pruned = prune_old_clips()
                if pruned:
                    logger.info("[motion-clips] Deleted %s clip(s) past retention", pruned)
        except Exception:
            logger.exception("[motion-clips] Worker error")
        finally:
            release_db()
        _stop.wait(interval)


def start_motion_clip_worker() -> None:
    global _thread
    if not getattr(settings, "MOTION_CLIP_PULL_ENABLED", True):
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_worker_loop, daemon=True, name="motion-clip-pull")
    _thread.start()


def stop_motion_clip_worker() -> None:
    _stop.set()
