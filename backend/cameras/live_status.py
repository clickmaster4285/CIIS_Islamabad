"""How many of this server's cameras are live right now (City Control Grid)."""

from __future__ import annotations

import logging
import time
from datetime import timedelta

import requests
from django.utils import timezone

logger = logging.getLogger(__name__)

# A camera that detected something this recently counts as live when no ML server answers.
RECENT_DETECTION = timedelta(minutes=15)
_CACHE_SEC = 60
_cache: tuple[float, dict, set[int]] | None = None


def _ml_connected_keys() -> set[str] | None:
    """Stream keys the ML servers report as connected; None when no ML server answered."""
    from ml.client import known_ml_base_urls

    keys: set[str] = set()
    answered = False
    for base in known_ml_base_urls():
        try:
            resp = requests.get(f"{base}/live/status", timeout=3, headers={"Accept": "application/json"})
            data = resp.json() if resp.status_code == 200 else None
        except (requests.RequestException, ValueError):
            continue
        streams = data.get("cameras") if isinstance(data, dict) else None
        if not isinstance(streams, list):
            continue
        answered = True
        for stream in streams:
            if isinstance(stream, dict) and (stream.get("connected") or stream.get("has_frame")):
                key = str(stream.get("key") or "").strip()
                if key:
                    keys.add(key)
    return keys if answered else None


def _live_ids() -> tuple[set[int], str, int]:
    """(ids of live cameras, source "ml" | "detections", active camera total)."""
    from cameras.models import Camera, DetectionEvent
    from city_sync.identity import local_rows_q

    cameras = list(Camera.objects.filter(is_active=True))
    connected = None
    try:
        connected = _ml_connected_keys()
    except Exception:  # noqa: BLE001
        logger.exception("Could not read live camera status from ML")
    if connected is not None:
        return {c.pk for c in cameras if c.stream_key in connected}, "ml", len(cameras)
    recent = set(
        DetectionEvent.objects.filter(
            local_rows_q(), created_at__gte=timezone.now() - RECENT_DETECTION, camera__isnull=False
        ).values_list("camera_id", flat=True)
    )
    return {c.pk for c in cameras if c.pk in recent}, "detections", len(cameras)


def _refresh() -> tuple[dict, set[int]]:
    global _cache
    if _cache and time.monotonic() - _cache[0] < _CACHE_SEC:
        return _cache[1], _cache[2]
    ids, source, total = _live_ids()
    result = {"cameras_total": total, "cameras_online": min(len(ids), total), "cameras_online_source": source}
    _cache = (time.monotonic(), result, ids)
    return result, ids


def camera_counts() -> dict:
    """{"cameras_total", "cameras_online", "cameras_online_source": "ml" | "detections"}."""
    return _refresh()[0]


def online_camera_ids() -> set[int]:
    """Ids of this server's cameras that are live now (same check and cache as camera_counts)."""
    return set(_refresh()[1])
