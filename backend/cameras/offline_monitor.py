"""
Camera offline monitor.

Every CAMERA_OFFLINE_CHECK_SEC the leader process asks each ML server which streams are live
(GET /live/status) and compares that with the cameras assigned to it:

- live (connected and sending frames)  -> Camera.status "Online"; any open alert is closed
- not live for CAMERA_OFFLINE_GRACE_SEC -> Camera.status "Offline" and one CameraOfflineAlert
  is opened for admins (header bell + "Offline" on the camera tile)

Cameras that are not assigned to an ML server are left alone (they are idle by design).
Runs inside `manage.py run_background_workers` (one leader process).
"""

from __future__ import annotations

import logging
import threading
import time

import requests
from django.conf import settings
from django.db import close_old_connections
from django.utils import timezone

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None
_stop = threading.Event()
# camera id -> monotonic time it was first seen not live (debounce)
_first_bad: dict[int, float] = {}


def _check_sec() -> float:
    return max(10.0, float(getattr(settings, "CAMERA_OFFLINE_CHECK_SEC", 30)))


def _grace_sec() -> float:
    return max(0.0, float(getattr(settings, "CAMERA_OFFLINE_GRACE_SEC", 60)))


def _node_streams(base: str) -> dict[str, dict] | None:
    """stream key -> status dict from one ML server; None when it did not answer."""
    from ml.client import ml_auth_headers

    try:
        resp = requests.get(
            f"{base}/live/status", timeout=5, headers={"Accept": "application/json", **ml_auth_headers()}
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    streams = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(streams, list):
        return None
    out: dict[str, dict] = {}
    for row in streams:
        if isinstance(row, dict):
            key = str(row.get("key") or "").strip()
            if key:
                out[key] = row
    return out


def _camera_state(camera, nodes: dict[str, dict[str, dict] | None]) -> tuple[bool | None, str]:
    """(live?, reason). live None = not monitored (no ML server assigned)."""
    from ml.client import camera_ml_base_url

    base = camera_ml_base_url(camera)
    if not base:
        return None, ""
    if base not in nodes:
        nodes[base] = _node_streams(base)
    streams = nodes[base]
    if streams is None:
        return False, "ML server not reachable"
    row = streams.get((camera.stream_key or "").strip())
    if row is None:
        return False, "Not streaming on its ML server"
    if not row.get("connected"):
        return False, "No video from camera (stream disconnected)"
    if not row.get("has_frame"):
        return False, "Connected but no picture yet"
    return True, ""


def _notify() -> None:
    try:
        from realtime.sio_app import emit_invalidate

        emit_invalidate(["cameras", "alerts", "notifications"], throttle_sec=2.0)
    except Exception:  # noqa: BLE001 - realtime is optional
        logger.debug("[camera-offline] realtime emit failed", exc_info=True)


def _set_status(camera, status: str) -> None:
    from city_sync.outbox import sync_update
    from cameras.models import Camera

    sync_update(Camera.objects.filter(pk=camera.pk), status=status)


def check_once() -> dict[str, int]:
    """One pass over all active cameras. Returns counts for logging/tests."""
    from cameras.models import Camera, CameraOfflineAlert, CameraStatus

    now = timezone.now()
    mono = time.monotonic()
    nodes: dict[str, dict[str, dict] | None] = {}
    went_offline = came_back = 0
    cameras = Camera.objects.filter(is_active=True).select_related("ml_server", "nvr")
    for camera in cameras:
        live, reason = _camera_state(camera, nodes)
        if live is None:
            _first_bad.pop(camera.pk, None)
            continue
        open_alert = CameraOfflineAlert.objects.filter(camera=camera, is_open=True).order_by("-went_offline_at").first()
        if live:
            _first_bad.pop(camera.pk, None)
            if camera.status == CameraStatus.OFFLINE:
                _set_status(camera, CameraStatus.ONLINE)
            if open_alert:
                open_alert.is_open = False
                open_alert.back_online_at = now
                open_alert.save(update_fields=["is_open", "back_online_at"])
                came_back += 1
                logger.info("[camera-offline] %s back online", camera.name or camera.pk)
            continue

        first = _first_bad.setdefault(camera.pk, mono)
        if mono - first < _grace_sec() and camera.status != CameraStatus.OFFLINE:
            continue  # short blip: wait before alerting
        if camera.status != CameraStatus.OFFLINE:
            _set_status(camera, CameraStatus.OFFLINE)
        if open_alert is None:
            CameraOfflineAlert.objects.create(
                camera=camera,
                location=(camera.location or "").strip(),
                reason=reason[:200],
                went_offline_at=now,
            )
            went_offline += 1
            logger.warning("[camera-offline] %s is OFFLINE: %s", camera.name or camera.pk, reason)
        elif open_alert.reason != reason:
            open_alert.reason = reason[:200]
            open_alert.save(update_fields=["reason"])
    if went_offline or came_back:
        _notify()
    return {"went_offline": went_offline, "came_back": came_back}


def _loop() -> None:
    while not _stop.is_set():
        try:
            close_old_connections()
            check_once()
        except Exception:  # noqa: BLE001 - keep monitoring
            logger.exception("[camera-offline] check failed")
        finally:
            close_old_connections()
        _stop.wait(_check_sec())


def start_camera_offline_monitor() -> bool:
    """Start the monitor thread in this process if it is the leader. Returns True if started."""
    global _thread
    if not bool(getattr(settings, "CAMERA_OFFLINE_MONITOR_ENABLED", True)):
        return False
    from cameras.locks import is_leader

    if not is_leader("camera_offline_monitor"):
        return False
    if _thread and _thread.is_alive():
        return True
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="camera-offline-monitor", daemon=True)
    _thread.start()
    logger.info("[camera-offline] monitor started (every %ss, grace %ss)", _check_sec(), _grace_sec())
    return True


def stop_camera_offline_monitor() -> None:
    _stop.set()
