"""Background worker: poll ML live detections and persist readings."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

from django.conf import settings
from django.db import close_old_connections
from django.db.utils import OperationalError

from config.db import release_db

logger = logging.getLogger(__name__)

_stop_event = threading.Event()
_worker_thread: threading.Thread | None = None
_lock_handle = None


def _lock_path() -> Path:
    base = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    return base / ".detection_worker.lock"


def _acquire_worker_lock():
    """Exclusive per-host lock so only one process runs the detection poll loop."""
    from .locks import acquire_process_lock

    return acquire_process_lock(_lock_path())


def _worker_enabled() -> bool:
    return bool(getattr(settings, "DETECTION_WORKER_ENABLED", True))


def _poll_interval() -> float:
    return max(1.0, float(getattr(settings, "DETECTION_WORKER_INTERVAL_SEC", 2)))


def _camera_refresh_interval() -> int:
    return max(15, int(getattr(settings, "DETECTION_WORKER_CAMERA_REFRESH_SEC", 60)))


def _wait_for_ml(timeout_sec: float = 180.0) -> bool:
    from ml.client import ml_health_any, ml_service_enabled

    if not ml_service_enabled():
        return False
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and not _stop_event.is_set():
        try:
            ml_health_any()
            return True
        except Exception:
            time.sleep(2.0)
    return False


def _active_camera_ids() -> list[int]:
    from .models import Camera

    return list(
        Camera.objects.filter(
            is_active=True,
            nvr__is_active=True,
            nvr__site__is_active=True,
            ml_server_id__isnull=False,
        )
        .order_by("id")
        .values_list("id", flat=True)
    )


def _poll_camera(camera_id: int) -> int:
    from ml.client import (
        MLServiceError,
        ml_live_detections_for_camera,
        ml_live_jpeg_evidence_url_for_camera,
        ml_service_enabled,
    )

    from .clip_capture import clear_stashed_evidence, stash_evidence_jpeg
    from .detection_utils import save_detection_batch
    from .models import Camera

    if not ml_service_enabled():
        return 0

    try:
        camera = Camera.objects.select_related("nvr", "nvr__site", "ml_server").get(pk=camera_id)
    except Camera.DoesNotExist:
        return 0

    if not camera.ml_server_id:
        return 0

    try:
        result = ml_live_detections_for_camera(camera, include_embeddings=True)
    except MLServiceError as exc:
        logger.debug("ML poll skipped for camera %s: %s", camera_id, exc)
        return 0

    detections = result.get("detections") or []
    if not detections:
        # Empty frame still clears crowd / fire / smoke / weapon alerts and counts as a
        # missed poll toward object exits.
        return save_detection_batch(camera, [])

    # Same-frame evidence right after detections (ML keeps the YOLO infer JPEG).
    if result.get("has_evidence"):
        try:
            import requests

            from ml.client import ml_auth_headers

            url = ml_live_jpeg_evidence_url_for_camera(camera)
            resp = requests.get(url, timeout=(1.5, 3.0), headers=ml_auth_headers())
            if resp.status_code == 200 and resp.content:
                stash_evidence_jpeg(camera.pk, resp.content)
        except Exception:
            logger.debug("Evidence fetch skipped for camera %s", camera_id)

    try:
        return save_detection_batch(camera, detections)
    finally:
        # Evidence belongs to this poll only; events of later polls fetch their own frame.
        clear_stashed_evidence(camera.pk)


def run_worker_forever() -> None:
    """Poll cameras round-robin and save detections until stopped."""
    global _lock_handle

    if not _worker_enabled():
        logger.info("[detection-worker] Disabled via DETECTION_WORKER_ENABLED")
        return

    # maybe_start_background_worker() already holds the lock for its thread; the management
    # commands call this directly and must take it here so two processes never poll at once.
    owns_lock = False
    if _lock_handle is None:
        _lock_handle = _acquire_worker_lock()
        if _lock_handle is None:
            logger.warning(
                "[detection-worker] Another process holds %s — not starting a second poll loop",
                _lock_path(),
            )
            return
        owns_lock = True

    if not _wait_for_ml():
        logger.warning("[detection-worker] ML service not ready — retrying in loop")

    camera_ids: list[int] = []
    index = 0
    last_refresh = 0.0
    interval = _poll_interval()
    refresh_sec = _camera_refresh_interval()

    logger.info(
        "[detection-worker] Started (pid=%s, interval=%.1fs, refresh=%ss)",
        os.getpid(),
        interval,
        refresh_sec,
    )

    while not _stop_event.is_set():
        close_old_connections()
        try:
            now = time.monotonic()
            if not camera_ids or now - last_refresh >= refresh_sec:
                camera_ids = _active_camera_ids()
                last_refresh = now
                if camera_ids:
                    logger.info("[detection-worker] Tracking %s active camera(s)", len(camera_ids))
                else:
                    logger.debug("[detection-worker] No active cameras configured")

            if camera_ids:
                cam_id = camera_ids[index % len(camera_ids)]
                index += 1
                try:
                    saved = _poll_camera(cam_id)
                    if saved:
                        logger.info("[detection-worker] Saved %s detection(s) for camera %s", saved, cam_id)
                except OperationalError:
                    logger.warning(
                        "[detection-worker] Postgres has no free slots; backing off 15s"
                    )
                    release_db()
                    _stop_event.wait(15)
                    continue
                except Exception:
                    logger.exception("[detection-worker] Unexpected error polling camera %s", cam_id)
        finally:
            release_db()

        from config.worker_throttle import maybe_pause_for_cpu

        maybe_pause_for_cpu(logger, label="detection-worker")
        _stop_event.wait(interval)

    logger.info("[detection-worker] Stopped")
    if owns_lock and _lock_handle is not None:
        try:
            _lock_handle.close()
        except OSError:
            pass
        _lock_handle = None


def stop_background_worker() -> None:
    global _lock_handle
    _stop_event.set()
    if _worker_thread is not None and _worker_thread.is_alive():
        _worker_thread.join(timeout=10.0)
    if _lock_handle is not None:
        try:
            _lock_handle.close()
        except OSError:
            pass
        _lock_handle = None


def should_autostart_in_process() -> bool:
    if not _worker_enabled():
        return False
    if not getattr(settings, "DETECTION_WORKER_AUTO_START", True):
        return False
    if os.environ.get("TEKEYE_DETECTION_WORKER") == "1":
        return False
    if "run_detection_worker" in sys.argv or "run_background_workers" in sys.argv:
        return False
    try:
        from config.runtime import skip_embedded_background_workers

        if skip_embedded_background_workers():
            return False
    except Exception:
        if "runserver" in sys.argv:
            return False
    if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
        return False
    return True


def maybe_start_background_worker() -> None:
    """Start daemon thread when Django boots (dev / single-process)."""
    global _worker_thread, _lock_handle

    if not should_autostart_in_process():
        return
    if _worker_thread is not None and _worker_thread.is_alive():
        return

    lock = _acquire_worker_lock()
    if lock is None:
        logger.debug("[detection-worker] Another process already holds the worker lock")
        return

    _lock_handle = lock
    _stop_event.clear()
    _worker_thread = threading.Thread(
        target=run_worker_forever,
        daemon=True,
        name="detection-worker",
    )
    _worker_thread.start()
    logger.info("[detection-worker] Background thread started")
