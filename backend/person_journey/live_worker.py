"""Poll ML live detections and record every visible person into Person Journey."""

from __future__ import annotations

import logging
import threading
import time

from django.conf import settings
from django.db import close_old_connections
from django.db.utils import OperationalError

from config.db import release_db

logger = logging.getLogger(__name__)

_stop = threading.Event()
_thread: threading.Thread | None = None


def _enabled() -> bool:
    return bool(getattr(settings, "PERSON_JOURNEY_LIVE_INGEST_ENABLED", True))


def _interval() -> float:
    return max(1.0, float(getattr(settings, "PERSON_JOURNEY_LIVE_INGEST_INTERVAL_SEC", 2)))


def _camera_refresh() -> int:
    return max(15, int(getattr(settings, "PERSON_JOURNEY_LIVE_CAMERA_REFRESH_SEC", 60)))


def _max_parallel() -> int:
    return max(1, min(32, int(getattr(settings, "PERSON_JOURNEY_LIVE_INGEST_PARALLEL", 8))))


def _active_cameras() -> list:
    from cameras.models import Camera

    return list(
        Camera.objects.filter(
            is_active=True,
            nvr__is_active=True,
            nvr__site__is_active=True,
            ml_server_id__isnull=False,
        )
        .select_related("nvr", "nvr__site", "ml_server")
        .order_by("id")
    )


def _fetch_detections(camera) -> list[dict]:
    """HTTP only (runs on the pool): the ML node's current detections for one camera."""
    from ml.client import MLServiceError, ml_live_detections_for_camera

    try:
        result = ml_live_detections_for_camera(camera, include_embeddings=True)
    except MLServiceError as exc:
        logger.debug("Journey live poll skipped camera %s: %s", camera.pk, exc)
        return []
    except Exception:
        logger.debug("Journey live poll failed for camera %s", camera.pk, exc_info=True)
        return []
    finally:
        release_db()  # pool threads must not keep a Postgres session if the URL lookup touched the DB
    return result.get("detections") or []


def _poll_all(cameras: list, pool) -> int:
    """
    One tick: fetch every camera's detections in parallel (bounded pool), then write them on this thread.

    Before, one camera was polled per tick, so with N cameras each camera was seen every N x interval
    seconds and people were missed entirely on busy sites.
    """
    from ml.client import ml_service_enabled

    from .live_ingest import ingest_camera_detections

    if not cameras or not ml_service_enabled():
        return 0
    futures = {pool.submit(_fetch_detections, cam): cam for cam in cameras}
    recorded = 0
    for fut, cam in futures.items():
        try:
            detections = fut.result(timeout=30)
        except Exception:
            continue
        if not detections:
            continue
        try:
            count = ingest_camera_detections(cam, detections)
            recorded += count
            if count:
                logger.debug("Journey live ingest camera %s: %s persons", cam.pk, count)
        except OperationalError:
            raise
        except Exception:
            logger.exception("Journey live ingest failed for camera %s", cam.pk)
    return recorded


def _worker_loop() -> None:
    from concurrent.futures import ThreadPoolExecutor

    cameras: list = []
    last_refresh = 0.0
    refresh_sec = _camera_refresh()
    interval = _interval()
    pool = ThreadPoolExecutor(max_workers=_max_parallel(), thread_name_prefix="journey-live-fetch")

    logger.info(
        "Person journey live ingest started (interval=%.1fs, cameras refresh=%ss, parallel=%s)",
        interval,
        refresh_sec,
        _max_parallel(),
    )

    while not _stop.is_set():
        close_old_connections()
        try:
            now = time.monotonic()
            if not cameras or now - last_refresh >= refresh_sec:
                cameras = _active_cameras()
                last_refresh = now
                if cameras:
                    logger.info("Journey live ingest tracking %s camera(s)", len(cameras))
            try:
                _poll_all(cameras, pool)
            except OperationalError:
                logger.warning("Journey live ingest: Postgres full; backing off 15s")
                release_db()
                _stop.wait(15)
                continue
        except Exception:
            logger.exception("Journey live ingest tick failed")
        finally:
            release_db()

        from config.worker_throttle import maybe_pause_for_cpu

        maybe_pause_for_cpu(logger, label="journey-live-ingest")
        _stop.wait(interval)


LOCK_NAME = "person_journey_live_ingest"


def _start_thread() -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_worker_loop, daemon=True, name="person-journey-live-ingest")
    _thread.start()


def start_live_ingest_worker() -> None:
    """Start the live ingest loop — in ONE process only (cross-process file lock, like CCTV autostart)."""
    if not _enabled():
        logger.info("Person journey live ingest disabled")
        return
    from recognition.services.autostart import run_with_process_lock

    run_with_process_lock(LOCK_NAME, _start_thread)
