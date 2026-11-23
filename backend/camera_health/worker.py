"""Periodic camera health scanner (1 frame per camera ~ every 45–60s)."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

from django.conf import settings
from django.db import close_old_connections

logger = logging.getLogger(__name__)

_stop_event = threading.Event()
_worker_thread: threading.Thread | None = None
_lock_handle = None


def _lock_path() -> Path:
    base = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    return base / ".camera_health_worker.lock"


def _acquire_lock():
    lock_path = _lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = None
    try:
        fh = open(lock_path, "a+", encoding="utf-8")
        if sys.platform == "win32":
            import msvcrt

            # "a+" opens at EOF and msvcrt locks from the current position: lock byte 0.
            fh.seek(0)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                fh.close()
                return None
        else:
            import fcntl

            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                return None
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        return fh
    except OSError:
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
        return None


def _enabled() -> bool:
    return bool(getattr(settings, "CAMERA_HEALTH_WORKER_ENABLED", True))


def _interval() -> float:
    return max(20.0, float(getattr(settings, "CAMERA_HEALTH_INTERVAL_SEC", 45)))


def _retention_interval() -> float:
    return max(60.0, float(getattr(settings, "CAMERA_HEALTH_RETENTION_INTERVAL_SEC", 3600)))


def _worker_loop() -> None:
    from .services import prune_old_snapshots, scan_all_cameras, scan_lock

    logger.info("[camera-health] worker started interval=%ss", _interval())
    # Stagger start so detection worker settles first
    _stop_event.wait(8.0)
    last_prune = 0.0
    while not _stop_event.is_set():
        close_old_connections()
        try:
            # Shared with API-triggered scans so two scans never run at once here.
            if scan_lock.acquire(timeout=_interval()):
                try:
                    result = scan_all_cameras()
                finally:
                    scan_lock.release()
                logger.info(
                    "[camera-health] scan ok=%s failed=%s",
                    result.get("ok"),
                    result.get("failed"),
                )
            if time.monotonic() - last_prune >= _retention_interval() or not last_prune:
                last_prune = time.monotonic()
                pruned = prune_old_snapshots()
                if pruned:
                    logger.info("[camera-health] retention pruned %s snapshot(s)", pruned)
        except Exception:
            logger.exception("[camera-health] scan loop error")
        finally:
            close_old_connections()
        _stop_event.wait(_interval())


def maybe_start_health_worker() -> None:
    global _worker_thread, _lock_handle
    if not _enabled():
        return
    if _worker_thread and _worker_thread.is_alive():
        return
    _lock_handle = _acquire_lock()
    if _lock_handle is None:
        logger.debug("[camera-health] another worker holds the lock")
        return
    _stop_event.clear()
    _worker_thread = threading.Thread(
        target=_worker_loop, daemon=True, name="camera-health-worker"
    )
    _worker_thread.start()
