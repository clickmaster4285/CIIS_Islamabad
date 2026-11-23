"""Autostart InsightFace CCTV attendance workers when the server boots.

Also holds the cross-process file lock helpers: Gunicorn/Waitress run several processes, and background
loops (CCTV workers, journey ingest, ...) must run in exactly one of them.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

from django.conf import settings
from django.db import close_old_connections

logger = logging.getLogger(__name__)
_lock_handle = None
_named_locks: dict[str, object] = {}
_named_guard = threading.Lock()

CCTV_LOCK_NAME = "cctv_autostart"


def _lock_path(name: str = CCTV_LOCK_NAME) -> Path:
    base = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    return base / f".{name}.lock"


def _try_lock_file(lock_path: Path):
    """Open and exclusively lock ``lock_path`` (non-blocking). Returns the open handle or None."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = None
    try:
        fh = open(lock_path, "a+", encoding="utf-8")
        # Always lock byte 0: "a+" opens at end of file, and locking "the current position" would let
        # every process lock a different byte once the file holds a PID.
        fh.seek(0)
        if sys.platform == "win32":
            import msvcrt

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
    except OSError as exc:
        logger.debug("Could not acquire lock %s: %s", lock_path, exc)
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
        return None


def acquire_process_lock(name: str) -> bool:
    """Hold the named lock for the life of this process. True if this process holds it (now or already)."""
    with _named_guard:
        if name in _named_locks:
            return True
        handle = _try_lock_file(_lock_path(name))
        if handle is None:
            return False
        _named_locks[name] = handle
        return True


def holds_process_lock(name: str) -> bool:
    with _named_guard:
        return name in _named_locks


def run_with_process_lock(name: str, start_fn, *, retry_seconds: float = 60.0) -> bool:
    """
    Call ``start_fn`` only in the process that holds lock ``name``.

    Processes that lose the race keep a small daemon thread that retries every ``retry_seconds``, so the
    loop moves to another process if the owner dies. Returns True when ``start_fn`` ran now.
    """
    if acquire_process_lock(name):
        start_fn()
        return True

    def _retry():
        stop = threading.Event()
        while not stop.wait(retry_seconds):
            if acquire_process_lock(name):
                logger.info("Took over background loop '%s' (pid %s)", name, os.getpid())
                try:
                    start_fn()
                except Exception:
                    logger.exception("Background loop '%s' failed to start", name)
                return

    logger.info("Background loop '%s' already runs in another process; this one stands by", name)
    threading.Thread(target=_retry, name=f"lock-wait-{name}", daemon=True).start()
    return False


def _acquire_autostart_lock():
    """Backwards-compatible: the CCTV worker lock handle, or None if another process owns it."""
    if acquire_process_lock(CCTV_LOCK_NAME):
        return _named_locks.get(CCTV_LOCK_NAME)
    return None


def is_cctv_owner() -> bool:
    """True when this process owns the CCTV attendance workers."""
    return holds_process_lock(CCTV_LOCK_NAME)


def claim_cctv_ownership() -> bool:
    """Become the CCTV worker process if nobody else is. Starts the command loop on success."""
    if is_cctv_owner():
        return True
    if not acquire_process_lock(CCTV_LOCK_NAME):
        return False
    global _lock_handle
    _lock_handle = _named_locks.get(CCTV_LOCK_NAME)
    from recognition.services.cctv_control import start_owner_loop

    start_owner_loop()
    logger.info("CCTV attendance workers are owned by pid %s", os.getpid())
    return True


def collect_attendance_cameras() -> list[dict]:
    """Active cameras with attendance/face purposes and a resolvable RTSP URL."""
    from recognition.services.attendance_cameras import collect_attendance_camera_payloads

    return collect_attendance_camera_payloads(for_workers=True)


def _autostart_worker(delay_seconds: float):
    import time

    time.sleep(delay_seconds)
    if not claim_cctv_ownership():
        logger.debug("[cctv-autostart] Another process already owns the CCTV workers")
        return
    try:
        close_old_connections()
        from recognition.services.cctv_worker import get_cctv_manager

        cameras = collect_attendance_cameras()
        if not cameras:
            logger.info("CCTV autostart: no attendance-purpose cameras found")
            return
        manager = get_cctv_manager()
        statuses = manager.start_all(cameras)
        logger.info("CCTV autostart: started %d attendance workers", len(statuses))
    except Exception:
        logger.exception("CCTV autostart failed")
    finally:
        close_old_connections()


def schedule_autostart(delay_seconds: float = 3.0):
    """Start workers in a daemon thread shortly after boot (non-blocking)."""
    thread = threading.Thread(
        target=_autostart_worker,
        args=(delay_seconds,),
        name="cctv-attendance-autostart",
        daemon=True,
    )
    thread.start()
