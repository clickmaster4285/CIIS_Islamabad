"""Background worker: sync cameras/NVRs + poll device health."""

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
_wake_event = threading.Event()
_worker_thread: threading.Thread | None = None
_lock_handle = None
# Guards one-off refresh cycles started from a request in a process without the worker.
_oneoff_lock = threading.Lock()

_WAIT_SLICE_SEC = 3.0


def _lock_path() -> Path:
    base = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    return base / ".infra_monitoring_worker.lock"


def _refresh_flag_path() -> Path:
    base = Path(getattr(settings, "BASE_DIR", Path.cwd()))
    return base / ".infra_monitoring_refresh.flag"


def _acquire_lock():
    lock_path = _lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = None
    try:
        fh = open(lock_path, "a+", encoding="utf-8")
        if sys.platform == "win32":
            import msvcrt

            # "a+" leaves the position at EOF; msvcrt locks bytes from the current
            # position, so lock/unlock must both address byte 0.
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
            except OSError:
                fh.close()
                return None
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except Exception:
        if fh:
            fh.close()
        return None


def _release_lock(fh) -> None:
    if not fh:
        return
    try:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        fh.close()
    except Exception:
        pass


def _consume_refresh_flag() -> bool:
    path = _refresh_flag_path()
    try:
        if path.exists():
            path.unlink()
            return True
    except OSError:
        pass
    return False


def _run_cycle() -> None:
    from .retention import maybe_prune_infra_tables
    from .services import sync_and_poll

    stats = sync_and_poll()
    logger.info("[infra-monitor] cycle ok: %s", stats)
    try:
        maybe_prune_infra_tables()
    except Exception:
        logger.exception("[infra-monitor] retention failed")


def _wait_for_next_cycle(interval: float) -> None:
    """Sleep up to ``interval``; wake early on stop, an in-process refresh or a refresh flag."""
    deadline = time.monotonic() + interval
    while not _stop_event.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        if _wake_event.wait(min(remaining, _WAIT_SLICE_SEC)):
            _wake_event.clear()
            return
        if _consume_refresh_flag():
            return


def _loop() -> None:
    interval = float(getattr(settings, "INFRA_POLL_INTERVAL_SEC", 45))
    interval = max(15.0, interval)
    logger.info("[infra-monitor] worker started (interval=%ss)", interval)
    # First cycle soon after boot
    first = True
    while not _stop_event.is_set():
        if not first:
            _wait_for_next_cycle(interval)
            if _stop_event.is_set():
                break
        first = False
        _consume_refresh_flag()  # this cycle covers any pending request
        close_old_connections()
        try:
            _run_cycle()
        except Exception:
            logger.exception("[infra-monitor] cycle failed")
        finally:
            close_old_connections()
    logger.info("[infra-monitor] worker stopped")


def request_refresh() -> str:
    """
    Ask for an immediate sync + poll without running it in the request.

    Returns "queued" (the worker will run it now), "started" (a one-off background
    cycle was started here) or "running" (a one-off cycle is already in progress).
    Never runs concurrently with the worker: the one-off path takes the same lock.
    """
    if _worker_thread and _worker_thread.is_alive():
        _wake_event.set()
        return "queued"
    if not _oneoff_lock.acquire(blocking=False):
        return "running"
    fh = _acquire_lock()
    if fh is None:
        _oneoff_lock.release()
        # The worker runs in another process: leave a flag it checks every few seconds.
        try:
            _refresh_flag_path().touch()
        except OSError:
            logger.warning("[infra-monitor] could not write refresh flag")
        return "queued"

    def _run() -> None:
        close_old_connections()
        try:
            _run_cycle()
        except Exception:
            logger.exception("[infra-monitor] one-off refresh failed")
        finally:
            close_old_connections()
            _release_lock(fh)
            _oneoff_lock.release()

    threading.Thread(target=_run, name="infra-monitoring-refresh", daemon=True).start()
    return "started"


def maybe_start_infra_worker() -> None:
    global _worker_thread, _lock_handle
    enabled = getattr(settings, "INFRA_MONITORING_WORKER_ENABLED", True)
    if not enabled:
        logger.info("[infra-monitor] worker disabled")
        return
    if _worker_thread and _worker_thread.is_alive():
        return
    _lock_handle = _acquire_lock()
    if _lock_handle is None:
        logger.info("[infra-monitor] another worker holds the lock; skip")
        return
    _stop_event.clear()
    _wake_event.clear()
    _worker_thread = threading.Thread(
        target=_loop, name="infra-monitoring-worker", daemon=True
    )
    _worker_thread.start()


def stop_infra_worker() -> None:
    global _worker_thread, _lock_handle
    _stop_event.set()
    _wake_event.set()
    if _worker_thread:
        _worker_thread.join(timeout=5)
    _worker_thread = None
    _release_lock(_lock_handle)
    _lock_handle = None
