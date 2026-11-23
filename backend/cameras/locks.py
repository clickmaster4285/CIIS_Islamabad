"""Cross-process locks (Windows + POSIX) for workers and shared files."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from django.conf import settings

logger = logging.getLogger(__name__)


def lock_dir() -> Path:
    return Path(getattr(settings, "BASE_DIR", Path.cwd()))


def acquire_process_lock(path: Path | str):
    """
    Non-blocking exclusive lock held for the life of the returned handle (leader election).

    Returns an open file handle (keep a reference; close it to release) or None when another
    process — or another handle in this process — already holds the lock. The OS drops the
    lock automatically when the holder exits, so a crashed leader never blocks a successor.
    """
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = None
    try:
        fh = open(lock_path, "a+", encoding="utf-8")
        if sys.platform == "win32":
            import msvcrt

            try:
                fh.seek(0)
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
        logger.debug("Could not acquire process lock %s: %s", lock_path, exc)
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
        return None


def release_process_lock(handle) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except OSError:
        pass


_leader_handles: dict[str, object] = {}


def is_leader(name: str) -> bool:
    """True when this process holds (or just acquired) the named leader lock.

    Once acquired, the lock is kept until the process exits, so exactly one process runs the
    gated job. Non-leaders may call again later: they take over if the leader process died.
    """
    if name in _leader_handles:
        return True
    handle = acquire_process_lock(lock_dir() / f".{name}.lock")
    if handle is None:
        return False
    _leader_handles[name] = handle
    return True


class FileLockTimeout(TimeoutError):
    pass


@contextmanager
def exclusive_file_lock(
    target: Path | str,
    *,
    timeout: float = 10.0,
    stale_after: float = 30.0,
    poll: float = 0.05,
) -> Iterator[Path]:
    """
    Short critical section on a shared file, safe across processes and languages.

    Protocol (ml_services must follow the same one for the same file):
      * lock file = "<target>.lock" in the same directory (e.g. "captures.csv.lock");
      * acquire = atomic create with O_CREAT | O_EXCL, then write "<pid> <unix_ts>";
      * if the lock file exists and its mtime is older than ``stale_after`` seconds,
        the holder is presumed dead: delete it and retry;
      * retry every ``poll`` seconds until ``timeout``;
      * release = delete the lock file. Hold it only while appending / rewriting.
    """
    lock_path = Path(f"{target}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.1, timeout)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            except OSError:
                age = 0.0
            if age > stale_after:
                try:
                    lock_path.unlink()
                    logger.warning("Removed stale lock %s (age %.0fs)", lock_path, age)
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise FileLockTimeout(f"Timed out waiting for {lock_path}")
            time.sleep(poll)
            continue
        except PermissionError:
            # Windows: the lock file is being deleted by its holder right now.
            if time.monotonic() >= deadline:
                raise FileLockTimeout(f"Timed out waiting for {lock_path}")
            time.sleep(poll)
            continue
        try:
            os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode("ascii"))
        finally:
            os.close(fd)
        break
    try:
        yield lock_path
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass
