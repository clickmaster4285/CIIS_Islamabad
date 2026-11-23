"""Run staff face enrollment after the DB commit; from HR views, off the request thread.

Enrollment decodes every staff photo with InsightFace and pushes embeddings to the
ML node, which takes seconds. Running it inside the save held the HTTP request open
and could read photos that the transaction then rolled back.

- Every Staff save that changes photos calls schedule_staff_face_sync() (signals.py).
- The job runs in transaction.on_commit, so a rolled-back save enrolls nothing.
- Inside ``with background_face_sync():`` (the HR staff views) the job runs on a
  daemon thread; elsewhere (management commands, sync workers) it runs inline after
  commit so a short-lived process does not exit before enrollment finishes.
- Several saves of one staff member collapse: while a job for that staff runs,
  further requests only mark it pending and the running job goes round once more.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

from django.db import close_old_connections, connection, transaction

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pending: dict[int, bool] = {}  # staff pk -> force
_running: set[int] = set()
_local = threading.local()


@contextmanager
def background_face_sync():
    """Face syncs scheduled inside this block run on a background thread."""
    _local.depth = getattr(_local, "depth", 0) + 1
    try:
        yield
    finally:
        _local.depth -= 1


def _in_background_mode() -> bool:
    return getattr(_local, "depth", 0) > 0


def _sync_one(staff_pk: int, force: bool) -> None:
    from ml.face_sync import sync_staff_faces_after_save
    from users.models import Staff

    staff = Staff.objects.filter(pk=staff_pk).first()
    if staff is not None:
        sync_staff_faces_after_save(staff, force=force)


def _drain(staff_pk: int) -> None:
    """Run until no request for this staff is pending. Caller has put pk in _running."""
    try:
        while True:
            with _lock:
                force = _pending.pop(staff_pk, None)
                if force is None:
                    _running.discard(staff_pk)
                    return
            try:
                _sync_one(staff_pk, force)
            except Exception:
                logger.exception("Face sync failed for staff %s", staff_pk)
    except BaseException:
        with _lock:
            _running.discard(staff_pk)
        raise


def _thread_main(staff_pk: int) -> None:
    close_old_connections()
    try:
        _drain(staff_pk)
    finally:
        connection.close()


def _claim(staff_pk: int) -> bool:
    with _lock:
        if staff_pk in _running:
            return False  # the running job will pick up the pending request
        _running.add(staff_pk)
        return True


def _start(staff_pk: int, background: bool) -> None:
    if not _claim(staff_pk):
        return
    if background:
        threading.Thread(
            target=_thread_main, args=(staff_pk,), name=f"staff-face-sync-{staff_pk}", daemon=True
        ).start()
    else:
        _drain(staff_pk)


def schedule_staff_face_sync(staff_pk: int | None, *, force: bool = False, background: bool | None = None) -> None:
    """Queue one face sync for this staff member once the current transaction commits."""
    if not staff_pk:
        return
    if background is None:
        background = _in_background_mode()
    with _lock:
        _pending[staff_pk] = bool(_pending.get(staff_pk)) or force
    # Every call registers a hook (a rolled-back transaction drops its hooks); the first
    # job to run takes the pending entry and later hooks find nothing to do.
    transaction.on_commit(lambda: _start(staff_pk, background))
