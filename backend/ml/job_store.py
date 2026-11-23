"""Shared disk-status helpers for Find-in-Video and Video AI Test jobs."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from django.conf import settings

STATUS_NAME = "status.json"
# Job ids are uuid4().hex[:12] (see video_clips.save_job_dir / video_analyze_jobs.save_job_dir).
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
ACTIVE_STATES = frozenset({"queued", "running"})

_active_guard = threading.Lock()
_active_jobs: set[str] = set()


def is_valid_job_id(job_id: str) -> bool:
    return bool(JOB_ID_RE.match(job_id or ""))


def max_job_seconds() -> float:
    return float(getattr(settings, "ML_VIDEO_JOB_MAX_SECONDS", 6 * 3600))


def stale_job_seconds() -> float:
    """A queued/running job whose status has not changed this long (and whose thread is not
    alive in this process) is treated as dead — e.g. the server restarted mid-job."""
    return float(getattr(settings, "ML_VIDEO_JOB_STALE_SECONDS", 2 * 3600))


def mark_active(job_id: str, active: bool) -> None:
    with _active_guard:
        if active:
            _active_jobs.add(job_id)
        else:
            _active_jobs.discard(job_id)


def is_active_here(job_id: str) -> bool:
    with _active_guard:
        return job_id in _active_jobs


def _read_raw(folder: Path) -> dict[str, Any] | None:
    path = folder / STATUS_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_status(folder: Path, payload: dict[str, Any]) -> None:
    """Atomically replace status.json, keeping the job's owner_id across updates."""
    payload = dict(payload)
    if "owner_id" not in payload:
        previous = _read_raw(folder) or {}
        if "owner_id" in previous:
            payload["owner_id"] = previous["owner_id"]
    payload["updated_at"] = time.time()
    tmp = folder / f".{STATUS_NAME}"
    dest = folder / STATUS_NAME
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)


def remove_source(folder: Path) -> None:
    for name in ("source.mp4", "query_upload.jpg"):
        try:
            (folder / name).unlink(missing_ok=True)
        except OSError:
            pass


def read_status(folder: Path) -> dict[str, Any] | None:
    data = _read_raw(folder)
    if data is None:
        return None
    if str(data.get("status") or "") in ACTIVE_STATES and not is_active_here(folder.name):
        try:
            age = time.time() - float(data.get("updated_at") or 0)
        except (TypeError, ValueError):
            age = 0.0
        if age > stale_job_seconds():
            message = "Job stopped responding (the server may have restarted). Please run it again."
            data = {
                **data,
                "status": "error",
                "progress": 100,
                "message": message,
                "error": message,
            }
            try:
                write_status(folder, data)
            except OSError:
                pass
            remove_source(folder)
    return data


def may_access(row: dict[str, Any], user) -> bool:
    """Job owner or a top admin. Jobs created before owner tracking are admin-only."""
    from users.permissions import is_global_admin, is_it_superadmin

    if is_global_admin(user) or is_it_superadmin(user):
        return True
    owner = row.get("owner_id")
    return owner is not None and getattr(user, "pk", None) is not None and str(owner) == str(user.pk)


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != "owner_id"}
