"""Background jobs for hour-long camera recording search.

Jobs run on a bounded worker pool (ML_VIDEO_SEARCH_MAX_JOBS, default 2); extra jobs wait as
"queued". Finished jobs are forgotten after ML_VIDEO_JOB_TTL_SEC (default 6 h) and their
upload folders under tmp/video_search are deleted (orphans from earlier runs too).
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_TMP = Path(__file__).resolve().parent / "tmp" / "video_search"
_TMP.mkdir(parents=True, exist_ok=True)

_FINISHED = frozenset({"done", "error"})


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


JOB_TTL_SEC = max(60.0, _env_float("ML_VIDEO_JOB_TTL_SEC", 6 * 3600.0))
MAX_CONCURRENT_JOBS = max(1, int(_env_float("ML_VIDEO_SEARCH_MAX_JOBS", 2)))
_PRUNE_EVERY_SEC = 60.0
_last_prune = 0.0
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="video-search")


def _inside_tmp(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        resolved = Path(path).resolve()
        root = _TMP.resolve()
        return resolved != root and root in resolved.parents
    except OSError:
        return False


def _remove_path(path: Path | None) -> None:
    """Delete a file or folder, but only inside this module's tmp root."""
    if path is None or not _inside_tmp(path):
        return
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


def prune_jobs(now: float | None = None, *, ttl_sec: float | None = None) -> int:
    """Forget finished jobs older than the TTL and delete their tmp files; sweep orphans.

    Returns the number of jobs removed. Queued / running jobs are never touched.
    """
    global _last_prune
    now = time.time() if now is None else float(now)
    ttl = JOB_TTL_SEC if ttl_sec is None else float(ttl_sec)
    expired: list[dict[str, Any]] = []
    with _LOCK:
        for job_id, row in list(_JOBS.items()):
            if row.get("status") in _FINISHED and now - float(row.get("updated_at") or 0) > ttl:
                expired.append(_JOBS.pop(job_id))
        referenced = {
            Path(row["work_dir"]).resolve()
            for row in _JOBS.values()
            if row.get("work_dir")
        }
        _last_prune = now
    for row in expired:
        if row.get("work_dir"):
            _remove_path(Path(row["work_dir"]))
    # Orphans (e.g. left by a previous process): old entries no live job refers to.
    try:
        children = list(_TMP.iterdir())
    except OSError:
        children = []
    for child in children:
        try:
            if child.resolve() in referenced:
                continue
            if now - child.stat().st_mtime > ttl:
                _remove_path(child)
        except OSError:
            continue
    return len(expired)


def _maybe_prune() -> None:
    if time.time() - _last_prune >= _PRUNE_EVERY_SEC:
        try:
            prune_jobs()
        except Exception as exc:
            print(f"[video-search] prune failed: {exc}")


def _patch(job_id: str, **fields: Any) -> None:
    with _LOCK:
        row = _JOBS.get(job_id)
        if not row:
            return
        row.update(fields)
        row["updated_at"] = time.time()


def get_search_job(job_id: str) -> dict[str, Any] | None:
    _maybe_prune()
    with _LOCK:
        row = _JOBS.get(job_id)
        if not row:
            return None
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "progress": int(row.get("progress") or 0),
            "message": row.get("message") or "",
            "error": row.get("error"),
            "result": row.get("result"),
        }


def start_search_job(
    image_path: str,
    video_path: str,
    *,
    face_threshold: float,
    reid_threshold: float,
    sample_fps: float,
    clip_seconds: float,
    cleanup_video: bool = True,
) -> str:
    _maybe_prune()
    job_id = uuid.uuid4().hex[:12]
    folder = Path(image_path).parent
    work_dir = str(folder) if folder.parent.resolve() == _TMP.resolve() else ""
    with _LOCK:
        running = sum(1 for r in _JOBS.values() if r.get("status") not in _FINISHED)
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 1,
            "message": "Queued" if running < MAX_CONCURRENT_JOBS else f"Queued ({running} job(s) ahead)",
            "result": None,
            "error": None,
            "work_dir": work_dir,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    _EXECUTOR.submit(
        _run_job,
        job_id,
        image_path,
        video_path,
        face_threshold,
        reid_threshold,
        sample_fps,
        clip_seconds,
        cleanup_video,
    )
    return job_id


def _run_job(
    job_id: str,
    image_path: str,
    video_path: str,
    face_threshold: float,
    reid_threshold: float,
    sample_fps: float,
    clip_seconds: float,
    cleanup_video: bool,
) -> None:
    _patch(job_id, status="running", progress=3, message="Loading query image")
    try:
        from video_search import search_video_path

        image_bytes = Path(image_path).read_bytes()

        def progress_cb(pct: int, message: str) -> None:
            _patch(job_id, status="running", progress=int(pct), message=message)

        result = search_video_path(
            image_bytes,
            video_path,
            face_threshold=face_threshold,
            reid_threshold=reid_threshold,
            sample_fps=sample_fps,
            clip_seconds=clip_seconds,
            progress_cb=progress_cb,
        )
        _patch(job_id, status="done", progress=100, message="Done", result=result, error=None)
    except Exception as exc:
        _patch(job_id, status="error", progress=100, message=str(exc), error=str(exc))
    finally:
        try:
            Path(image_path).unlink(missing_ok=True)
        except OSError:
            pass
        if cleanup_video:
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError:
                pass
        folder = Path(image_path).parent
        try:
            if folder.is_dir() and folder.parent.resolve() == _TMP.resolve():
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


def new_upload_dir() -> Path:
    job_dir = _TMP / uuid.uuid4().hex[:12]
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def copy_upload(src_file, dest: Path) -> None:
    with dest.open("wb") as out:
        shutil.copyfileobj(src_file, out)
