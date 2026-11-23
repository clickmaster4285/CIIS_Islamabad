"""Background jobs for offline video AI tagging.

Live camera inference is NOT paused for these jobs any more: the analyzer throttles itself
(see video_analyze.OfflineThrottle) so it shares the GPU with the live streams.

Jobs run on a bounded worker pool (ML_VIDEO_ANALYZE_MAX_JOBS, default 1 — it was strictly
serial before); extra jobs wait as "queued". Tagged videos are written under
tmp/video_analyze/outputs/ (never next to the source file). Finished jobs are forgotten after
ML_VIDEO_JOB_TTL_SEC (default 6 h) and their output / upload folders are deleted.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}
_TMP = Path(__file__).resolve().parent / "tmp" / "video_analyze"
_TMP.mkdir(parents=True, exist_ok=True)
_OUTPUTS_NAME = "outputs"

_FINISHED = frozenset({"done", "error"})


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


JOB_TTL_SEC = max(60.0, _env_float("ML_VIDEO_JOB_TTL_SEC", 6 * 3600.0))
MAX_CONCURRENT_JOBS = max(1, int(_env_float("ML_VIDEO_ANALYZE_MAX_JOBS", 1)))
_PRUNE_EVERY_SEC = 60.0
_last_prune = 0.0
_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="video-analyze")


def _outputs_dir() -> Path:
    out = _TMP / _OUTPUTS_NAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_stem(name: str, fallback: str = "video") -> str:
    stem = Path(str(name or "")).stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:48]
    return stem or fallback


def unique_tagged_filename(job_id: str, source_path: str = "") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = _safe_stem(source_path)
    if stem.lower() in {"source", "video"}:
        stem = "video"
    return f"{stem}_tagged_{job_id}_{stamp}.mp4"


def _inside_tmp(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        resolved = Path(path).resolve()
        root = _TMP.resolve()
        return resolved != root and resolved != (root / _OUTPUTS_NAME) and root in resolved.parents
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


def _upload_dir_of(video_path: str) -> Path | None:
    """The api_server upload folder (tmp/video_analyze/<id>/) holding this video, if any."""
    folder = Path(video_path).parent
    try:
        if folder.parent.resolve() == _TMP.resolve() and folder.name != _OUTPUTS_NAME:
            return folder
    except OSError:
        pass
    return None


def prune_jobs(now: float | None = None, *, ttl_sec: float | None = None) -> int:
    """Forget finished jobs older than the TTL, delete their outputs / upload dirs, sweep orphans.

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
        referenced: set[Path] = set()
        for row in _JOBS.values():
            for key in ("work_dir", "output_path", "video_path"):
                if row.get(key):
                    referenced.add(Path(row[key]).resolve())
        _last_prune = now
    for row in expired:
        if row.get("output_path"):
            _remove_path(Path(row["output_path"]))
        if row.get("work_dir"):
            _remove_path(Path(row["work_dir"]))
    # Orphans (left by an earlier process, or empty upload dirs): old and unreferenced.
    candidates: list[Path] = []
    for folder in (_TMP, _TMP / _OUTPUTS_NAME):
        try:
            candidates.extend(c for c in folder.iterdir() if c.name != _OUTPUTS_NAME)
        except OSError:
            continue
    for child in candidates:
        try:
            resolved = child.resolve()
            if resolved in referenced or any(resolved in p.parents for p in referenced):
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
            print(f"[video-ai] prune failed: {exc}")


def _patch(job_id: str, **fields: Any) -> None:
    with _LOCK:
        row = _JOBS.get(job_id)
        if not row:
            return
        row.update(fields)
        row["updated_at"] = time.time()


def get_analyze_job(job_id: str) -> dict[str, Any] | None:
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
            # Internal (api_server strips it from the public response).
            "output_path": row.get("output_path") or "",
        }


def get_output_path(job_id: str) -> Path | None:
    with _LOCK:
        row = _JOBS.get(job_id)
        if not row:
            return None
        path = Path(str(row.get("output_path") or ""))
    if path.is_file():
        return path
    return None


def start_analyze_job(
    video_path: str,
    *,
    person: bool,
    vehicle: bool,
    weapon: bool,
    fire: bool,
    match_staff: bool,
    sample_fps: float,
    cleanup_video: bool = True,
    plate: bool = False,
) -> str:
    _maybe_prune()
    job_id = uuid.uuid4().hex[:12]
    # Always under our own tmp root — never next to an arbitrary (caller-supplied) source file.
    out_path = _outputs_dir() / unique_tagged_filename(job_id, video_path)
    upload_dir = _upload_dir_of(video_path)
    with _LOCK:
        ahead = sum(1 for r in _JOBS.values() if r.get("status") not in _FINISHED)
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 1,
            "message": "Queued" if ahead < MAX_CONCURRENT_JOBS else f"Queued ({ahead} job(s) ahead)",
            "result": None,
            "error": None,
            "output_path": str(out_path),
            "video_path": str(video_path) if not cleanup_video else "",
            "work_dir": str(upload_dir) if (upload_dir is not None and cleanup_video) else "",
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    _EXECUTOR.submit(
        _run_job,
        job_id,
        video_path,
        str(out_path),
        person,
        vehicle,
        weapon,
        fire,
        match_staff,
        sample_fps,
        cleanup_video,
        plate,
    )
    return job_id


def _run_job(
    job_id: str,
    video_path: str,
    output_path: str,
    person: bool,
    vehicle: bool,
    weapon: bool,
    fire: bool,
    match_staff: bool,
    sample_fps: float,
    cleanup_video: bool,
    plate: bool = False,
) -> None:
    _patch(job_id, status="running", progress=2, message="Starting (sharing GPU with live cameras)")
    try:
        from video_analyze import analyze_video_path

        def progress_cb(pct: int, message: str) -> None:
            _patch(job_id, status="running", progress=int(pct), message=message)
            print(f"[video-ai] {job_id} {pct}% {message}", flush=True)

        result = analyze_video_path(
            video_path,
            output_path,
            person=person,
            vehicle=vehicle,
            weapon=weapon,
            fire=fire,
            match_staff=match_staff,
            plate=plate,
            sample_fps=sample_fps,
            progress_cb=progress_cb,
        )
        # No absolute server path in the result: only the file name (output_name) is public.
        result.pop("output_path", None)
        _patch(job_id, status="done", progress=100, message="Done", result=result, error=None)
    except Exception as exc:
        _patch(job_id, status="error", progress=100, message=str(exc), error=str(exc))
        try:
            Path(output_path).unlink(missing_ok=True)
        except OSError:
            pass
    finally:
        if cleanup_video:
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError:
                pass
            upload_dir = _upload_dir_of(video_path)
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)


def new_upload_dir() -> Path:
    job_dir = _TMP / uuid.uuid4().hex[:12]
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def copy_upload(src_file, dest: Path) -> None:
    with dest.open("wb") as out:
        shutil.copyfileobj(src_file, out)
