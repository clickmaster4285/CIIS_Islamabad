"""Disk-backed Video AI Test jobs (tagged output video)."""

from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from django.conf import settings

from . import job_store
from .video_search_jobs import save_upload


def _safe_stem(name: str, fallback: str = "video") -> str:
    stem = Path(str(name or "")).stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")[:48]
    return stem or fallback


def unique_tagged_filename(job_id: str, source_name: str = "") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = _safe_stem(source_name)
    return f"{stem}_tagged_{job_id}_{stamp}.mp4"


def _job_dir(job_id: str) -> Path:
    return Path(settings.MEDIA_ROOT) / "video_analyze" / job_id


def save_job_dir() -> Path:
    job_id = uuid.uuid4().hex[:12]
    folder = _job_dir(job_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_status(folder: Path, payload: dict[str, Any]) -> None:
    job_store.write_status(folder, payload)


def read_status(job_id: str) -> dict[str, Any] | None:
    if not job_store.is_valid_job_id(job_id):
        return None
    return job_store.read_status(_job_dir(job_id))


def tagged_file_path(job_id: str) -> Path | None:
    if not job_store.is_valid_job_id(job_id):
        return None
    folder = _job_dir(job_id)
    status = read_status(job_id) or {}
    result = status.get("result") if isinstance(status.get("result"), dict) else {}
    name = Path(str(result.get("output_name") or "")).name
    if name:
        path = folder / name
        if path.is_file():
            return path
    matches = sorted(folder.glob("*_tagged_*.mp4")) + sorted(folder.glob("tagged*.mp4"))
    for path in matches:
        if path.is_file():
            return path
    return None


def _run_job(folder: Path, params: dict[str, Any]) -> None:
    from .client import (
        MLServiceError,
        ml_download_video_analyze,
        ml_poll_video_analyze,
        ml_reload_faces,
        ml_start_video_analyze,
    )
    from .face_sync import collect_db_face_embeddings

    job_id = folder.name
    source = folder / "source.mp4"
    source_label = str(params.get("source_name") or "video")
    tagged_name = unique_tagged_filename(job_id, source_label)
    tagged = folder / tagged_name
    job_store.mark_active(job_id, True)
    deadline = time.monotonic() + job_store.max_job_seconds()
    _write_status(
        folder,
        {
            "job_id": job_id,
            "status": "running",
            "progress": 4,
            "message": "Loading attendance staff faces",
        },
    )
    try:
        if params.get("match_staff") and params.get("person"):
            try:
                ml_reload_faces(embeddings=collect_db_face_embeddings())
            except Exception:
                pass
        _write_status(
            folder,
            {
                "job_id": job_id,
                "status": "running",
                "progress": 8,
                "message": "Sending video to AI engine",
            },
        )
        started = ml_start_video_analyze(
            video_path=str(source),
            person=bool(params.get("person")),
            vehicle=bool(params.get("vehicle")),
            weapon=bool(params.get("weapon")),
            fire=bool(params.get("fire")),
            match_staff=bool(params.get("match_staff")),
            plate=bool(params.get("plate")),
            sample_fps=float(params.get("sample_fps") or 1),
        )
        ml_job_id = str(started.get("job_id") or "").strip()
        if not ml_job_id:
            raise MLServiceError("AI engine did not return an analyze job id.", 503)

        while True:
            row = ml_poll_video_analyze(ml_job_id)
            status = str(row.get("status") or "")
            _write_status(
                folder,
                {
                    "job_id": job_id,
                    "status": "running",
                    "progress": int(row.get("progress") or 8),
                    "message": row.get("message") or "Tagging video",
                },
            )
            if status == "done":
                result = row.get("result") if isinstance(row.get("result"), dict) else {}
                _write_status(
                    folder,
                    {
                        "job_id": job_id,
                        "status": "running",
                        "progress": 96,
                        "message": "Saving tagged video",
                    },
                )
                ml_download_video_analyze(ml_job_id, str(tagged))
                public = {k: v for k, v in result.items() if k not in {"output_path", "hits"}}
                public["hits"] = result.get("hits") or []
                public["output_name"] = tagged_name
                public["output_url"] = f"/media/video_analyze/{job_id}/{tagged_name}"
                public["job_id"] = job_id
                try:
                    source.unlink(missing_ok=True)
                except OSError:
                    pass
                _write_status(
                    folder,
                    {
                        "job_id": job_id,
                        "status": "done",
                        "progress": 100,
                        "message": "Done",
                        "result": public,
                    },
                )
                break
            if status == "error":
                raise MLServiceError(str(row.get("error") or row.get("message") or "Analyze failed."), 400)
            if time.monotonic() > deadline:
                raise MLServiceError("Video tagging took too long and was stopped.", 504)
            time.sleep(2.0)
    except Exception as exc:
        _write_status(
            folder,
            {
                "job_id": job_id,
                "status": "error",
                "progress": 100,
                "message": str(exc),
                "error": str(exc),
            },
        )
        job_store.remove_source(folder)
    finally:
        job_store.mark_active(job_id, False)


def start_job(video_file, params: dict[str, Any], *, owner_id=None) -> dict[str, Any]:
    folder = save_job_dir()
    save_upload(video_file, folder / "source.mp4")
    params = dict(params or {})
    params["source_name"] = getattr(video_file, "name", "") or "video"
    payload = {
        "job_id": folder.name,
        "status": "queued",
        "progress": 1,
        "message": "Uploaded. Starting AI tagging…",
    }
    _write_status(folder, {**payload, "owner_id": owner_id})
    threading.Thread(
        target=_run_job,
        args=(folder, params),
        daemon=True,
        name=f"django-video-analyze-{folder.name}",
    ).start()
    return payload
