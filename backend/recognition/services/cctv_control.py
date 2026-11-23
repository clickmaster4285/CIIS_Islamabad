"""
Cross-process control of the CCTV attendance workers.

Web servers run several processes, but each camera must be read by ONE worker. Only the process holding the
``cctv_autostart`` file lock (see autostart.py) runs CCTVWorkerManager threads. Every other process:

  * sends start/stop requests as small JSON command files the owner picks up within ~2 s;
  * reads worker status / events / preview JPEGs from files the owner refreshes every ~2 s.

If no process owns the workers yet, the first process that receives a start request takes ownership.
All files live on the local disk next to the lock, so this works for every process of one server.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from django.conf import settings

logger = logging.getLogger(__name__)

POLL_SECONDS = 2.0
COMMAND_MAX_AGE = 120.0
STATUS_MAX_AGE = 30.0

_owner_thread: threading.Thread | None = None
_owner_guard = threading.Lock()
_last_written: dict[int, bytes] = {}


def _base_dir() -> Path:
    root = Path(getattr(settings, "BASE_DIR", Path.cwd())) / ".cache" / "cctv_control"
    (root / "commands").mkdir(parents=True, exist_ok=True)
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    return root


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def is_owner() -> bool:
    from recognition.services.autostart import is_cctv_owner

    return is_cctv_owner()


def claim_ownership() -> bool:
    from recognition.services.autostart import claim_cctv_ownership

    return claim_cctv_ownership()


def submit_command(action: str, camera_id: int | None = None) -> None:
    """Queue a start/stop request for the owning process."""
    payload = {"action": action, "camera_id": camera_id, "at": time.time(), "pid": os.getpid()}
    name = f"{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex[:6]}.json"
    _write_atomic(_base_dir() / "commands" / name, json.dumps(payload).encode("utf-8"))


def _run_command(manager, cmd: dict) -> None:
    from recognition.services.attendance_cameras import (
        camera_rtsp_payload,
        collect_attendance_camera_payloads,
        sync_camera_attendance_worker,
    )

    action = cmd.get("action")
    camera_id = cmd.get("camera_id")
    if action == "start_all":
        manager.start_all(collect_attendance_camera_payloads(for_workers=False))
    elif action == "stop_all":
        manager.stop_all()
    elif action == "stop" and camera_id is not None:
        manager.stop_camera(int(camera_id))
    elif action in ("start", "sync") and camera_id is not None:
        from cameras.models import Camera

        cam = Camera.objects.filter(pk=int(camera_id)).select_related("nvr").first()
        if cam is None:
            manager.stop_camera(int(camera_id))
            return
        if action == "sync":
            sync_camera_attendance_worker(cam)
            return
        payload = camera_rtsp_payload(cam) if cam.is_active else None
        if payload:
            manager.start_camera(
                payload["id"], payload["name"], payload["rtsp_url"], stream_key=payload.get("stream_key") or ""
            )
    else:
        logger.warning("Unknown CCTV command %s", cmd)


def _drain_commands(manager) -> int:
    folder = _base_dir() / "commands"
    done = 0
    for path in sorted(folder.glob("*.json")):
        try:
            cmd = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cmd = None
        try:
            path.unlink()
        except OSError:
            continue  # another owner loop iteration already took it
        if not cmd or time.time() - float(cmd.get("at") or 0) > COMMAND_MAX_AGE:
            continue
        try:
            _run_command(manager, cmd)
            done += 1
        except Exception:
            logger.exception("CCTV command failed: %s", cmd)
    return done


def _publish_status(manager) -> None:
    root = _base_dir()
    statuses = manager.list_status()
    doc = {
        "at": time.time(),
        "pid": os.getpid(),
        "statuses": statuses,
        "events": manager.get_events(limit=60),
    }
    _write_atomic(root / "status.json", json.dumps(doc, default=str).encode("utf-8"))
    for status in statuses:
        if not status.get("has_snapshot"):
            continue
        camera_id = int(status["camera_id"])
        jpeg = manager.get_snapshot_jpeg(camera_id)
        # Workers replace last_jpeg with a new bytes object per preview; skip unchanged ones.
        if jpeg and _last_written.get(camera_id) is not jpeg:
            _write_atomic(root / "snapshots" / f"{camera_id}.jpg", jpeg)
            _last_written[camera_id] = jpeg


def _owner_loop() -> None:
    from django.db import close_old_connections

    from config.db import release_db
    from recognition.services.cctv_worker import get_cctv_manager

    manager = get_cctv_manager()
    while True:
        try:
            close_old_connections()
            if _drain_commands(manager):
                try:
                    from realtime.sio_app import emit_invalidate

                    emit_invalidate(["attendance", "recognition", "hr"])
                except Exception:
                    pass
            _publish_status(manager)
        except Exception:
            logger.exception("CCTV owner loop iteration failed")
        finally:
            release_db()
        time.sleep(POLL_SECONDS)


def start_owner_loop() -> None:
    global _owner_thread
    with _owner_guard:
        if _owner_thread and _owner_thread.is_alive():
            return
        _owner_thread = threading.Thread(target=_owner_loop, name="cctv-owner-control", daemon=True)
        _owner_thread.start()


class RemoteRuntime:
    """Read-only view of the owner's CCTV workers, for processes that do not run them."""

    def __init__(self):
        self._doc = None

    def _load(self) -> dict:
        if self._doc is None:
            try:
                doc = json.loads((_base_dir() / "status.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                doc = {}
            if time.time() - float(doc.get("at") or 0) > STATUS_MAX_AGE:
                doc = {}  # owner is gone or stalled; report nothing rather than stale "running"
            self._doc = doc
        return self._doc

    def list_status(self) -> list[dict]:
        return list(self._load().get("statuses") or [])

    def get_status(self, camera_id: int) -> dict | None:
        for status in self.list_status():
            if int(status.get("camera_id") or -1) == int(camera_id):
                return status
        return None

    def get_events(self, camera_id: int | None = None, limit: int = 40) -> list[dict]:
        if camera_id is not None:
            status = self.get_status(camera_id) or {}
            return list(status.get("recent_events") or [])[:limit]
        return list(self._load().get("events") or [])[:limit]

    def get_snapshot_jpeg(self, camera_id: int) -> bytes | None:
        path = _base_dir() / "snapshots" / f"{int(camera_id)}.jpg"
        try:
            if time.time() - path.stat().st_mtime > STATUS_MAX_AGE:
                return None
            return path.read_bytes()
        except OSError:
            return None


def get_runtime():
    """CCTVWorkerManager in the owning process, otherwise a read-only RemoteRuntime."""
    if is_owner():
        from recognition.services.cctv_worker import get_cctv_manager

        return get_cctv_manager()
    return RemoteRuntime()


def request(action: str, camera_id: int | None = None) -> dict:
    """
    Run a start/stop request in the right process.

    Returns {"local": True} when this process owns (or just claimed) the workers and the caller should act on
    its manager directly, else {"local": False, "queued": True} after handing the request to the owner.
    """
    if is_owner() or (action in ("start", "start_all") and claim_ownership()):
        return {"local": True}
    submit_command(action, camera_id)
    return {"local": False, "queued": True}

