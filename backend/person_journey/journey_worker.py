"""Sync cameras to ML journey pipeline and start ML journey processing."""

from __future__ import annotations

from ml.client import ml_auth_headers
import logging
import threading
from collections import defaultdict

from django.conf import settings

logger = logging.getLogger(__name__)

_stop = threading.Event()
_thread: threading.Thread | None = None
# ML nodes that got cameras last round: a node that loses all of them is sent one empty list to clear it.
_last_targets: set[str] = set()
LOCK_NAME = "person_journey_sync"


def _interval() -> int:
    return max(10, int(getattr(settings, "PERSON_JOURNEY_SYNC_INTERVAL_SEC", 60)))


def _journey_ingest_payload(entries: list[dict]) -> dict:
    backend_url = getattr(settings, "PERSON_JOURNEY_BACKEND_URL", "http://127.0.0.1:8000")
    ingest_token = getattr(settings, "PERSON_JOURNEY_INGEST_TOKEN", "")
    return {
        "cameras": entries,
        "backend_ingest_url": f"{backend_url.rstrip('/')}/api/person-journey/ingest/",
        "ingest_token": ingest_token,
    }


def _post_journey_bulk(base_url: str, entries: list[dict]) -> dict:
    import requests

    res = requests.post(
        f"{base_url.rstrip('/')}/journey/register/bulk",
        json=_journey_ingest_payload(entries),
        timeout=30,
        headers=ml_auth_headers(),
    )
    res.raise_for_status()
    return res.json()


def sync_cameras_to_journey_ml() -> dict:
    from cameras.models import Camera
    from ml.client import camera_ml_base_url, ml_service_enabled

    if not ml_service_enabled():
        return {"synced": 0, "reason": "ml_disabled"}

    cameras = Camera.objects.filter(
        is_active=True,
        nvr__isnull=False,
        ml_server_id__isnull=False,
    ).select_related("nvr", "nvr__site", "ml_server")

    by_url: dict[str, list[dict]] = defaultdict(list)
    for cam in cameras:
        try:
            base = camera_ml_base_url(cam)
            if not base:
                continue
            by_url[base].append(
                {
                    "key": cam.stream_key,
                    "rtsp_url": cam.effective_stream_url(),
                    "camera_id": cam.pk,
                    "zone": cam.zone or "",
                    "name": cam.name,
                }
            )
        except Exception as exc:
            logger.warning("Journey sync skip camera %s: %s", cam.pk, exc)

    global _last_targets
    # Only nodes that own cameras (plus nodes that just lost their last one, once) — not every known node.
    owning = set(by_url.keys())
    targets = owning | (_last_targets - owning)
    totals = {"synced": 0, "by_server": {}, "cleared": []}
    failed: set[str] = set()

    for base in sorted(targets):
        entries = by_url.get(base, [])
        try:
            data = _post_journey_bulk(base, entries)
            registered = int(data.get("registered") or data.get("synced") or len(entries) or 0)
            totals["synced"] += len(entries)
            totals["by_server"][base] = {
                "cameras": len(entries),
                "result": data,
                "registered": registered,
            }
            if not entries:
                totals["cleared"].append(base)
        except Exception as exc:
            logger.warning(
                "Journey ML sync failed at %s (restart ml_services api_server.py): %s",
                base,
                exc,
            )
            totals["by_server"][base] = {"cameras": len(entries), "error": str(exc)}
            failed.add(base)

    # Keep a node that failed to clear in the list so the next round retries it.
    _last_targets = owning | {b for b in failed if b not in owning}
    return totals


def _worker_loop():
    from django.db import close_old_connections

    from config.db import release_db

    while not _stop.is_set():
        close_old_connections()
        try:
            sync_cameras_to_journey_ml()
        except Exception:
            logger.exception("Journey sync worker iteration failed")
        try:
            from .identity import consolidate_recent_persons

            merged = consolidate_recent_persons()
            if merged:
                logger.info("Person journey: merged %s split identities", merged)
        except Exception:
            logger.exception("Person journey identity consolidation failed")
        finally:
            release_db()
        from config.worker_throttle import maybe_pause_for_cpu

        maybe_pause_for_cpu(logger, label="journey-sync")
        _stop.wait(_interval())


def _start_thread() -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    # The loop's first iteration is the initial ML sync: nothing blocks AppConfig.ready / server boot.
    _thread = threading.Thread(target=_worker_loop, daemon=True, name="person-journey-sync")
    _thread.start()
    logger.info("Person journey camera sync worker started (interval=%ss)", _interval())


def start_journey_worker_thread():
    """Camera sync + identity consolidation — in ONE process only (cross-process file lock)."""
    from recognition.services.autostart import run_with_process_lock

    run_with_process_lock(LOCK_NAME, _start_thread)
