"""Phase 5 — city media queue + HQ chunked upload / verify."""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import requests
from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.models import Q
from django.utils import timezone

from city_sync.audit import audit
from city_sync.identity import identity_dict, is_city, is_hq, location_code
from city_sync.models import HqMediaObject, MediaQueue, SyncOutbox
from city_sync.worker import _backoff, _load_api_key, warn_if_insecure_hq_url

logger = logging.getLogger(__name__)

# Extensions HQ keeps on stored files (served from MEDIA_URL). Anything else (e.g. .html,
# .svg, .js) is stored as .bin so a city path can never plant an active page at HQ.
ALLOWED_MEDIA_SUFFIXES = frozenset(
    {
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff",
        ".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v", ".3gp",
        ".mp3", ".wav", ".ogg", ".m4a",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".csv", ".txt", ".rtf", ".zip",
    }
)

# HTTP answers that mean "HQ cannot take it right now" rather than "this file is bad".
_TRANSIENT_HTTP = {401, 403, 408, 425, 429}


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def enqueue_media_file(
    *,
    absolute_or_media_path: str,
    record_entity_type: str = "",
    record_entity_id: UUID | str | None = None,
    priority: int = MediaQueue.PRIORITY_ROUTINE,
    wait_outbox_done: bool = True,
) -> MediaQueue | None:
    """Register a local file for upload to HQ (city only). The same file of a record is queued once."""
    if not is_city() or not getattr(settings, "SYNC_MEDIA_ENABLED", True):
        return None

    path = Path(absolute_or_media_path)
    if not path.is_file():
        # Try MEDIA_ROOT-relative
        media_root = Path(getattr(settings, "MEDIA_ROOT", "") or "")
        candidate = media_root / absolute_or_media_path
        if candidate.is_file():
            path = candidate
        else:
            logger.warning("enqueue_media_file: missing file %s", absolute_or_media_path)
            return None

    digest, size = sha256_file(path)
    rel = str(absolute_or_media_path)
    if str(path).startswith(str(getattr(settings, "MEDIA_ROOT", ""))):
        try:
            rel = str(path.relative_to(settings.MEDIA_ROOT)).replace("\\", "/")
        except Exception:  # noqa: BLE001
            rel = str(path)

    entity_uuid = UUID(str(record_entity_id)) if record_entity_id else None
    existing = MediaQueue.objects.filter(record_entity_id=entity_uuid, path=rel, sha256=digest).first()
    if existing is not None:
        return existing

    row = MediaQueue.objects.create(
        location_code=location_code(),
        record_entity_type=record_entity_type or "",
        record_entity_id=entity_uuid,
        path=rel,
        sha256=digest,
        size_bytes=size,
        priority=priority,
        wait_outbox_done=wait_outbox_done,
    )
    audit("media_enqueued", detail={"media_id": str(row.media_id), "path": rel, "sha256": digest})
    return row


def _resolve_local_path(rel: str) -> Path | None:
    p = Path(rel)
    if p.is_file():
        return p
    media_root = Path(getattr(settings, "MEDIA_ROOT", "") or "")
    candidate = media_root / rel
    return candidate if candidate.is_file() else None


def _outbox_ready(row: MediaQueue) -> bool:
    if not row.wait_outbox_done or not row.record_entity_id:
        return True
    pending = SyncOutbox.objects.filter(
        entity_id=row.record_entity_id,
        status__in=[SyncOutbox.STATUS_PENDING, SyncOutbox.STATUS_SENDING],
    ).exists()
    return not pending


def _hq_media_base() -> str:
    return (getattr(settings, "HQ_SYNC_URL", "") or "").rstrip("/")


def _claim(limit: int, stats: dict) -> list[tuple[MediaQueue, Path]]:
    """
    Lock and claim up to `limit` due rows (skip rows another worker holds). An "uploading"
    row is only taken over once its claim is older than SYNC_MEDIA_STALE_UPLOAD_SEC.
    """
    now = timezone.now()
    stale = now - timedelta(seconds=int(getattr(settings, "SYNC_MEDIA_STALE_UPLOAD_SEC", 1800)))
    due = Q(next_attempt_at__isnull=True) | Q(next_attempt_at__lte=now)
    claimable = (Q(status=MediaQueue.STATUS_PENDING) & due) | (
        Q(status=MediaQueue.STATUS_UPLOADING) & (Q(claimed_at__isnull=True) | Q(claimed_at__lt=stale))
    )
    ready: list[tuple[MediaQueue, Path]] = []
    with transaction.atomic():
        # Newest first: fresh detections get their picture at HQ right away; a backlog
        # (e.g. full history) fills in behind them. Extra candidates so rows still waiting
        # for their Outbox event do not stall the queue.
        candidates = list(
            MediaQueue.objects.select_for_update(skip_locked=True)
            .filter(claimable)
            .order_by("priority", "-created_at")[: max(limit, 1) * 4]
        )
        for row in candidates:
            if len(ready) >= limit:
                break
            if not _outbox_ready(row):
                stats["waiting_outbox"] += 1
                continue
            local = _resolve_local_path(row.path)
            if local is None:
                row.status = MediaQueue.STATUS_FAILED
                row.last_error = "local_file_missing"
                row.attempts += 1
                row.claimed_at = None
                row.save(update_fields=["status", "last_error", "attempts", "claimed_at"])
                stats["failed"] += 1
                continue
            row.status = MediaQueue.STATUS_UPLOADING
            row.claimed_at = now
            row.save(update_fields=["status", "claimed_at"])
            ready.append((row, local))
    return ready


def push_media_once(*, limit: int = 5) -> dict:
    if not is_city():
        return {"skipped": True, "reason": "not_city"}
    if not getattr(settings, "SYNC_MEDIA_ENABLED", True):
        return {"skipped": True, "reason": "disabled"}
    base = _hq_media_base()
    if not base:
        return {"skipped": True, "reason": "no_url"}
    warn_if_insecure_hq_url()
    from city_sync.control import city_is_paused

    if city_is_paused():
        return {"skipped": True, "reason": "paused_by_hq"}

    chunk_size = int(getattr(settings, "SYNC_MEDIA_CHUNK_SIZE", 4 * 1024 * 1024))
    max_attempts = int(getattr(settings, "SYNC_MEDIA_MAX_ATTEMPTS", 30))
    api_key = _load_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}

    stats = {"attempted": 0, "verified": 0, "failed": 0, "waiting_outbox": 0, "retry_later": 0}
    ready = _claim(limit, stats)

    # Transfers run in parallel; every database update stays on this thread.
    stats["attempted"] = len(ready)
    workers = max(1, int(getattr(settings, "SYNC_MEDIA_PARALLEL", 4)))
    with ThreadPoolExecutor(max_workers=min(workers, len(ready) or 1)) as pool:
        outcomes = list(
            pool.map(
                lambda item: _transfer(item[0], item[1], base=base, headers=headers, chunk_size=chunk_size),
                ready,
            )
        )

    now = timezone.now()
    for (row, _local), (error, transient) in zip(ready, outcomes):
        if error is None:
            row.status = MediaQueue.STATUS_VERIFIED
            row.verified_at = now
            row.bytes_uploaded = row.size_bytes
            row.last_error = ""
            row.claimed_at = None
            row.transient_failures = 0
            row.save(
                update_fields=["status", "verified_at", "bytes_uploaded", "last_error", "claimed_at", "transient_failures"]
            )
            stats["verified"] += 1
            audit("media_verified", detail={"media_id": str(row.media_id)})
            continue
        row.last_error = error[:2000]
        row.claimed_at = None
        if transient:
            # HQ unreachable / overloaded / key rotating: back off, never count toward FAILED.
            row.transient_failures += 1
            row.status = MediaQueue.STATUS_PENDING
            row.next_attempt_at = now + _backoff(row.transient_failures)
            stats["retry_later"] += 1
        else:
            row.attempts += 1
            if row.attempts >= max_attempts:
                row.status = MediaQueue.STATUS_FAILED
                stats["failed"] += 1
            else:
                row.status = MediaQueue.STATUS_PENDING
                row.next_attempt_at = now + _backoff(row.attempts)
        row.save(
            update_fields=["attempts", "transient_failures", "last_error", "status", "claimed_at", "next_attempt_at"]
        )
        logger.warning("media upload failed media_id=%s: %s", row.media_id, error)
    return stats


def _http_error(step: str, resp) -> tuple[str, bool]:
    transient = resp.status_code >= 500 or resp.status_code in _TRANSIENT_HTTP
    return f"{step} HTTP {resp.status_code}: {resp.text[:300]}", transient


def _transfer(row: MediaQueue, local: Path, *, base: str, headers: dict, chunk_size: int) -> tuple[str | None, bool]:
    """
    Send one file to HQ (init, chunks, complete); no database access.
    Returns (None, False) on success, else (error, transient).
    """
    try:
        init = requests.post(
            f"{base}/media/init",
            json={
                "media_id": str(row.media_id),
                "location_code": row.location_code,
                "sha256": row.sha256,
                "size_bytes": row.size_bytes,
                "record_entity_type": row.record_entity_type,
                "record_entity_id": str(row.record_entity_id) if row.record_entity_id else None,
                "source_path": row.path,
            },
            headers=headers,
            timeout=30,
        )
        if init.status_code >= 400:
            return _http_error("init", init)
        offset = int((init.json() or {}).get("offset") or 0)

        with open(local, "rb") as fh:
            fh.seek(offset)
            pos = offset
            while pos < row.size_bytes:
                chunk = fh.read(min(chunk_size, row.size_bytes - pos))
                if not chunk:
                    break
                put = requests.put(
                    f"{base}/media/{row.media_id}/chunk",
                    data=chunk,
                    headers={
                        **headers,
                        "Content-Type": "application/octet-stream",
                        "X-CIIS-Offset": str(pos),
                        "X-CIIS-Location": row.location_code,
                    },
                    timeout=120,
                )
                if put.status_code >= 400:
                    return _http_error("chunk", put)
                pos += len(chunk)

        complete = requests.post(
            f"{base}/media/{row.media_id}/complete",
            json={"sha256": row.sha256, "location_code": row.location_code},
            headers=headers,
            timeout=60,
        )
        try:
            body = complete.json() if complete.content else {}
        except ValueError:
            body = {}
        if complete.status_code >= 500 or complete.status_code in _TRANSIENT_HTTP:
            return f"complete HTTP {complete.status_code}", True
        if complete.status_code >= 400 or not body.get("verified"):
            return f"complete failed: HTTP {complete.status_code} {body}", False
        return None, False
    except requests.RequestException as exc:
        return (str(exc) or exc.__class__.__name__), True
    except Exception as exc:  # noqa: BLE001
        return (str(exc) or exc.__class__.__name__), False


def requeue_failed_media(*, include_missing: bool = False) -> int:
    """City: FAILED uploads back to pending with a fresh attempt budget."""
    qs = MediaQueue.objects.filter(status=MediaQueue.STATUS_FAILED)
    if not include_missing:
        qs = qs.exclude(last_error="local_file_missing")
    return qs.update(
        status=MediaQueue.STATUS_PENDING,
        attempts=0,
        transient_failures=0,
        next_attempt_at=timezone.now(),
        claimed_at=None,
    )


def hq_media_root_for(loc: str) -> Path:
    root = Path(getattr(settings, "SYNC_HQ_MEDIA_ROOT", "") or (Path(settings.MEDIA_ROOT) / "hq_sync_media"))
    root.mkdir(parents=True, exist_ok=True)
    now = timezone.now()
    dest = root / loc.upper() / f"{now.year:04d}" / f"{now.month:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _safe_suffix(source_path: str) -> str:
    suffix = Path(source_path or "").suffix.lower()
    if not suffix:
        return ""
    return suffix if suffix in ALLOWED_MEDIA_SUFFIXES else ".bin"


def hq_media_init(
    *, loc: str, media_id: str, sha256: str, size_bytes: int, entity_type: str, entity_id, source_path: str = ""
) -> dict:
    if not is_hq():
        raise PermissionError("HQ only")
    loc = loc.upper()
    mid = UUID(str(media_id))
    size_bytes = int(size_bytes)
    if size_bytes < 0:
        raise ValueError("size_bytes must not be negative")
    obj = HqMediaObject.objects.select_for_update().filter(media_id=mid).first()
    created = False
    if obj is None:
        try:
            with transaction.atomic():
                obj = HqMediaObject.objects.create(
                    media_id=mid,
                    location_code=loc,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    record_entity_type=entity_type or "",
                    record_entity_id=UUID(str(entity_id)) if entity_id else None,
                    source_path=(source_path or "")[:512],
                    status=HqMediaObject.STATUS_PARTIAL,
                )
            created = True
        except IntegrityError:
            obj = HqMediaObject.objects.select_for_update().get(media_id=mid)
    # A media id is owned by the city that registered it; another city can never touch it.
    if obj.location_code != loc:
        raise PermissionError("media_id belongs to another location")
    # Verified objects are immutable.
    if obj.status == HqMediaObject.STATUS_VERIFIED:
        return {"offset": obj.size_bytes, "verified": True}
    if source_path and obj.source_path != source_path[:512]:
        obj.source_path = source_path[:512]
        obj.save(update_fields=["source_path"])
    if not created and (obj.sha256 != sha256 or obj.size_bytes != size_bytes):
        # Restart
        if obj.storage_path and Path(obj.storage_path).exists():
            Path(obj.storage_path).unlink(missing_ok=True)
        obj.sha256 = sha256
        obj.size_bytes = size_bytes
        obj.bytes_received = 0
        obj.status = HqMediaObject.STATUS_PARTIAL
        obj.storage_path = ""
        obj.save()
    if not obj.storage_path:
        dest = hq_media_root_for(loc) / f"{mid}.partial"
        dest.touch(exist_ok=True)
        obj.storage_path = str(dest)
        obj.status = HqMediaObject.STATUS_PARTIAL
        obj.save(update_fields=["storage_path", "status"])
    offset = int(obj.bytes_received or 0)
    if obj.storage_path and Path(obj.storage_path).exists():
        offset = Path(obj.storage_path).stat().st_size
        obj.bytes_received = offset
        obj.save(update_fields=["bytes_received"])
    return {"offset": offset, "verified": False}


def hq_media_chunk(*, loc: str, media_id: str, offset: int, data: bytes) -> dict:
    obj = HqMediaObject.objects.select_for_update().get(media_id=UUID(str(media_id)), location_code=loc.upper())
    if obj.status == HqMediaObject.STATUS_VERIFIED:
        return {"bytes_received": obj.size_bytes}
    if not obj.storage_path:
        raise ValueError("upload not initialised")
    path = Path(obj.storage_path)
    current = path.stat().st_size if path.exists() else 0
    if int(offset) != current:
        raise ValueError(f"offset mismatch expected={current} got={offset}")
    if current + len(data) > int(obj.size_bytes or 0):
        raise ValueError(f"chunk exceeds declared size {obj.size_bytes}")
    with open(path, "ab") as fh:
        fh.write(data)
    obj.bytes_received = current + len(data)
    obj.save(update_fields=["bytes_received"])
    return {"bytes_received": obj.bytes_received}


def hq_media_complete(*, loc: str, media_id: str, sha256: str) -> dict:
    obj = HqMediaObject.objects.select_for_update().get(media_id=UUID(str(media_id)), location_code=loc.upper())
    if obj.status == HqMediaObject.STATUS_VERIFIED:
        return {"verified": True, "storage_path": obj.storage_path}
    path = Path(obj.storage_path) if obj.storage_path else None
    if path is None or not path.exists():
        obj.status = HqMediaObject.STATUS_FAILED
        obj.save(update_fields=["status"])
        return {"verified": False, "detail": "missing_file"}
    digest, size = sha256_file(path)
    if digest != sha256 or digest != obj.sha256:
        path.unlink(missing_ok=True)
        obj.bytes_received = 0
        obj.storage_path = ""
        obj.status = HqMediaObject.STATUS_FAILED
        obj.save(update_fields=["bytes_received", "storage_path", "status"])
        audit("media_checksum_mismatch", detail={"media_id": str(media_id), "got": digest}, loc=loc)
        return {"verified": False, "detail": "checksum_mismatch"}

    # Drop .partial; keep the original extension when it is a known media/document type.
    suffix = _safe_suffix(obj.source_path)
    final = path.with_suffix("").with_suffix(suffix) if suffix else path.with_suffix("")
    if final.exists():
        final.unlink()
    path.replace(final)
    # Best-effort immutability on Windows may fail — ignore.
    try:
        os.chmod(final, 0o444)
    except Exception:  # noqa: BLE001
        pass
    obj.storage_path = str(final)
    obj.size_bytes = size
    obj.bytes_received = size
    obj.status = HqMediaObject.STATUS_VERIFIED
    obj.verified_at = timezone.now()
    obj.save()
    audit("hq_media_verified", detail={"media_id": str(media_id), "path": str(final)}, loc=loc)
    from city_sync.materialize import attach_media

    try:
        attach_media(obj)
    except Exception:  # noqa: BLE001
        logger.exception("attach_media failed for %s", media_id)
    return {"verified": True, "storage_path": str(final)}


def media_stats() -> dict:
    return {
        "pending_media": MediaQueue.objects.filter(
            status__in=[MediaQueue.STATUS_PENDING, MediaQueue.STATUS_UPLOADING]
        ).count(),
        "failed_media": MediaQueue.objects.filter(status=MediaQueue.STATUS_FAILED).count(),
    }
