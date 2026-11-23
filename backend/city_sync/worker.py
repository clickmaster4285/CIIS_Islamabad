"""City Outbox worker — push pending events to HQ (Phase 3)."""

from __future__ import annotations

import logging
import random
from datetime import timedelta

import requests
from django.conf import settings
from django.db import transaction
from django.db.models import Exists, OuterRef, Q
from django.utils import timezone

from city_sync.identity import identity_dict, is_city, location_code
from city_sync.models import SyncOutbox

logger = logging.getLogger(__name__)

BACKOFF_SECONDS = (5, 15, 60, 300, 900)  # then cap 15m


def _backoff(attempts: int) -> timedelta:
    idx = min(max(attempts - 1, 0), len(BACKOFF_SECONDS) - 1)
    base = BACKOFF_SECONDS[idx] if attempts <= len(BACKOFF_SECONDS) else 900
    jitter = base * random.uniform(0, 0.2)
    return timedelta(seconds=base + jitter)


def _load_api_key() -> str:
    path = getattr(settings, "SYNC_API_KEY_FILE", "") or ""
    if path:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError as exc:
            raise RuntimeError(f"Cannot read SYNC_API_KEY_FILE={path}: {exc}") from exc
    key = getattr(settings, "SYNC_API_KEY", "") or ""
    if not key:
        # Fall back to map entry for this location
        raw = getattr(settings, "SYNC_API_KEYS", "") or ""
        loc = location_code()
        for part in str(raw).split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip().upper() == loc:
                    return v.strip()
    return key.strip()


def warn_if_insecure_hq_url() -> None:
    """Log once per process when HQ_SYNC_URL is not https (LAN http is allowed, but keys travel in clear)."""
    global _INSECURE_URL_WARNED
    url = (getattr(settings, "HQ_SYNC_URL", "") or "").strip()
    if _INSECURE_URL_WARNED or not url:
        return
    _INSECURE_URL_WARNED = True
    if not url.lower().startswith("https://"):
        logger.warning(
            "HQ_SYNC_URL=%s is not https: the sync API key and data are sent unencrypted. "
            "Use https unless this link is a trusted private network.",
            url,
        )


_INSECURE_URL_WARNED = False


def recover_stuck_sending(*, older_than_seconds: int = 300) -> int:
    """Rows a crashed worker left in "sending" go back to pending (keyed on when they were claimed)."""
    cutoff = timezone.now() - timedelta(seconds=older_than_seconds)
    return (
        SyncOutbox.objects.filter(status=SyncOutbox.STATUS_SENDING)
        .filter(Q(claimed_at__lt=cutoff) | Q(claimed_at__isnull=True, next_attempt_at__lt=cutoff))
        .update(status=SyncOutbox.STATUS_PENDING, next_attempt_at=timezone.now(), claimed_at=None)
    )


def claim_batch(*, limit: int, min_age_seconds: float) -> list[SyncOutbox]:
    """
    Claim the oldest due rows. A row is held back while an earlier seq of the same entity
    is still in flight or backing off, so HQ always receives one entity's versions in order.
    Earlier rows that are due go in the same batch (HQ applies a batch in seq order).
    Dead rows do not block later versions (HQ accepts a newer full-state version).
    """
    now = timezone.now()
    min_created = now - timedelta(seconds=min_age_seconds)
    blocking = SyncOutbox.objects.filter(
        entity_type=OuterRef("entity_type"),
        entity_id=OuterRef("entity_id"),
        seq__lt=OuterRef("seq"),
    ).filter(
        Q(status=SyncOutbox.STATUS_SENDING)
        | Q(status=SyncOutbox.STATUS_PENDING, next_attempt_at__gt=now)
        | Q(status=SyncOutbox.STATUS_PENDING, created_at__gt=min_created)
    )
    with transaction.atomic():
        qs = (
            SyncOutbox.objects.select_for_update(skip_locked=True)
            .filter(
                status=SyncOutbox.STATUS_PENDING,
                next_attempt_at__lte=now,
                created_at__lte=min_created,
            )
            .exclude(Exists(blocking))
            .order_by("seq")[:limit]
        )
        rows = list(qs)
        # An earlier row of the same entity that another worker holds locked (skip_locked
        # hid it) must still hold back the later versions claimed here.
        first_seq: dict[tuple, int] = {}
        for row in rows:
            first_seq.setdefault((row.entity_type, row.entity_id), row.seq)
        kept = rows
        if rows:
            earlier = (
                SyncOutbox.objects.filter(
                    status__in=[SyncOutbox.STATUS_PENDING, SyncOutbox.STATUS_SENDING],
                    entity_id__in={k[1] for k in first_seq},
                    seq__lt=max(first_seq.values()),
                )
                .exclude(seq__in=[r.seq for r in rows])
                .values_list("entity_type", "entity_id", "seq")
            )
            blocked = {(t, e) for t, e, sq in earlier if sq < first_seq.get((t, e), 0)}
            kept = [r for r in rows if (r.entity_type, r.entity_id) not in blocked]
        for row in kept:
            row.status = SyncOutbox.STATUS_SENDING
            row.claimed_at = now
        if kept:
            SyncOutbox.objects.bulk_update(kept, ["status", "claimed_at"])
        return kept


def _serialize_event(row: SyncOutbox) -> dict:
    return {
        "seq": row.seq,
        "event_id": str(row.event_id),
        "location_code": row.location_code,
        "entity_type": row.entity_type,
        "entity_id": str(row.entity_id),
        "operation": row.operation,
        "entity_version": row.entity_version,
        "schema_version": row.schema_version,
        "payload": row.payload,
        "payload_hash": row.payload_hash,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _retry_later(row: SyncOutbox, error: str, now, *, count_attempt: bool = True) -> None:
    """Back to pending with a capped backoff. Transient failures never dead-letter."""
    if count_attempt:
        row.attempts = int(row.attempts or 0) + 1
    row.last_error = (error or "retry")[:2000]
    row.status = SyncOutbox.STATUS_PENDING
    row.claimed_at = None
    row.next_attempt_at = now + _backoff(max(int(row.attempts or 0), 1))
    row.save(update_fields=["attempts", "last_error", "status", "claimed_at", "next_attempt_at"])


def _dead_letter(row: SyncOutbox, error: str, now) -> None:
    row.status = SyncOutbox.STATUS_DEAD
    row.last_error = (error or "rejected")[:2000]
    row.done_at = now
    row.claimed_at = None
    row.save(update_fields=["status", "last_error", "done_at", "claimed_at"])


def _apply_hq_results(rows: list[SyncOutbox], results: list[dict]) -> None:
    by_id = {str(r.event_id): r for r in rows}
    now = timezone.now()
    answered: set[str] = set()

    for item in results:
        if not isinstance(item, dict):
            continue
        event_id = str(item.get("event_id") or "")
        row = by_id.get(event_id)
        if not row or event_id in answered:
            continue
        answered.add(event_id)
        result = str(item.get("result") or "").lower()
        detail = str(item.get("detail") or "")

        if result in {"applied", "duplicate", "stale", "conflict"}:
            row.status = SyncOutbox.STATUS_DONE
            row.done_at = now
            row.last_error = detail
            row.schema_hold = False
            row.claimed_at = None
            row.save(update_fields=["status", "done_at", "last_error", "schema_hold", "claimed_at"])
            continue

        if result == "rejected":
            # HQ validated the event and refused it: retrying cannot help (permanent 4xx-style).
            _dead_letter(row, detail or "rejected", now)
            logger.error("Outbox event %s rejected: %s", event_id, detail)
            continue

        if result == "retry" or item.get("schema_hold"):
            # Do NOT increment toward dead.
            row.status = SyncOutbox.STATUS_PENDING
            row.schema_hold = detail == "schema_unsupported" or bool(item.get("schema_hold"))
            row.claimed_at = None
            row.next_attempt_at = now + timedelta(minutes=5)
            row.last_error = detail or "retry"
            row.save(update_fields=["status", "schema_hold", "claimed_at", "next_attempt_at", "last_error"])
            continue

        # Unknown / retryable (e.g. an HQ-side error for this one event): keep retrying.
        _retry_later(row, detail or result or "retry", now)

    # Events HQ did not answer must not stay "sending" until stuck recovery.
    for event_id, row in by_id.items():
        if event_id not in answered:
            _retry_later(row, "no_result_from_hq", now)


def _requeue_network_failure(rows: list[SyncOutbox], error: str) -> None:
    """Network errors, HQ 5xx, auth failures, bad responses: retry forever at the capped backoff."""
    now = timezone.now()
    for row in rows:
        _retry_later(row, error, now)


def _reject_batch(rows: list[SyncOutbox], error: str) -> None:
    """HQ refused the whole request as invalid (400/422). Retry a few times, then dead-letter."""
    now = timezone.now()
    max_attempts = int(getattr(settings, "OUTBOX_MAX_ATTEMPTS_BEFORE_DEAD", 20))
    for row in rows:
        if int(row.attempts or 0) + 1 >= max_attempts and not row.schema_hold:
            row.attempts = int(row.attempts or 0) + 1
            _dead_letter(row, error, now)
            logger.error("Outbox event %s dead after %s invalid-request answers", row.event_id, row.attempts)
        else:
            _retry_later(row, error, now)


# Request-level answers that mean "this request is invalid" rather than "try again later".
_PERMANENT_HTTP = {400, 422}


def push_once() -> dict:
    """One poll cycle. Returns stats."""
    if not is_city():
        return {"skipped": True, "reason": "not_city"}

    hq_url = (getattr(settings, "HQ_SYNC_URL", "") or "").rstrip("/")
    if not hq_url:
        return {"skipped": True, "reason": "HQ_SYNC_URL empty"}
    warn_if_insecure_hq_url()

    from city_sync.control import city_is_paused

    if city_is_paused():
        return {"skipped": True, "reason": "paused_by_hq"}

    recover_stuck_sending()
    batch_size = int(getattr(settings, "OUTBOX_BATCH_SIZE", 200))
    min_age = float(getattr(settings, "OUTBOX_MIN_AGE_SECONDS", 2))
    rows = claim_batch(limit=batch_size, min_age_seconds=min_age)
    if not rows:
        return {"claimed": 0}

    payload = {
        "location_code": location_code(),
        "node": identity_dict(),
        "events": [_serialize_event(r) for r in rows],
    }
    api_key = _load_api_key()
    timeout = float(getattr(settings, "OUTBOX_HTTP_TIMEOUT", 30))
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-CIIS-Location": location_code(),
    }
    url = f"{hq_url}/events"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code >= 500:
            _requeue_network_failure(rows, f"HTTP {resp.status_code}: {resp.text[:500]}")
            return {"claimed": len(rows), "error": f"http_{resp.status_code}"}
        if resp.status_code in {401, 403}:
            # Usually a key being rotated: keep the events and retry.
            _requeue_network_failure(rows, f"auth_failed HTTP {resp.status_code}")
            return {"claimed": len(rows), "error": "auth_failed"}
        if resp.status_code in _PERMANENT_HTTP:
            _reject_batch(rows, f"HTTP {resp.status_code}: {resp.text[:500]}")
            return {"claimed": len(rows), "error": f"http_{resp.status_code}"}
        if resp.status_code >= 400:
            # 404/408/409/413/429 ...: configuration or load problems, not bad data.
            _requeue_network_failure(rows, f"HTTP {resp.status_code}: {resp.text[:500]}")
            return {"claimed": len(rows), "error": f"http_{resp.status_code}"}
        try:
            data = resp.json() if resp.content else {}
        except ValueError:
            data = {}
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            _requeue_network_failure(rows, f"bad_response HTTP {resp.status_code}")
            return {"claimed": len(rows), "error": "bad_response"}
        _apply_hq_results(rows, results)
        return {"claimed": len(rows), "results": len(results)}
    except requests.RequestException as exc:
        _requeue_network_failure(rows, str(exc))
        return {"claimed": len(rows), "error": str(exc)}


def outbox_stats() -> dict:
    pending = SyncOutbox.objects.filter(status=SyncOutbox.STATUS_PENDING).count()
    dead = SyncOutbox.objects.filter(status=SyncOutbox.STATUS_DEAD).count()
    oldest = (
        SyncOutbox.objects.filter(status=SyncOutbox.STATUS_PENDING)
        .order_by("created_at")
        .values_list("created_at", flat=True)
        .first()
    )
    age = int((timezone.now() - oldest).total_seconds()) if oldest else 0
    return {
        "pending_outbox": pending,
        "dead_outbox": dead,
        "oldest_pending_age_sec": age,
    }


def send_heartbeat() -> dict:
    if not is_city():
        return {"skipped": True}
    hq_url = (getattr(settings, "HQ_SYNC_URL", "") or "").rstrip("/")
    if not hq_url:
        return {"skipped": True, "reason": "no_url"}
    warn_if_insecure_hq_url()

    from city_sync.control import apply_hq_control, heartbeat_extras
    from city_sync.media_sync import media_stats
    from city_sync.models import BackupRecord, SyncOutbox

    last_backup = (
        BackupRecord.objects.filter(location_code=location_code(), success=True)
        .order_by("-completed_at")
        .values_list("completed_at", flat=True)
        .first()
    )
    last_biz = (
        SyncOutbox.objects.filter(location_code=location_code())
        .order_by("-created_at")
        .values_list("created_at", flat=True)
        .first()
    )
    disk_percent = None
    try:
        import shutil

        usage = shutil.disk_usage(str(getattr(settings, "MEDIA_ROOT", ".") or "."))
        disk_percent = round((usage.used / usage.total) * 100.0, 2) if usage.total else None
    except Exception:  # noqa: BLE001
        disk_percent = None

    body = {
        **identity_dict(),
        **outbox_stats(),
        **media_stats(),
        "disk_percent": disk_percent,
        "last_backup_at": last_backup.isoformat() if last_backup else None,
        "last_business_event_at": last_biz.isoformat() if last_biz else None,
        **heartbeat_extras(),
    }
    api_key = _load_api_key()
    try:
        resp = requests.post(
            f"{hq_url}/heartbeat",
            json=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=10,
        )
        control = None
        if resp.ok:
            try:
                control = (resp.json() or {}).get("control")
            except ValueError:
                control = None
        if control is not None:
            apply_hq_control(control)
        return {"status_code": resp.status_code, "control": control}
    except requests.RequestException as exc:
        return {"error": str(exc)}
