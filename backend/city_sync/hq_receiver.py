"""HQ Sync API application logic (Phase 4) — idempotent, ordered, version-aware."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from django.conf import settings
from django.db import IntegrityError, transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from city_sync.models import HqConflict, HqEventLog, HqReplicaRow, HqSyncCursor
from city_sync.outbox import payload_hash
from city_sync.syncable import LEGAL_ENTITY_TYPES

logger = logging.getLogger(__name__)

RESULT_APPLIED = "applied"
RESULT_DUPLICATE = "duplicate"
RESULT_STALE = "stale"
RESULT_REJECTED = "rejected"
RESULT_CONFLICT = "conflict"
RESULT_RETRY = "retry"  # schema not yet supported — city must not dead-letter
RESULT_ERROR = "error"  # transient HQ-side failure — the city retries with backoff

# HqConflict.reason prefix for a replica row that could not be copied into its business
# table. retry_failed_materializations() retries these and resolves them once it works.
MATERIALIZE_FAILED = "materialize_failed"


def _parse_updated_at(payload: dict) -> timezone.datetime:
    raw = payload.get("updated_at") or payload.get("created_at")
    if raw:
        dt = parse_datetime(str(raw)) if not hasattr(raw, "isoformat") else raw
        if dt is not None:
            if timezone.is_naive(dt):
                return timezone.make_aware(dt, timezone.get_current_timezone())
            return dt
    return timezone.now()


def _materialize(loc: str, entity_type: str, entity_id: UUID, version: int, payload: dict, tombstone: bool) -> None:
    """Copy into the business table. A failure here must not lose the replica row."""
    from city_sync.materialize import materialize_event

    try:
        with transaction.atomic():
            materialize_event(
                loc=loc,
                entity_type=entity_type,
                entity_id=entity_id,
                version=version,
                payload=payload,
                tombstone=tombstone,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("materialize failed %s/%s/%s", loc, entity_type, entity_id)
        # Open conflict = retryable state; see retry_failed_materializations().
        HqConflict.objects.create(
            location_code=loc,
            entity_type=entity_type,
            entity_id=entity_id,
            incoming_version=version,
            stored_version=version,
            reason=f"{MATERIALIZE_FAILED}:{type(exc).__name__}"[:64],
            payload=payload,
        )


def apply_event(event: dict[str, Any], *, credential_location: str) -> dict[str, Any]:
    """
    Apply one Outbox event inside its own transaction.
    Returns {"event_id", "result", "detail"?}.
    """
    if not isinstance(event, dict):
        return {"event_id": "", "result": RESULT_REJECTED, "detail": "event_not_object"}
    event_id = str(event.get("event_id") or "")
    loc = str(event.get("location_code") or "").strip().upper()
    entity_type = str(event.get("entity_type") or "").strip()
    entity_id_raw = event.get("entity_id")
    operation = str(event.get("operation") or "").strip()
    try:
        entity_version = int(event.get("entity_version") or 0)
        schema_ver = int(event.get("schema_version") or 1)
        city_seq = int(event.get("seq") or 0)
    except (TypeError, ValueError):
        return {"event_id": event_id, "result": RESULT_REJECTED, "detail": "invalid_numeric_fields"}

    if not event_id or not loc or not entity_type or not entity_id_raw:
        return {"event_id": event_id, "result": RESULT_REJECTED, "detail": "missing_required_fields"}
    if loc != credential_location.upper():
        return {"event_id": event_id, "result": RESULT_REJECTED, "detail": "location_mismatch"}
    if operation not in {"upsert", "tombstone"}:
        return {"event_id": event_id, "result": RESULT_REJECTED, "detail": "bad_operation"}

    max_schema = int(getattr(settings, "SYNC_SCHEMA_VERSION", 1))
    if schema_ver > max_schema:
        # City must retry without counting toward dead.
        return {
            "event_id": event_id,
            "result": RESULT_RETRY,
            "detail": "schema_unsupported",
            "schema_hold": True,
        }

    try:
        entity_id = entity_id_raw if isinstance(entity_id_raw, UUID) else UUID(str(entity_id_raw))
        event_uuid = UUID(str(event_id))
    except (TypeError, ValueError):
        return {"event_id": event_id, "result": RESULT_REJECTED, "detail": "bad_uuid"}

    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    ph = str(event.get("payload_hash") or "") or payload_hash(payload)

    for attempt in range(2):
        try:
            # Own transaction / savepoint per event: one failure never poisons the batch.
            with transaction.atomic():
                return _apply_checked(
                    event_id=event_id,
                    event_uuid=event_uuid,
                    loc=loc,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    operation=operation,
                    entity_version=entity_version,
                    city_seq=city_seq,
                    payload=payload,
                    ph=ph,
                )
        except IntegrityError:
            # The same event delivered twice at once, or two first inserts of one entity racing.
            if HqEventLog.objects.filter(location_code=loc, event_id=event_uuid).exists():
                return {"event_id": event_id, "result": RESULT_DUPLICATE, "detail": "event_already_applied"}
            if attempt:
                logger.exception("apply_event integrity error %s/%s", loc, event_id)
    return {"event_id": event_id, "result": RESULT_ERROR, "detail": "integrity_error"}


def _apply_checked(
    *,
    event_id: str,
    event_uuid: UUID,
    loc: str,
    entity_type: str,
    entity_id: UUID,
    operation: str,
    entity_version: int,
    city_seq: int,
    payload: dict,
    ph: str,
) -> dict[str, Any]:
    if HqEventLog.objects.filter(location_code=loc, event_id=event_uuid).exists():
        return {"event_id": event_id, "result": RESULT_DUPLICATE, "detail": "event_already_applied"}

    stored = (
        HqReplicaRow.objects.select_for_update()
        .filter(location_code=loc, entity_type=entity_type, entity_id=entity_id)
        .first()
    )

    result = RESULT_APPLIED
    detail = ""

    if stored is None:
        HqReplicaRow.objects.create(
            location_code=loc,
            entity_type=entity_type,
            entity_id=entity_id,
            entity_version=entity_version,
            is_deleted=(operation == "tombstone"),
            payload=payload,
            payload_hash=ph,
            updated_at=_parse_updated_at(payload),
        )
    elif entity_version > stored.entity_version:
        # Every event carries the whole row (outbox.model_to_payload), so a newer version is
        # safe to apply even when versions in between never arrived (a dead-lettered event, a
        # version skipped by a failed save). Legal entities get an audit entry for the gap.
        if entity_type in LEGAL_ENTITY_TYPES and entity_version != stored.entity_version + 1:
            from city_sync.audit import audit

            audit(
                "version_gap_applied",
                detail={
                    "entity_type": entity_type,
                    "entity_id": str(entity_id),
                    "stored_version": stored.entity_version,
                    "incoming_version": entity_version,
                },
                loc=loc,
            )
            detail = "version_gap_closed"
        stored.entity_version = entity_version
        stored.is_deleted = operation == "tombstone"
        stored.payload = payload
        stored.payload_hash = ph
        stored.updated_at = _parse_updated_at(payload)
        stored.save(
            update_fields=[
                "entity_version",
                "is_deleted",
                "payload",
                "payload_hash",
                "updated_at",
                "received_at",
            ]
        )
    elif entity_version == stored.entity_version:
        if ph and stored.payload_hash and ph != stored.payload_hash:
            HqConflict.objects.create(
                location_code=loc,
                entity_type=entity_type,
                entity_id=entity_id,
                incoming_version=entity_version,
                stored_version=stored.entity_version,
                reason="equal_version_payload_mismatch",
                payload=payload,
            )
            result = RESULT_CONFLICT
            detail = "equal_version_payload_mismatch"
        else:
            result = RESULT_DUPLICATE
            detail = "same_version"
    else:
        # Incoming older than stored — do not overwrite.
        HqConflict.objects.create(
            location_code=loc,
            entity_type=entity_type,
            entity_id=entity_id,
            incoming_version=entity_version,
            stored_version=stored.entity_version,
            reason="stale",
            payload=payload,
        )
        result = RESULT_STALE
        detail = "older_version"

    if result == RESULT_APPLIED:
        _materialize(loc, entity_type, entity_id, entity_version, payload, operation == "tombstone")

    # Always record receipt (including stale/conflict) so retries are idempotent.
    HqEventLog.objects.create(
        location_code=loc,
        city_seq=city_seq,
        event_id=event_uuid,
        entity_type=entity_type,
        entity_id=entity_id,
        entity_version=entity_version,
        result=result,
        payload_hash=ph,
    )

    # last_seq_applied is the HIGHEST city seq received, not a contiguous watermark: the city
    # holds back later versions of an entity and retries failures, so seqs arrive out of
    # order, and outbox seqs have gaps (rolled-back inserts). Use it for monitoring only.
    cursor, _ = HqSyncCursor.objects.select_for_update().get_or_create(
        location_code=loc,
        defaults={"last_seq_applied": 0},
    )
    if city_seq > cursor.last_seq_applied:
        cursor.last_seq_applied = city_seq
    cursor.last_success_at = timezone.now()
    cursor.last_event_at = timezone.now()
    cursor.save(update_fields=["last_seq_applied", "last_success_at", "last_event_at"])

    return {"event_id": event_id, "result": result, "detail": detail}


def apply_event_batch(events: list[dict], *, credential_location: str) -> list[dict]:
    """Process events in ascending city seq order (required for correct versions)."""

    def _seq_key(ev: dict) -> int:
        try:
            return int(ev.get("seq") or 0) if isinstance(ev, dict) else 0
        except (TypeError, ValueError):
            return 0

    ordered = sorted(events, key=_seq_key)
    return [apply_event(ev, credential_location=credential_location) for ev in ordered]


def retry_failed_materializations(*, loc: str = "", limit: int = 200) -> dict[str, int]:
    """
    HQ: copy replica rows whose materialization failed into the business tables again,
    from the latest replica (not the failed payload). A success resolves every open
    materialize_failed conflict of that entity; a repeat failure leaves them open.
    """
    from city_sync.materialize import materialize_event

    qs = HqConflict.objects.filter(reason__startswith=MATERIALIZE_FAILED, resolved_at__isnull=True)
    if loc:
        qs = qs.filter(location_code=loc.upper())
    stats = {"retried": 0, "fixed": 0, "failed": 0}
    seen: set[tuple] = set()
    for key in qs.order_by("detected_at").values_list("location_code", "entity_type", "entity_id").iterator():
        if key in seen:
            continue
        if len(seen) >= limit:
            break
        seen.add(key)
        row_loc, row_type, row_eid = key
        replica = HqReplicaRow.objects.filter(
            location_code=row_loc, entity_type=row_type, entity_id=row_eid
        ).first()
        stats["retried"] += 1
        if replica is not None:
            try:
                with transaction.atomic():
                    materialize_event(
                        loc=row_loc,
                        entity_type=row_type,
                        entity_id=replica.entity_id,
                        version=replica.entity_version,
                        payload=replica.payload,
                        tombstone=replica.is_deleted,
                    )
            except Exception:  # noqa: BLE001
                stats["failed"] += 1
                logger.warning(
                    "materialize retry still failing %s/%s/%s", row_loc, row_type, row_eid, exc_info=True
                )
                continue
        HqConflict.objects.filter(
            location_code=row_loc,
            entity_type=row_type,
            entity_id=row_eid,
            reason__startswith=MATERIALIZE_FAILED,
            resolved_at__isnull=True,
        ).update(
            resolved_by="auto_retry" if replica is not None else "auto_retry:no_replica",
            resolved_at=timezone.now(),
        )
        stats["fixed"] += 1
    return stats
