"""Enqueue Outbox rows in the same DB transaction as the business write."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.utils.duration import duration_iso_string

from city_sync.identity import location_code, schema_version
from city_sync.models import SyncOutbox
from city_sync.syncable import PAYLOAD_REFS

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "total_seconds"):
        try:
            return duration_iso_string(obj)
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def payload_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def model_to_payload(instance) -> dict:
    """Serialize concrete field values (no binary file contents)."""
    data: dict[str, Any] = {}
    for field in instance._meta.concrete_fields:
        name = field.name
        if name in {"password"}:
            continue
        value = getattr(instance, field.attname if hasattr(field, "attname") else name)
        # File/Image fields → store path only
        if field.get_internal_type() in {"FileField", "ImageField"}:
            data[name] = value.name if value else None
            continue
        data[name] = value
    label = f"{instance._meta.app_label}.{instance._meta.object_name}"
    for name in PAYLOAD_REFS.get(label, []):
        target = getattr(instance, name, None)
        if target is not None:
            data[f"_ref_{name}"] = {"code": getattr(target, "code", ""), "name": getattr(target, "name", "")}
    data["_model"] = label
    data["_pk"] = instance.pk
    return json.loads(json.dumps(data, default=_json_default))


def enqueue_outbox(
    *,
    entity_type: str,
    entity_id: UUID | str,
    entity_version: int,
    operation: str,
    payload: dict,
    location: str | None = None,
) -> SyncOutbox:
    """
    Insert an Outbox row. Must be called inside the same transaction.atomic()
    as the business write (or from post_save during that write).
    """
    loc = (location or location_code()).strip().upper()
    if not loc:
        raise RuntimeError("Cannot enqueue Outbox without LOCATION_CODE")
    if operation not in {SyncOutbox.OP_UPSERT, SyncOutbox.OP_TOMBSTONE}:
        raise ValueError(f"Invalid operation {operation}")

    body = dict(payload)
    ph = payload_hash(body)
    row = SyncOutbox.objects.create(
        location_code=loc,
        entity_type=entity_type,
        entity_id=entity_id if isinstance(entity_id, UUID) else UUID(str(entity_id)),
        operation=operation,
        entity_version=int(entity_version),
        schema_version=schema_version(),
        payload=body,
        payload_hash=ph,
        status=SyncOutbox.STATUS_PENDING,
        next_attempt_at=timezone.now(),
    )
    logger.debug(
        "Outbox enqueued seq=%s type=%s id=%s v=%s op=%s",
        row.seq,
        entity_type,
        entity_id,
        entity_version,
        operation,
    )
    return row


def enqueue_from_instance(instance, *, entity_type: str, created: bool = False) -> SyncOutbox | None:
    """Bump is handled by caller/pre_save; this only writes Outbox."""
    if not getattr(settings, "SYNC_OUTBOX_ENABLED", True):
        return None
    # Cities produce Outbox; HQ does not push its own city writes for other cities.
    from city_sync.identity import is_city

    if not is_city():
        return None
    if not getattr(instance, "entity_id", None):
        return None

    operation = (
        SyncOutbox.OP_TOMBSTONE
        if getattr(instance, "is_deleted", False)
        else SyncOutbox.OP_UPSERT
    )
    return enqueue_outbox(
        entity_type=entity_type,
        entity_id=instance.entity_id,
        entity_version=int(getattr(instance, "version", 1) or 1),
        operation=operation,
        payload=model_to_payload(instance),
        location=getattr(instance, "location_code", None) or location_code(),
    )


@transaction.atomic
def soft_delete_instance(instance) -> None:
    """Preferred delete path for synced models — tombstone via save()+signal."""
    instance.is_deleted = True
    instance.save()


# ---------------------------------------------------------------------------
# Writes that bypass save() (bulk_create / queryset.update) on synced models
# ---------------------------------------------------------------------------
#
# post_save never fires for bulk_create() or QuerySet.update(), so those rows silently
# never reach HQ. Call sites in other apps should switch to these helpers:
#
#   sync_bulk_create(Model, objs)        instead of Model.objects.bulk_create(objs)
#   sync_update(Model.objects.filter(...), field=value, ...)
#                                         instead of Model.objects.filter(...).update(...)
#   enqueue_bulk(instances)              after any other write that skipped save()
#
# Each bumps the version (atomically, in the database) and writes the Outbox rows in the
# same transaction as the business write, exactly like a normal save() would.


def _entity_type_or_none(model) -> str | None:
    from city_sync.syncable import entity_type_for_model

    return entity_type_for_model(model)


def _enqueue_media_for(instance, entity_type: str, fields: list[str] | None) -> None:
    from city_sync.syncable import EVIDENCE_ENTITY_TYPES, SYNC_MEDIA_FIELDS, media_field_path

    if not getattr(settings, "SYNC_MEDIA_ENABLED", True):
        return
    wanted = [f for f in (SYNC_MEDIA_FIELDS.get(entity_type) or []) if fields is None or f in fields]
    if not wanted:
        return
    from city_sync.media_sync import enqueue_media_file
    from city_sync.models import MediaQueue

    priority = MediaQueue.PRIORITY_EVIDENCE if entity_type in EVIDENCE_ENTITY_TYPES else MediaQueue.PRIORITY_ROUTINE
    for fname in wanted:
        name = media_field_path(instance, fname)
        if not name:
            continue
        try:
            enqueue_media_file(
                absolute_or_media_path=name,
                record_entity_type=entity_type,
                record_entity_id=instance.entity_id,
                priority=priority,
                wait_outbox_done=True,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Media enqueue failed for %s pk=%s", entity_type, instance.pk)


def _ensure_identity(model, instances: list) -> None:
    """Give rows written without save() an entity_id / location_code (persisted)."""
    from city_sync.syncable import new_entity_id

    for obj in instances:
        updates = {}
        if not getattr(obj, "entity_id", None):
            pk = getattr(obj, "pk", None)
            obj.entity_id = pk if pk is not None and hasattr(pk, "hex") else new_entity_id()
            updates["entity_id"] = obj.entity_id
        if hasattr(obj, "location_code") and not getattr(obj, "location_code", None):
            obj.location_code = location_code()
            updates["location_code"] = obj.location_code
        if updates and obj.pk is not None:
            model._base_manager.filter(pk=obj.pk).update(**updates)


def enqueue_bulk(instances, *, bump_version: bool = True, media_fields: list[str] | None = ()) -> int:
    """
    Sync rows that were written without save() (e.g. by QuerySet.update()).

    Bumps each row's version in the database (version = version + 1), reloads the rows
    and writes one Outbox event per row, all in one transaction. media_fields: () = no
    media, None = every media field of the type, or a list of field names whose files
    changed. Returns the number of Outbox rows written.
    """
    from django.db.models import F

    instances = [obj for obj in instances if obj is not None and getattr(obj, "pk", None) is not None]
    if not instances:
        return 0
    model = type(instances[0])
    entity_type = _entity_type_or_none(model)
    if not entity_type:
        return 0
    pks = [obj.pk for obj in instances]
    written = 0
    with transaction.atomic():
        manager = model._base_manager
        rows = list(manager.select_for_update().filter(pk__in=pks))
        _ensure_identity(model, rows)
        if bump_version:
            manager.filter(pk__in=pks).update(version=F("version") + 1)
            rows = list(manager.filter(pk__in=pks))
        for obj in rows:
            if enqueue_from_instance(obj, entity_type=entity_type) is not None:
                written += 1
            if media_fields != ():
                _enqueue_media_for(obj, entity_type, None if media_fields is None else list(media_fields))
    return written


def sync_bulk_create(model, objs, *, batch_size: int | None = None) -> list:
    """
    bulk_create() for synced models that still sends the new rows to HQ.

    Sets entity_id / location_code / version=1 like the pre_save hook, inserts, and writes
    the Outbox rows (and queues media files) in the same transaction. Not for
    ignore_conflicts / update_conflicts (rows without a pk cannot be synced).
    """
    from city_sync.syncable import new_entity_id

    objs = list(objs)
    entity_type = _entity_type_or_none(model)
    if not entity_type:
        return model._default_manager.bulk_create(objs, batch_size=batch_size)
    for obj in objs:
        if not getattr(obj, "entity_id", None):
            pk = getattr(obj, "pk", None)
            obj.entity_id = pk if pk is not None and hasattr(pk, "hex") else new_entity_id()
        if hasattr(obj, "location_code") and not getattr(obj, "location_code", None):
            obj.location_code = location_code()
        if hasattr(obj, "version") and not getattr(obj, "version", None):
            obj.version = 1
    with transaction.atomic():
        created = model._default_manager.bulk_create(objs, batch_size=batch_size)
        for obj in created:
            if obj.pk is None:
                raise RuntimeError("sync_bulk_create needs a database that returns primary keys")
            enqueue_from_instance(obj, entity_type=entity_type, created=True)
            _enqueue_media_for(obj, entity_type, None)
    return created


def sync_update(queryset, **values) -> int:
    """
    QuerySet.update() for synced models that still sends the change to HQ.

    Locks the matching rows, applies the update with version = version + 1 in the same
    statement and writes one Outbox event per changed row. Returns the row count.
    """
    from django.db.models import F

    from city_sync.syncable import SYNC_MEDIA_FIELDS

    model = queryset.model
    entity_type = _entity_type_or_none(model)
    if not entity_type:
        return queryset.update(**values)
    with transaction.atomic():
        pks = list(queryset.select_for_update().values_list("pk", flat=True))
        if not pks:
            return 0
        manager = model._base_manager
        count = manager.filter(pk__in=pks).update(**values, version=F("version") + 1)
        changed_media = [f for f in (SYNC_MEDIA_FIELDS.get(entity_type) or []) if f in values]
        enqueue_bulk(
            list(manager.filter(pk__in=pks)),
            bump_version=False,
            media_fields=changed_media or (),
        )
    return count
