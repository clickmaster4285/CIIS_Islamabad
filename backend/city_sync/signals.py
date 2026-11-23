"""Generic hooks: every SYNCED_ENTITY_TYPES model → Outbox (+ media)."""

from __future__ import annotations

import logging

from django.apps import apps
from django.conf import settings
from django.db import router, transaction
from django.db.models import F, Value
from django.db.models.functions import Greatest
from django.db.models.signals import post_delete, post_save, pre_delete, pre_save

from city_sync.identity import location_code
from city_sync.outbox import enqueue_from_instance
from city_sync.syncable import (
    EVIDENCE_ENTITY_TYPES,
    SYNC_MEDIA_FIELDS,
    SYNCED_ENTITY_TYPES,
    entity_type_for_model,
    media_field_path,
    media_url_to_path,
    new_entity_id,
)

logger = logging.getLogger(__name__)

_SYNC_SKIP_ATTR = "_city_sync_skip_outbox"
_CONNECTED = False


def _pre_save(sender, instance, **kwargs):
    entity_type = entity_type_for_model(sender)
    if not entity_type:
        return
    if getattr(instance, _SYNC_SKIP_ATTR, False):
        return

    if not getattr(instance, "location_code", None):
        from city_sync.downlink import default_location_for

        instance.location_code = default_location_for(instance) or location_code()
    if not getattr(instance, "entity_id", None):
        # Reuse UUID primary keys when present
        pk = getattr(instance, "pk", None)
        if pk is not None and hasattr(pk, "hex"):
            instance.entity_id = pk
        else:
            instance.entity_id = new_entity_id()

    _reserve_version(sender, instance, entity_type)


def _media_name(value) -> str:
    """Same normalisation as syncable.media_field_path, for a raw DB value."""
    if not value:
        return ""
    if isinstance(value, str):
        return media_url_to_path(value)
    return getattr(value, "name", "") or ""


def _reserve_version(sender, instance, entity_type: str) -> None:
    """
    Give this save the next version, atomically.

    The old read-then-write let two concurrent saves both pick prev+1 (two different
    payloads with one version -> conflict at HQ). Here one UPDATE bumps the stored version
    and the new value is read back inside the same (savepoint) transaction, so the row
    lock serialises concurrent writers and each save gets a distinct version. The stored
    version is also correct when the caller's save(update_fields=...) omits "version".
    Also remembers the stored media file names so post_save only re-uploads changed files.
    """
    current = int(getattr(instance, "version", 1) or 1)
    instance._city_sync_old_media = None
    pk = getattr(instance, "pk", None)
    if pk is None:
        instance.version = current if current >= 1 else 1
        return
    media_fields = [
        f for f in (SYNC_MEDIA_FIELDS.get(entity_type) or []) if _has_concrete_field(sender, f)
    ]
    try:
        with transaction.atomic(using=router.db_for_write(sender, instance=instance)):
            manager = sender._base_manager
            bumped = manager.filter(pk=pk).update(
                version=Greatest(F("version") + 1, Value(current))
            )
            if not bumped:
                instance.version = current if current >= 1 else 1
                return
            row = manager.filter(pk=pk).values("version", *media_fields).first()
    except Exception:  # noqa: BLE001 - never block the business write on the sync bump
        logger.exception("Could not reserve a sync version for %s pk=%s", entity_type, pk)
        return
    if row is None:
        return
    instance.version = int(row.pop("version") or current)
    instance._city_sync_old_media = {k: _media_name(v) for k, v in row.items()}


def _has_concrete_field(model, name: str) -> bool:
    try:
        return model._meta.get_field(name).concrete
    except Exception:  # noqa: BLE001
        return False


def _post_save(sender, instance, created, **kwargs):
    entity_type = entity_type_for_model(sender)
    if not entity_type:
        return
    if getattr(instance, _SYNC_SKIP_ATTR, False):
        return
    if not getattr(settings, "SYNC_OUTBOX_ENABLED", True):
        return

    enqueue_from_instance(instance, entity_type=entity_type, created=created)

    if not getattr(settings, "SYNC_MEDIA_ENABLED", True):
        return
    fields = SYNC_MEDIA_FIELDS.get(entity_type) or []
    if not fields:
        return
    try:
        from city_sync.media_sync import enqueue_media_file
        from city_sync.models import MediaQueue

        priority = (
            MediaQueue.PRIORITY_EVIDENCE
            if entity_type in EVIDENCE_ENTITY_TYPES
            else MediaQueue.PRIORITY_ROUTINE
        )
        old_media = getattr(instance, "_city_sync_old_media", None) or {}
        for fname in fields:
            name = media_field_path(instance, fname)
            if not name:
                continue
            # Only a new or replaced file is uploaded; saving other fields does not re-hash it.
            if not created and fname in old_media and old_media[fname] == name:
                continue
            enqueue_media_file(
                absolute_or_media_path=name,
                record_entity_type=entity_type,
                record_entity_id=instance.entity_id,
                priority=priority,
                wait_outbox_done=True,
            )
    except Exception:  # noqa: BLE001
        logger.exception("Media enqueue failed for %s pk=%s", entity_type, instance.pk)


def _forbid_hard_delete(sender, instance, **kwargs):
    entity_type = entity_type_for_model(sender)
    if not entity_type:
        return
    if not getattr(settings, "SYNC_FORBID_HARD_DELETE", True):
        return
    raise RuntimeError(
        f"Hard delete forbidden for synced entity '{entity_type}'. "
        "Set is_deleted=True and save() so a tombstone is synced to HQ."
    )


def _post_delete(sender, instance, **kwargs):
    """A normal delete at a city still reaches HQ as a tombstone; HQ keeps its copy marked deleted."""
    entity_type = entity_type_for_model(sender)
    if not entity_type or getattr(instance, _SYNC_SKIP_ATTR, False):
        return
    if not getattr(settings, "SYNC_OUTBOX_ENABLED", True) or not getattr(instance, "entity_id", None):
        return
    from city_sync.identity import is_city
    from city_sync.outbox import enqueue_outbox, model_to_payload

    if not is_city():
        return
    instance.is_deleted = True
    instance.version = int(getattr(instance, "version", 1) or 1) + 1
    # Runs inside the delete's transaction, so the tombstone and the delete commit together.
    enqueue_outbox(
        entity_type=entity_type,
        entity_id=instance.entity_id,
        entity_version=instance.version,
        operation="tombstone",
        payload=model_to_payload(instance),
        location=getattr(instance, "location_code", None) or location_code(),
    )


def connect_signals() -> None:
    global _CONNECTED
    if _CONNECTED:
        return
    for entity_type, label in SYNCED_ENTITY_TYPES.items():
        try:
            model = apps.get_model(label)
        except LookupError:
            logger.warning("Synced model not found: %s (%s)", entity_type, label)
            continue
        pre_save.connect(
            _pre_save,
            sender=model,
            dispatch_uid=f"city_sync.pre_save.{entity_type}",
        )
        post_save.connect(
            _post_save,
            sender=model,
            dispatch_uid=f"city_sync.post_save.{entity_type}",
        )
        pre_delete.connect(
            _forbid_hard_delete,
            sender=model,
            dispatch_uid=f"city_sync.pre_delete.{entity_type}",
        )
        post_delete.connect(
            _post_delete,
            sender=model,
            dispatch_uid=f"city_sync.post_delete.{entity_type}",
        )
    _connect_downlink()
    _CONNECTED = True
    logger.info("city_sync signals connected for %s entities", len(SYNCED_ENTITY_TYPES))


def _connect_downlink() -> None:
    """HQ -> city HR changes (Stage 4)."""
    from city_sync.downlink import on_hq_post_save, on_hq_pre_save

    for label in ("users.Staff", "users.Attendance", "users.LeaveRequest", "recognition.FaceEnrollment"):
        try:
            model = apps.get_model(label)
        except LookupError:
            continue
        pre_save.connect(on_hq_pre_save, sender=model, dispatch_uid=f"city_sync.downlink.pre_save.{label}")
        post_save.connect(on_hq_post_save, sender=model, dispatch_uid=f"city_sync.downlink.post_save.{label}")


try:
    connect_signals()
except Exception as _exc:  # noqa: BLE001
    from django.core.exceptions import AppRegistryNotReady

    if isinstance(_exc, AppRegistryNotReady):
        # Imported before the app registry is ready; CitySyncConfig.ready() connects them.
        logger.debug("city_sync signals deferred", exc_info=True)
    else:
        logger.error("city_sync signals could not be connected at import", exc_info=True)
