"""Phase 10 — city↔HQ reconciliation summaries for all synced entity types."""

from __future__ import annotations

import hashlib
import logging
from datetime import date, timedelta

from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime

from city_sync.audit import audit
from city_sync.identity import location_code
from city_sync.models import HqCityStatus, HqReplicaRow, ReconcileReport
from city_sync.syncable import SYNCED_ENTITY_TYPES, resolve_model

logger = logging.getLogger(__name__)


def _hash_pairs(pairs: list[tuple[str, int]]) -> str:
    raw = "|".join(f"{eid}:{ver}" for eid, ver in pairs)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _day_field(model) -> str | None:
    """The field a "day" filter uses for this model: date, else created_at, else none."""
    names = {f.name for f in model._meta.concrete_fields}
    if "date" in names:
        return "date"
    if "created_at" in names:
        return "created_at"
    return None


def city_entity_summary(
    entity_type: str, day: date | None = None, loc: str | None = None
) -> tuple[int, str, list[str]]:
    """(count, hash, sorted entity ids) of live rows of one type, for one location and day."""
    label = SYNCED_ENTITY_TYPES.get(entity_type)
    if not label:
        return 0, "", []
    try:
        model = resolve_model(label)
    except LookupError:
        return 0, "", []

    names = {f.name for f in model._meta.concrete_fields}
    qs = model._base_manager.all()
    if "is_deleted" in names:
        qs = qs.filter(is_deleted=False)
    qs = qs.exclude(entity_id__isnull=True)
    # At HQ the business table also holds HQ's own rows and every other city's copies.
    if loc and "location_code" in names:
        codes = [loc.upper()]
        if loc.upper() == location_code():
            codes.append("")  # this server's own untagged (legacy) rows
        qs = qs.filter(location_code__in=codes)

    day_field = _day_field(model)
    if day is not None and day_field == "date":
        qs = qs.filter(date=day)
    elif day is not None and day_field == "created_at":
        qs = qs.filter(created_at__date=day)

    pairs = [
        (str(eid), int(ver or 1))
        for eid, ver in qs.order_by("entity_id").values_list("entity_id", "version").iterator(chunk_size=2000)
    ]
    pairs.sort(key=lambda x: x[0])
    return len(pairs), _hash_pairs(pairs), [p[0] for p in pairs]


def _payload_day(payload: dict, day_field: str | None) -> date | None:
    raw = (payload or {}).get(day_field) if day_field else None
    if not raw:
        return None
    if day_field == "date":
        return parse_date(str(raw)[:10])
    dt = parse_datetime(str(raw))
    if dt is None:
        return parse_date(str(raw)[:10])
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt, timezone.get_current_timezone())
    return timezone.localtime(dt).date()


def hq_entity_summary(loc: str, entity_type: str, day: date | None = None) -> tuple[int, str, list[str]]:
    """Same summary from HQ's replica rows of one city (streamed; payloads only read for a day filter)."""
    qs = HqReplicaRow.objects.filter(
        location_code=loc.upper(),
        entity_type=entity_type,
        is_deleted=False,
    )
    pairs: list[tuple[str, int]] = []
    if day is None:
        for eid, ver in qs.values_list("entity_id", "entity_version").iterator(chunk_size=2000):
            pairs.append((str(eid), int(ver)))
    else:
        label = SYNCED_ENTITY_TYPES.get(entity_type)
        try:
            day_field = _day_field(resolve_model(label)) if label else None
        except LookupError:
            day_field = None
        for row in qs.only("entity_id", "entity_version", "payload").iterator(chunk_size=500):
            if day_field is not None and _payload_day(row.payload, day_field) != day:
                continue
            pairs.append((str(row.entity_id), int(row.entity_version)))
    pairs.sort(key=lambda x: x[0])
    return len(pairs), _hash_pairs(pairs), [p[0] for p in pairs]


def run_reconcile_for_location(loc: str, day: date | None = None) -> list[ReconcileReport]:
    day = day or (timezone.localdate() - timedelta(days=1))
    reports: list[ReconcileReport] = []
    total_mismatch = 0
    all_ok = True

    for entity_type in sorted(SYNCED_ENTITY_TYPES.keys()):
        # One type at a time; nothing of the previous type stays in memory.
        city_count, city_hash, city_ids = city_entity_summary(entity_type, day=day, loc=loc)
        hq_count, hq_hash, hq_ids = hq_entity_summary(loc, entity_type, day=day)
        # Skip empty-empty (entity never used)
        if city_count == 0 and hq_count == 0:
            continue
        ok = city_count == hq_count and city_hash == hq_hash
        city_set, hq_set = set(city_ids), set(hq_ids)
        del city_ids, hq_ids
        missing = sorted(city_set - hq_set)
        extra = sorted(hq_set - city_set)
        del city_set, hq_set
        total_mismatch += len(missing) + len(extra)
        all_ok = all_ok and ok
        reports.append(
            ReconcileReport.objects.create(
                location_code=loc.upper(),
                day=day,
                entity_type=entity_type,
                city_count=city_count,
                hq_count=hq_count,
                city_hash=city_hash,
                hq_hash=hq_hash,
                ok=ok,
                mismatched_ids={"missing_at_hq": missing[:200], "extra_at_hq": extra[:200]},
            )
        )

    status, _ = HqCityStatus.objects.get_or_create(location_code=loc.upper())
    status.last_reconcile_at = timezone.now()
    status.last_reconcile_ok = all_ok
    status.reconcile_mismatch_count = total_mismatch
    status.save(update_fields=["last_reconcile_at", "last_reconcile_ok", "reconcile_mismatch_count"])
    audit(
        "reconcile_finished",
        detail={"location": loc, "day": day.isoformat(), "ok": all_ok, "mismatch": total_mismatch},
        loc=loc,
    )
    return reports


def run_reconcile_all(day: date | None = None) -> list[ReconcileReport]:
    locs = set(HqReplicaRow.objects.values_list("location_code", flat=True))
    locs |= set(HqCityStatus.objects.values_list("location_code", flat=True))
    if not locs:
        locs = {location_code()} if location_code() else set()
    out: list[ReconcileReport] = []
    for loc in sorted(locs):
        out.extend(run_reconcile_for_location(loc, day=day))
    return out
