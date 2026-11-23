"""Retention pruning for high-volume infra tables (events, NVR logs, server logs).

Days come from settings (0 or less disables a table):
  INFRA_EVENT_RETENTION_DAYS       (default 30)
  INFRA_NVR_LOG_RETENTION_DAYS     (default 30)
  INFRA_SERVER_LOG_RETENTION_DAYS  (default 30)
  INFRA_RETENTION_INTERVAL_SEC     (default 3600, how often the worker prunes)
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta

from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

BATCH_SIZE = 2000
MAX_BATCHES_PER_RUN = 50  # cap one run at ~100k rows per table; the rest goes next hour

_last_run = 0.0
_run_lock = threading.Lock()


def prune_queryset(qs, *, batch_size: int = BATCH_SIZE, max_batches: int = MAX_BATCHES_PER_RUN) -> int:
    """Delete rows of ``qs`` in primary-key batches (short transactions, no table lock)."""
    deleted = 0
    model = qs.model
    for _ in range(max_batches):
        ids = list(qs.order_by("pk").values_list("pk", flat=True)[:batch_size])
        if not ids:
            break
        model.objects.filter(pk__in=ids).delete()
        deleted += len(ids)
        if len(ids) < batch_size:
            break
    return deleted


def _days(name: str, default: int) -> int:
    try:
        return int(getattr(settings, name, default))
    except (TypeError, ValueError):
        return default


def prune_infra_tables() -> dict[str, int]:
    from .models import InfraEvent, InfraNvrLog, InfraServerLog

    now = timezone.now()
    out: dict[str, int] = {}
    plan = (
        ("events", InfraEvent, "created_at", _days("INFRA_EVENT_RETENTION_DAYS", 30)),
        ("nvr_logs", InfraNvrLog, "log_time", _days("INFRA_NVR_LOG_RETENTION_DAYS", 30)),
        ("server_logs", InfraServerLog, "log_time", _days("INFRA_SERVER_LOG_RETENTION_DAYS", 30)),
    )
    for label, model, field, days in plan:
        if days <= 0:
            continue
        cutoff = now - timedelta(days=days)
        try:
            out[label] = prune_queryset(model.objects.filter(**{f"{field}__lt": cutoff}))
        except Exception:
            logger.exception("[infra-monitor] retention prune failed for %s", label)
    return out


def maybe_prune_infra_tables() -> dict[str, int] | None:
    """Prune at most once per INFRA_RETENTION_INTERVAL_SEC (default hourly)."""
    global _last_run
    interval = max(60.0, float(getattr(settings, "INFRA_RETENTION_INTERVAL_SEC", 3600)))
    if _last_run and time.monotonic() - _last_run < interval:
        return None
    if not _run_lock.acquire(blocking=False):
        return None
    try:
        _last_run = time.monotonic()
        stats = prune_infra_tables()
        if any(stats.values()):
            logger.info("[infra-monitor] retention pruned %s", stats)
        return stats
    finally:
        _run_lock.release()
