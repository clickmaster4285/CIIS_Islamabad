"""Live health numbers for the City Control Grid ("Quick Status" / Infrastructure).

``local_vitals()`` describes THIS server: uptime, disk, AI (ML) service, the
infrastructure devices it monitors and its newest detection. A city sends it to
HQ inside the sync heartbeat (``city_sync.control.heartbeat_extras``); HQ calls
it directly for its own row. ``row_health()`` turns it into the values the
city detail page shows. Nothing here is guessed: a value a server does not
report is returned as None and shown as "Not reported".
"""

from __future__ import annotations

import logging
import shutil
import time
from datetime import timedelta
from typing import Any

from django.conf import settings
from django.db.models import Count, Q
from django.utils import timezone

logger = logging.getLogger(__name__)

# A camera counts as live from detections if one arrived in this window.
RECENT_DETECTION = timedelta(minutes=15)


def _disk_percent() -> float | None:
    try:
        usage = shutil.disk_usage(str(getattr(settings, "MEDIA_ROOT", ".") or "."))
        return round(usage.used / usage.total * 100.0, 1) if usage.total else None
    except Exception:  # noqa: BLE001
        return None


def _uptime_sec() -> int | None:
    try:
        import psutil

        return max(0, int(time.time() - psutil.boot_time()))
    except Exception:  # noqa: BLE001
        return None


def _infra_summary() -> dict[str, Any] | None:
    """Counts of the infrastructure devices this server monitors (None if module unavailable)."""
    try:
        from infrastructure_monitoring.models import DeviceStatus, DeviceType, InfraDevice
    except Exception:  # noqa: BLE001
        return None
    devices = InfraDevice.objects.filter(is_active=True)
    by_status = dict(devices.values_list("status").annotate(n=Count("pk")))
    total = sum(by_status.values())
    online = by_status.get(DeviceStatus.ONLINE, 0)
    maintenance = by_status.get(DeviceStatus.MAINTENANCE, 0)
    nvrs = devices.filter(device_type=DeviceType.NVR)
    by_type = {
        row["device_type"]: {"total": row["total"], "online": row["online"]}
        for row in devices.values("device_type").annotate(
            total=Count("pk"), online=Count("pk", filter=Q(status=DeviceStatus.ONLINE))
        )
    }
    return {
        "total": total,
        "online": online,
        "issues": max(0, total - online - maintenance),
        "nvrs_total": nvrs.count(),
        "nvrs_online": nvrs.filter(status=DeviceStatus.ONLINE).count(),
        "by_type": by_type,
    }


def _last_detection_iso() -> str | None:
    try:
        from cameras.models import DetectionEvent
        from city_sync.identity import local_rows_q

        at = (
            DetectionEvent.objects.filter(local_rows_q())
            .order_by("-created_at")
            .values_list("created_at", flat=True)
            .first()
        )
        return at.isoformat() if at else None
    except Exception:  # noqa: BLE001
        return None


def local_vitals() -> dict[str, Any]:
    """This server's live numbers (sent to HQ in the heartbeat as ``vitals``)."""
    try:
        from cameras.live_status import camera_counts

        cams = camera_counts()
    except Exception:  # noqa: BLE001
        logger.exception("Could not count live cameras for vitals")
        cams = {}
    try:
        infra = _infra_summary()
    except Exception:  # noqa: BLE001
        logger.exception("Could not summarise infrastructure devices for vitals")
        infra = None
    return {
        "uptime_sec": _uptime_sec(),
        "disk_percent": _disk_percent(),
        "ml_source": cams.get("cameras_online_source"),  # "ml" = ML server answered
        "infra": infra,
        "last_detection_at": _last_detection_iso(),
        "reported_at": timezone.now().isoformat(),
    }


# ----------------------------------------------------------------- formatting


def format_uptime(sec: int | None) -> str | None:
    if sec is None:
        return None
    days, rem = divmod(int(sec), 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _parse_iso(raw):
    if not raw:
        return None
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(str(raw))
        return dt if timezone.is_aware(dt) else timezone.make_aware(dt)
    except (TypeError, ValueError):
        return None


def _ago(dt, now) -> str | None:
    if not dt:
        return None
    sec = max(0, int((now - dt).total_seconds()))
    if sec < 60:
        return f"{max(1, sec)} sec ago"
    if sec < 3600:
        return f"{sec // 60} min ago"
    if sec < 48 * 3600:
        return f"{sec // 3600} hr ago"
    return f"{sec // 86400} d ago"


def row_health(*, is_hq_row: bool, state: str, raw: dict, heartbeat_age: int | None, now) -> dict[str, Any]:
    """Quick Status values for one board row.

    HQ uses ``local_vitals()``; a city uses what its last heartbeat reported.
    """
    offline = state in ("OFFLINE", "LONG_OFFLINE", "UNKNOWN")
    vitals = local_vitals() if is_hq_row else (raw.get("vitals") if isinstance(raw.get("vitals"), dict) else {})

    # API / sync connection
    if is_hq_row:
        api, api_detail = "Connected", "This server (HQ)"
    elif state == "UNKNOWN":
        api, api_detail = "Disconnected", "No heartbeat received yet"
    elif offline:
        api, api_detail = "Disconnected", f"Last heartbeat {_ago(now - timedelta(seconds=heartbeat_age or 0), now)}"
    elif state in ("DEGRADED", "PAUSED"):
        api, api_detail = "Intermittent", "Connected with sync problems" if state == "DEGRADED" else "Sync paused"
    else:
        api, api_detail = "Connected", f"Heartbeat {_ago(now - timedelta(seconds=heartbeat_age or 0), now)}"

    # AI (ML) service
    ml_source = vitals.get("ml_source") or raw.get("cameras_online_source")
    last_det = _parse_iso(vitals.get("last_detection_at"))
    if offline and not is_hq_row:
        ai, ai_detail = "Down", "City not connected"
    elif ml_source == "ml":
        ai, ai_detail = "Healthy", "ML server answering"
    elif ml_source == "detections":
        if last_det and now - last_det <= RECENT_DETECTION:
            ai, ai_detail = "Degraded", "ML server not answering; detections still arriving"
        else:
            ai, ai_detail = "Down", "ML server not answering"
    else:
        ai, ai_detail = "Unknown", "Not reported"

    # Uptime, disk, sync queue
    uptime = format_uptime(vitals.get("uptime_sec")) if not (offline and not is_hq_row) else None
    disk = vitals.get("disk_percent")
    if disk is None:
        disk = raw.get("disk_percent")
    queue = None
    if not is_hq_row and raw:
        queue = {
            "pending_records": int(raw.get("pending_outbox") or 0),
            "pending_media": int(raw.get("pending_media") or 0),
            "failed": int(raw.get("dead_outbox") or 0) + int(raw.get("failed_media") or 0),
            "oldest_pending_sec": int(raw.get("oldest_pending_age_sec") or 0),
        }

    # Infrastructure
    infra = vitals.get("infra") if isinstance(vitals.get("infra"), dict) else None
    infra_health = None
    if infra:
        if offline and not is_hq_row:
            infra_health = "Critical"
        elif infra["total"] and infra["online"] == 0:
            infra_health = "Critical"
        elif infra["issues"]:
            infra_health = "Degraded"
        else:
            infra_health = "Good"

    return {
        "api_connection": api,
        "api_detail": api_detail,
        "ai_service": ai,
        "ai_detail": ai_detail,
        "uptime": uptime,
        "disk_percent": disk,
        "sync_queue": queue,
        "infra": infra,
        "infra_health": infra_health,
        "last_detection": _ago(last_det, now),
        "vitals_reported": bool(vitals),
    }
