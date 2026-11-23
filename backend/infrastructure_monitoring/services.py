"""Sync CIIS cameras/NVRs into Infrastructure Monitoring + evaluate alerts."""

from __future__ import annotations

import ipaddress
import logging
from typing import Any

from django.db import transaction
from django.db.models import Count
from django.utils import timezone

from .models import (
    AlertSeverity,
    DeviceStatus,
    DeviceType,
    InfraAlert,
    InfraAlertRule,
    InfraDevice,
    InfraEvent,
    InfraSite,
    ProtocolType,
)

logger = logging.getLogger(__name__)


def _ensure_site_from_camera_site(cam_site) -> InfraSite | None:
    if not cam_site:
        return None
    site, _ = InfraSite.objects.get_or_create(
        code=cam_site.code,
        defaults={
            "name": cam_site.name,
            "location": cam_site.code,
            "description": getattr(cam_site, "description", "") or "",
            "is_active": getattr(cam_site, "is_active", True),
        },
    )
    return site


def _valid_ip(raw) -> str | None:
    """Normalised IP string, or None when the source value is not a valid IPv4/IPv6 address."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


# Fields the Camera Management source owns. Everything else on an auto-synced row
# (is_active, notes, ports, protocols, poll interval, credentials an admin changed)
# belongs to the infra admin and is only set when the row is first created.
_NVR_SOURCE_FIELDS = (
    "site",
    "device_type",
    "name",
    "manufacturer",
    "model_number",
    "ip_address",
    "channel_count",
)
_CAMERA_SOURCE_FIELDS = (
    "site",
    "device_type",
    "name",
    "manufacturer",
    "model_number",
    "ip_address",
    "install_location",
    "asset_tag",
)


def _apply_source_fields(obj: InfraDevice, values: dict[str, Any], fields, source_meta: dict) -> None:
    for field in fields:
        if field in values:
            setattr(obj, field, values[field])
    meta = dict(obj.metadata) if isinstance(obj.metadata, dict) else {}
    meta.update(source_meta)
    if meta.pop("auto_deactivated", False):
        # The source record is back: undo our own deactivation (never an admin's).
        obj.is_active = True
    obj.metadata = meta


def _deactivate_missing(prefix: str, seen_keys: set[str]) -> int:
    """Deactivate auto-synced rows whose source record was deleted or deactivated."""
    gone = 0
    qs = InfraDevice.objects.filter(source_key__startswith=prefix, is_active=True).exclude(
        source_key__in=seen_keys
    )
    for obj in qs:
        meta = dict(obj.metadata) if isinstance(obj.metadata, dict) else {}
        meta["auto_deactivated"] = True
        obj.metadata = meta
        obj.is_active = False
        obj.save(update_fields=["is_active", "metadata", "updated_at"])
        InfraEvent.objects.create(
            device=obj,
            event_type="auto_sync",
            title=f"Deactivated: {obj.name}"[:200],
            message="Source record removed or deactivated in Camera Management",
            actor="system",
        )
        gone += 1
    return gone


def _sync_one_nvr(nvr) -> str:
    key = f"cameras.nvr:{nvr.id}"
    ip = _valid_ip(nvr.ip_address)
    if ip is None:
        logger.warning("infra sync: skipping NVR %s (%s): invalid IP %r", nvr.id, nvr.name, nvr.ip_address)
        return "skipped"
    site = _ensure_site_from_camera_site(nvr.site)
    source = {
        "site": site,
        "device_type": DeviceType.NVR,
        "name": nvr.name,
        "manufacturer": (nvr.brand or "").title(),
        "model_number": nvr.brand or "",
        "ip_address": ip,
        "channel_count": getattr(nvr, "cam_count", None),
    }
    source_meta = {"brand": nvr.brand, "source": "cameras.nvr"}
    src_user = (nvr.username or "").strip()
    obj = InfraDevice.objects.filter(source_key=key).first()
    if obj is None:
        obj = InfraDevice(
            source_key=key,
            port=nvr.port or 554,
            username=src_user,
            primary_protocol=ProtocolType.SNMP,
            secondary_protocol=ProtocolType.RTSP,
            supports_snmp=True,
            supports_rtsp=True,
            is_active=True,
            poll_interval_sec=60,
            notes="Auto-synced from Camera Management",
            onvif_port=80,  # HTTP/ISAPI management port
            metadata={},
        )
        if nvr.password:
            obj.password = nvr.password
        _apply_source_fields(obj, source, _NVR_SOURCE_FIELDS, source_meta)
        obj.save()
        InfraEvent.objects.create(
            device=obj,
            event_type="auto_sync",
            title=f"NVR imported: {nvr.name}"[:200],
            message="Synced from Camera Management",
            actor="system",
        )
        return "created"
    _apply_source_fields(obj, source, _NVR_SOURCE_FIELDS, source_meta)
    # Follow password changes made in Camera Management, but only while the infra row
    # still uses the same login (an admin who set a different login keeps it).
    cur_user = (obj.username or "").strip()
    if not cur_user or cur_user == src_user:
        obj.username = src_user
        if nvr.password:
            obj.password = nvr.password
    obj.save()
    return "updated"


def _sync_one_camera(cam) -> str:
    key = f"cameras.camera:{cam.id}"
    nvr = cam.nvr if cam.nvr_id else None
    ip = None
    if nvr is not None:
        ip = _valid_ip(nvr.ip_address)
        if ip is None:
            logger.warning(
                "infra sync: skipping camera %s (%s): parent NVR has invalid IP %r",
                cam.id,
                cam.name or cam.code,
                nvr.ip_address,
            )
            return "skipped"
    site = _ensure_site_from_camera_site(nvr.site if nvr else None)
    source = {
        "site": site,
        "device_type": DeviceType.CAMERA,
        "name": cam.name or cam.code or f"Camera {cam.id}",
        "manufacturer": (nvr.brand if nvr else "") or "",
        "model_number": getattr(cam, "camera_type", "") or "",
        "ip_address": ip,
        "install_location": cam.zone or cam.location or "",
        "asset_tag": cam.code or "",
    }
    source_meta = {
        "channel": cam.channel,
        "camera_code": cam.code,
        "source": "cameras.camera",
        "nvr_id": cam.nvr_id,
    }
    obj = InfraDevice.objects.filter(source_key=key).first()
    if obj is None:
        # Seed status from camera record until first poll
        cam_status = (cam.status or "").lower()
        seed_status = (
            DeviceStatus.ONLINE
            if cam_status == "online"
            else DeviceStatus.OFFLINE
            if cam_status == "offline"
            else DeviceStatus.UNKNOWN
        )
        # No credentials are copied: camera ISAPI is read through the parent NVR.
        # Saving the camera's own login sets metadata.camera_has_own_creds (serializer).
        obj = InfraDevice(
            source_key=key,
            port=(nvr.port if nvr else None) or 554,
            primary_protocol=ProtocolType.ICMP,
            secondary_protocol=ProtocolType.RTSP,
            supports_rtsp=True,
            supports_onvif=False,
            is_active=True,
            poll_interval_sec=60,
            notes="Auto-synced from Camera Management",
            status=seed_status,
            metadata={"camera_has_own_creds": False},
        )
        _apply_source_fields(obj, source, _CAMERA_SOURCE_FIELDS, source_meta)
        obj.save()
        return "created"
    _apply_source_fields(obj, source, _CAMERA_SOURCE_FIELDS, source_meta)
    obj.save()
    return "updated"


def sync_cameras_and_nvrs() -> dict[str, int]:
    """
    Auto-import active NVRs and Cameras from the cameras app so they appear
    in Infrastructure Monitoring without manual re-entry.

    Only source-owned identity fields are refreshed on existing rows. A bad record is
    logged and skipped; rows whose source disappeared are deactivated.
    """
    stats = {"created": 0, "updated": 0, "skipped": 0, "failed": 0, "deactivated": 0}
    try:
        from cameras.models import Camera, Nvr
    except Exception as exc:
        logger.warning("cameras app unavailable for sync: %s", exc)
        return {**stats, "skipped": 1}

    def _run(kind: str, rows, fn, seen: set[str]) -> bool:
        """True when the whole source list was walked (safe to deactivate missing rows)."""
        try:
            for row in rows:
                seen.add(f"cameras.{kind}:{row.id}")
                try:
                    with transaction.atomic():
                        outcome = fn(row)
                    stats[outcome] += 1
                except Exception:
                    stats["failed"] += 1
                    logger.exception("infra sync: %s %s failed", kind, row.id)
            return True
        except Exception:
            logger.exception("infra sync: listing %s records failed", kind)
            return False

    nvr_seen: set[str] = set()
    nvr_rows = (
        Nvr.objects.select_related("site")
        .filter(is_active=True)
        .annotate(cam_count=Count("cameras"))
    )
    if _run("nvr", nvr_rows, _sync_one_nvr, nvr_seen):
        stats["deactivated"] += _deactivate_missing("cameras.nvr:", nvr_seen)

    cam_seen: set[str] = set()
    cam_rows = Camera.objects.select_related("nvr", "nvr__site").filter(
        is_active=True, nvr__is_active=True
    )
    if _run("camera", cam_rows, _sync_one_camera, cam_seen):
        stats["deactivated"] += _deactivate_missing("cameras.camera:", cam_seen)

    return stats


def _compare(op: str, left: float, right: float) -> bool:
    op = (op or "lt").lower()
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "eq":
        return left == right
    if op == "neq":
        return left != right
    return False


def evaluate_alert_rules(device: InfraDevice) -> int:
    """Create/update open alerts from active rules vs last_metrics / status."""
    raised = 0
    metrics = device.last_metrics or {}
    rules = InfraAlertRule.objects.filter(is_active=True)
    for rule in rules:
        if rule.device_type and rule.device_type != device.device_type:
            continue
        key = (rule.metric_key or "").strip()
        if not key:
            continue

        # Special: offline
        if key in ("offline", "status_offline"):
            triggered = device.status == DeviceStatus.OFFLINE
            metric_value = 1.0 if triggered else 0.0
        elif key in ("fault", "status_fault"):
            triggered = device.status == DeviceStatus.FAULT
            metric_value = 1.0 if triggered else 0.0
        else:
            raw = metrics.get(key)
            if raw is None:
                continue
            try:
                metric_value = float(raw)
            except (TypeError, ValueError):
                continue
            if rule.threshold_value is None:
                continue
            triggered = _compare(rule.operator, metric_value, float(rule.threshold_value))

        alert_type = f"rule:{rule.id}:{key}"
        existing = (
            InfraAlert.objects.filter(
                device=device, alert_type=alert_type, resolved=False
            )
            .order_by("-last_detected")
            .first()
        )
        if triggered:
            title = rule.message_template or f"{device.name}: {key} alert"
            if existing:
                existing.metric_value = metric_value
                existing.message = title
                existing.save(update_fields=["metric_value", "message", "last_detected"])
            else:
                InfraAlert.objects.create(
                    device=device,
                    alert_type=alert_type,
                    severity=rule.severity or AlertSeverity.MEDIUM,
                    title=title[:200],
                    message=title,
                    metric_key=key,
                    metric_value=metric_value,
                )
                raised += 1
                InfraEvent.objects.create(
                    device=device,
                    event_type="alert_raised",
                    title=title[:200],
                    message=f"{key}={metric_value}",
                    actor="system",
                )
        elif existing:
            existing.resolved = True
            existing.resolved_at = timezone.now()
            existing.save(update_fields=["resolved", "resolved_at", "last_detected"])
            InfraEvent.objects.create(
                device=device,
                event_type="alert_cleared",
                title=f"Cleared: {existing.title}",
                actor="system",
            )
    return raised


@transaction.atomic
def apply_probe_result(device: InfraDevice, result: dict[str, Any]) -> InfraDevice:
    old_status = device.status
    device.status = result["status"]
    device.last_metrics = result.get("metrics") or {}
    device.last_error = (result.get("error") or "")[:512]
    device.last_polled_at = timezone.now()
    if result.get("reachable"):
        device.last_seen_at = device.last_polled_at
    device.save(
        update_fields=[
            "status",
            "last_metrics",
            "last_error",
            "last_polled_at",
            "last_seen_at",
            "updated_at",
        ]
    )
    if old_status != device.status:
        InfraEvent.objects.create(
            device=device,
            event_type="status_change",
            title=f"{device.name}: {old_status} → {device.status}",
            message=device.last_error,
            actor="poller",
            payload={"from": old_status, "to": device.status},
        )
    evaluate_alert_rules(device)
    return device


def poll_all_devices(*, limit: int | None = None) -> dict[str, int]:
    from .poller import probe_device

    qs = InfraDevice.objects.filter(is_active=True).order_by("id")
    if limit:
        qs = qs[:limit]
    online = offline = degraded = fault = errors = 0
    for device in qs:
        try:
            result = probe_device(device)
            apply_probe_result(device, result)
            st = result["status"]
            if st == DeviceStatus.ONLINE:
                online += 1
            elif st == DeviceStatus.OFFLINE:
                offline += 1
            elif st == DeviceStatus.DEGRADED:
                degraded += 1
            elif st == DeviceStatus.FAULT:
                fault += 1
        except Exception as exc:
            errors += 1
            logger.exception("probe failed for device %s: %s", device.id, exc)
            device.status = DeviceStatus.UNKNOWN
            device.last_error = str(exc)[:512]
            device.last_polled_at = timezone.now()
            device.save(update_fields=["status", "last_error", "last_polled_at", "updated_at"])
    return {
        "polled": online + offline + degraded + fault + errors,
        "online": online,
        "offline": offline,
        "degraded": degraded,
        "fault": fault,
        "errors": errors,
    }


def sync_and_poll() -> dict[str, Any]:
    # A sync failure must never stop device polling.
    try:
        sync_stats = sync_cameras_and_nvrs()
    except Exception as exc:
        logger.exception("infra sync failed")
        sync_stats = {"created": 0, "updated": 0, "skipped": 0, "failed": 1, "error": str(exc)[:200]}
    poll_stats = poll_all_devices()
    return {"sync": sync_stats, "poll": poll_stats}
