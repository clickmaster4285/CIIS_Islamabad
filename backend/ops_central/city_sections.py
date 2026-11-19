"""
Data for the left-menu screens of /city-control-grid/<city> (Cameras, AI Detections,
Seizures & Detentions, Customs Operations, Infrastructure, Charts & Reports).

Everything is read from what the city already synced to this HQ — no request goes to the
city server. Seizure / detention lists carry case details only: no accused, owner or
driver names, CNICs, phone numbers or addresses. Visitors are counted, never listed.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

from django.db.models import Count, Max, Q
from django.db.models.functions import ExtractHour, TruncDate
from django.utils import timezone

from city_sync.identity import is_hq, location_code
from ops_central.synced_board import (
    GROUPS,
    TREND_KEYS,
    _camera_names,
    _delta,
    _event_row,
    _loc_q,
    _tracked_q,
    _when,
    detection_group,
)

SECTIONS = ("cameras", "ai", "seizures", "customs", "infra", "charts")
RANGES = {1: "Today", 7: "Last 7 days", 30: "Last 30 days", 90: "Last 90 days"}
PAGE_SIZE = 25
LIST_LIMIT = 50


class UnknownCity(Exception):
    pass


# ----------------------------------------------------------------------- helpers


def resolve_city(city_id: str) -> tuple[str, str, bool]:
    """Board id ("di-khan") -> (location code, display name, is HQ). Raises UnknownCity."""
    from city_sync.models import SyncSite
    from ops_central.city_control import display_city_name

    hq = location_code()
    wanted = (city_id or "").strip().lower()
    if hq.lower().replace("_", "-") == wanted:
        return hq, display_city_name(hq), True
    for site in SyncSite.objects.all():
        if site.location_code.lower().replace("_", "-") == wanted:
            return site.location_code, site.name or display_city_name(site.location_code), False
    raise UnknownCity(city_id)


def parse_days(raw) -> int:
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return 7
    return days if days in RANGES else 7


def _window(days: int, now) -> tuple[datetime, datetime]:
    """(start, previous start). "Today" starts at local midnight; others are rolling days."""
    if days == 1:
        start = timezone.make_aware(datetime.combine(timezone.localdate(), time.min))
        return start, start - timedelta(days=1)
    start = now - timedelta(days=days)
    return start, start - timedelta(days=days)


def _day_list(days: int) -> list:
    n = max(days, 1)
    first = timezone.localdate() - timedelta(days=n - 1)
    return [first + timedelta(days=i) for i in range(n)]


def _day_label(d, days: int) -> str:
    return d.strftime("%a") if days <= 7 else d.strftime("%b %d")


def _money(raw) -> Decimal:
    text = str(raw or "").replace(",", "").replace("PKR", "").replace("Rs", "").strip()
    try:
        return Decimal(text) if text else Decimal(0)
    except InvalidOperation:
        return Decimal(0)


def _when_dict(dt) -> dict[str, str]:
    absolute, relative, iso = _when(dt)
    return {"at": iso, "time": absolute if dt else "—", "ago": relative if dt else "—"}


def _city_state(code: str, is_hq_row: bool, now) -> tuple[str, dict, int | None]:
    """(sync state, heartbeat raw, heartbeat age seconds)."""
    if is_hq_row:
        return "ONLINE", {}, 0
    from city_sync.models import HqCityStatus, SyncSite
    from city_sync.views import city_health_state

    st = HqCityStatus.objects.filter(location_code=code).first()
    if not st:
        return "UNKNOWN", {}, None
    state, age = city_health_state(st, now)
    if SyncSite.objects.filter(location_code=code, is_paused=True).exists():
        state = "PAUSED"
    return state, st.raw or {}, age


# ----------------------------------------------------------------------- cameras


def _cameras(code: str, hq: str, is_hq_row: bool, now, params) -> dict[str, Any]:
    from cameras.models import DetectionEvent
    from ops_central.synced_board import _hq_cameras

    state, raw, _age = _city_state(code, is_hq_row, now)
    online, total = _hq_cameras(code, hq, raw, state, now)
    loc = _loc_q(code, hq)
    day = now - timedelta(hours=24)

    if is_hq_row:
        from cameras.live_status import online_camera_ids
        from cameras.models import Camera

        try:
            live = online_camera_ids()
        except Exception:  # noqa: BLE001
            live = set()
        cams = [
            {
                "key": c.pk,
                "code": c.code,
                "name": c.name,
                "zone": c.zone,
                "location": c.location,
                "channel": c.channel,
                "site": c.nvr.site.name if c.nvr_id and c.nvr.site_id else "",
                # Same live check as the online count; cities only send totals.
                "status": ("online" if c.pk in live else "offline") if c.is_active else "inactive",
            }
            for c in Camera.objects.select_related("nvr__site").order_by("name")
        ]
        stats = (
            DetectionEvent.objects.filter(loc, camera__isnull=False)
            .values("camera_id")
            .annotate(n=Count("pk", filter=Q(created_at__gte=day)), last=Max("created_at"))
        )
        by_cam = {s["camera_id"]: s for s in stats}
    else:
        from city_sync.models import CityCamera

        cams = [
            {
                "key": c.origin_pk,
                "code": c.code,
                "name": c.name or c.code or f"Camera {c.origin_pk}",
                "zone": c.zone,
                "location": c.location,
                "channel": c.channel,
                "site": c.site_name,
                "status": None,
            }
            for c in CityCamera.objects.filter(location_code=code).order_by("name")
        ]
        stats = (
            DetectionEvent.objects.filter(location_code=code, source_camera_id__isnull=False)
            .values("source_camera_id")
            .annotate(n=Count("pk", filter=Q(created_at__gte=day)), last=Max("created_at"))
        )
        by_cam = {s["source_camera_id"]: s for s in stats}

    q = (params.get("q") or "").strip().lower()
    out = []
    for cam in cams:
        s = by_cam.get(cam["key"]) or {}
        cam["detections_24h"] = s.get("n", 0)
        cam["last_detection"] = _when_dict(s.get("last")) if s.get("last") else None
        if q and not any(q in str(cam.get(f) or "").lower() for f in ("name", "code", "zone", "location", "site")):
            continue
        out.append(cam)

    zones = sorted({c["zone"] for c in cams if c["zone"]})
    return {
        "summary": {
            "online": online,
            "offline": max(0, total - online),
            "total": total,
            "listed": len(cams),
            "active_24h": sum(1 for c in cams if c["detections_24h"]),
            "per_camera_status": is_hq_row,
        },
        "zones": zones,
        "cameras": out,
    }


# ----------------------------------------------------------------------- AI detections


def _ai(code: str, hq: str, days: int, now, params) -> dict[str, Any]:
    from cameras.models import DetectionEvent

    start, prev = _window(days, now)
    base = DetectionEvent.objects.filter(_loc_q(code, hq), _tracked_q())
    in_range = base.filter(created_at__gte=start)

    counts = {label: 0 for label, _ in GROUPS}
    for row in in_range.values("class_name").annotate(n=Count("pk")):
        group = detection_group(row["class_name"])
        if group:
            counts[group] += row["n"]
    total = sum(counts.values())
    before = base.filter(created_at__gte=prev, created_at__lt=start).count()

    by_hour = [{"hour": f"{h:02d}", "count": 0} for h in range(24)]
    for row in in_range.annotate(h=ExtractHour("created_at", tzinfo=timezone.get_current_timezone())).values(
        "h"
    ).annotate(n=Count("pk")):
        if row["h"] is not None:
            by_hour[row["h"]]["count"] = row["n"]

    # Busiest cameras
    cam_field = "camera_id" if code == hq else "source_camera_id"
    top = list(
        in_range.exclude(**{f"{cam_field}__isnull": True})
        .values(cam_field)
        .annotate(n=Count("pk"))
        .order_by("-n")[:5]
    )
    names = _top_camera_names(code, hq, [t[cam_field] for t in top])
    top_cameras = [{"key": t[cam_field], "camera": names.get(t[cam_field], f"Camera {t[cam_field]}"), "count": t["n"]} for t in top]

    # Filtered, paged list
    listed = in_range
    wanted_type = params.get("type") or ""
    for label, members in GROUPS:
        if wanted_type == label:
            tq = Q()
            for m in members:
                tq |= Q(class_name__iexact=m)
            listed = listed.filter(tq)
    if params.get("camera"):
        try:
            listed = listed.filter(**{cam_field: int(params["camera"])})
        except ValueError:
            pass
    if params.get("alerts") in ("1", "true"):
        listed = listed.filter(is_alert=True)
    try:
        page = max(1, int(params.get("page") or 1))
    except ValueError:
        page = 1
    matched = listed.count()
    events = list(listed.select_related("camera").order_by("-created_at")[(page - 1) * PAGE_SIZE : page * PAGE_SIZE])
    names_all = _camera_names(events)
    items = []
    for e in events:
        row = _event_row(e, names_all, hq)
        row["date"] = timezone.localtime(e.created_at).strftime("%b %d, %H:%M")
        row["confidence"] = round(float(e.confidence or 0) * 100)
        items.append(row)

    return {
        "summary": {
            "total": total,
            "delta": _delta(total, before),
            "alerts": in_range.filter(is_alert=True).count(),
            "by_type": counts,
        },
        "by_hour": by_hour,
        "top_cameras": top_cameras,
        "items": items,
        "page": page,
        "pages": max(1, -(-matched // PAGE_SIZE)),
        "matched": matched,
    }


def _top_camera_names(code: str, hq: str, keys: list) -> dict:
    if not keys:
        return {}
    if code == hq:
        from cameras.models import Camera

        return {c.pk: c.name for c in Camera.objects.filter(pk__in=keys)}
    from city_sync.models import CityCamera

    return {
        c.origin_pk: c.name or c.code
        for c in CityCamera.objects.filter(location_code=code, origin_pk__in=keys)
    }


# ----------------------------------------------------------------------- seizures & detentions


def _daily(days: int, keys: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per day: {day, <key>: count} for each (key -> queryset) pair."""
    day_list = _day_list(days)
    rows = {d: {"day": _day_label(d, days), **{k: 0 for k in keys}} for d in day_list}
    for key, q in keys.items():
        for r in q.filter(created_at__date__gte=day_list[0]).annotate(d=TruncDate("created_at")).values("d").annotate(
            n=Count("pk")
        ):
            if r["d"] in rows:
                rows[r["d"]][key] = r["n"]
    return [rows[d] for d in day_list]


def _seizures(code: str, hq: str, days: int, now, params) -> dict[str, Any]:
    from detentions.models import DetentionMemo
    from seizure_management.models import NoteSheet

    start, prev = _window(days, now)
    loc = _loc_q(code, hq)
    sheets_all = NoteSheet.objects.filter(loc, is_deleted=False)
    memos_all = DetentionMemo.objects.filter(loc, is_deleted=False)
    sheets = sheets_all.filter(created_at__gte=start)
    memos = memos_all.filter(created_at__gte=start)

    q = (params.get("q") or "").strip()
    if q:
        sheets = sheets.filter(
            Q(note_sheet_no__icontains=q) | Q(case_no__icontains=q) | Q(reference_number__icontains=q) | Q(subject__icontains=q)
        )
        memos = memos.filter(Q(case_no__icontains=q) | Q(reference_number__icontains=q) | Q(fir_number__icontains=q))

    sheet_rows = []
    for s in sheets.prefetch_related("items").order_by("-created_at")[:LIST_LIMIT]:
        items = [i for i in s.items.all() if not i.is_deleted]
        sheet_rows.append(
            {
                "id": str(s.pk),
                "note_sheet_no": s.note_sheet_no,
                "case_no": s.case_no,
                "reference_number": s.reference_number,
                "subject": s.subject,
                "office": s.office,
                "officer": s.prepared_by,
                "designation": s.designation,
                "status": s.get_status_display(),
                "priority": s.get_priority_display(),
                "items": len(items),
                "value_pkr": float(sum((_money(i.estimated_value) for i in items), Decimal(0))),
                "goods": ", ".join(i.product for i in items[:3] if i.product),
                **_when_dict(s.created_at),
            }
        )

    memo_rows = []
    for m in memos.prefetch_related("goods_lines").order_by("-created_at")[:LIST_LIMIT]:
        lines = [g for g in m.goods_lines.all() if not g.is_deleted]
        memo_rows.append(
            {
                "id": str(m.pk),
                "case_no": m.case_no,
                "reference_number": m.reference_number,
                "fir_number": m.fir_number,
                "detention_type": m.detention_type,
                "place_of_detention": m.place_of_detention,
                "officer": m.receipt_officer or m.created_by,
                "status": m.disposition_status or m.settlement_status or "—",
                "items": len(lines),
                "value_pkr": float(sum((_money(g.assessable_value_pkr) for g in lines), Decimal(0))),
                "goods": ", ".join(g.description for g in lines[:3] if g.description),
                **_when_dict(m.created_at),
            }
        )

    from detentions.models import DetentionMemoGoodsLine
    from seizure_management.models import NoteSheetItem

    sheet_values = NoteSheetItem.objects.filter(
        is_deleted=False, note_sheet__in=sheets_all.filter(created_at__gte=start)
    ).values_list("estimated_value", flat=True)
    memo_values = DetentionMemoGoodsLine.objects.filter(
        is_deleted=False, memo__in=memos_all.filter(created_at__gte=start)
    ).values_list("assessable_value_pkr", flat=True)

    n_sheets = sheets_all.filter(created_at__gte=start).count()
    n_memos = memos_all.filter(created_at__gte=start).count()
    return {
        "summary": {
            "seizures": n_sheets,
            "seizure_delta": _delta(n_sheets, sheets_all.filter(created_at__gte=prev, created_at__lt=start).count()),
            "detentions": n_memos,
            "detention_delta": _delta(n_memos, memos_all.filter(created_at__gte=prev, created_at__lt=start).count()),
            "seizure_value_pkr": float(sum((_money(v) for v in sheet_values), Decimal(0))),
            "detention_value_pkr": float(sum((_money(v) for v in memo_values), Decimal(0))),
            "seizure_status": _status_counts(sheets_all.filter(created_at__gte=start), "status", NoteSheet),
            "detention_status": _status_counts(memos_all.filter(created_at__gte=start), "disposition_status", None),
        },
        "daily": _daily(max(days, 7), {"seizures": sheets_all, "detentions": memos_all}),
        "note_sheets": sheet_rows,
        "detention_memos": memo_rows,
        "list_limit": LIST_LIMIT,
    }


def _status_counts(qs, field: str, model) -> list[dict[str, Any]]:
    labels = dict(model.STATUS_CHOICES) if model is not None and hasattr(model, "STATUS_CHOICES") else {}
    rows = qs.values(field).annotate(n=Count("pk")).order_by("-n")
    return [{"status": labels.get(r[field], r[field] or "Not set"), "count": r["n"]} for r in rows]


# ----------------------------------------------------------------------- customs operations


def _customs(code: str, hq: str, days: int, now, params) -> dict[str, Any]:
    from visitors.models import SecurityAlert, Vehicle, Visitor, ZoneAccessLog
    from warehouse.models import DestructionAlert, MemoDistribution, WarehouseStockItem

    start, _prev = _window(days, now)
    loc = _loc_q(code, hq)
    today = timezone.localdate()
    midnight = timezone.make_aware(datetime.combine(today, time.min))

    stock = WarehouseStockItem.objects.filter(loc, is_deleted=False)
    stock_rows = [
        {
            "case_ref": s.case_ref,
            "qr_code": s.qr_code,
            "description": s.description,
            "quantity": float(s.quantity or 0),
            "unit": s.unit,
            "godown": s.godown_warehouse,
            "status": s.status or "—",
            **_when_dict(s.updated_at),
        }
        for s in stock.order_by("-updated_at")[:20]
    ]

    dists = MemoDistribution.objects.filter(loc, is_deleted=False, created_at__gte=start)
    dist_rows = [
        {
            "case_no": d.detention_case_no,
            "outcome": d.get_outcome_display() or "—",
            "status": d.get_status_display(),
            "performed_by": d.performed_by,
            "smoke_fire": d.smoke_fire_detected,
            **_when_dict(d.created_at),
        }
        for d in dists.order_by("-created_at")[:10]
    ]

    d_alerts = DestructionAlert.objects.filter(loc, is_deleted=False)
    d_alert_rows = [
        {
            "case_no": a.detention_case_no,
            "type": a.alert_type,
            "severity": a.get_severity_display(),
            "message": a.message,
            "acknowledged": a.acknowledged,
            **_when_dict(a.created_at),
        }
        for a in d_alerts.filter(created_at__gte=start).order_by("-created_at")[:10]
    ]

    visitors = Visitor.objects.filter(loc, is_deleted=False)
    visitors_today = visitors.filter(Q(visit_date=today) | Q(visit_date__isnull=True, created_at__gte=midnight))
    stages = dict(visitors_today.values_list("flow_stage").annotate(n=Count("pk")))
    scans = ZoneAccessLog.objects.filter(loc, is_deleted=False, scanned_at__gte=midnight)
    sec = SecurityAlert.objects.filter(loc, is_deleted=False)
    alert_labels = dict(SecurityAlert.ALERT_TYPE_CHOICES) if hasattr(SecurityAlert, "ALERT_TYPE_CHOICES") else {}
    # No names or message text: messages can name the visitor.
    sec_rows = [
        {
            "type": alert_labels.get(a.alert_type, a.alert_type),
            "severity": a.get_severity_display(),
            "zone": a.zone,
            "gate": a.gate,
            "acknowledged": a.acknowledged,
            **_when_dict(a.created_at),
        }
        for a in sec.filter(created_at__gte=start).order_by("-created_at")[:10]
    ]

    return {
        "warehouse": {
            "items": stock.count(),
            "by_status": [
                {"status": r["status"] or "Not set", "count": r["n"]}
                for r in stock.values("status").annotate(n=Count("pk")).order_by("-n")
            ],
            "recent": stock_rows,
        },
        "distributions": {
            "total": dists.count(),
            "destructed": dists.filter(outcome=MemoDistribution.OUTCOME_DESTRUCTED).count(),
            "inventory": dists.filter(outcome=MemoDistribution.OUTCOME_INVENTORY).count(),
            "recent": dist_rows,
        },
        "destruction_alerts": {
            "open": d_alerts.filter(acknowledged=False).count(),
            "in_range": d_alerts.filter(created_at__gte=start).count(),
            "recent": d_alert_rows,
        },
        "gate_today": {
            "visitors": visitors_today.count(),
            "on_premises": visitors_today.exclude(flow_stage__in=["exited", ""]).count(),
            "exited": stages.get("exited", 0),
            "vehicles": Vehicle.objects.filter(loc, is_deleted=False, created_at__gte=midnight).count(),
            "scans_allowed": scans.filter(allowed=True).count(),
            "scans_denied": scans.filter(allowed=False).count(),
            "by_stage": [{"stage": k or "Not set", "count": v} for k, v in sorted(stages.items(), key=lambda x: -x[1])],
        },
        "security_alerts": {
            "open": sec.filter(acknowledged=False).count(),
            "in_range": sec.filter(created_at__gte=start).count(),
            "recent": sec_rows,
        },
        "daily_visitors": _daily(max(days, 7), {"visitors": visitors, "vehicles": Vehicle.objects.filter(loc, is_deleted=False)}),
    }


# ----------------------------------------------------------------------- infrastructure


def _infra(code: str, hq: str, is_hq_row: bool, now) -> dict[str, Any]:
    from ops_central.server_vitals import row_health

    state, raw, age = _city_state(code, is_hq_row, now)
    health = row_health(is_hq_row=is_hq_row, state=state, raw=raw, heartbeat_age=age, now=now)
    devices: list[dict[str, Any]] = []
    if is_hq_row:
        # Only HQ's own devices are here; cities send counts by type, not the device list.
        try:
            from infrastructure_monitoring.models import InfraDevice

            devices = [
                {
                    "name": d.name,
                    "type": d.get_device_type_display(),
                    "status": d.status,
                    "location": d.install_location,
                    "site": d.site.name if d.site_id else "",
                    "last_seen": _when_dict(d.last_seen_at) if d.last_seen_at else None,
                }
                for d in InfraDevice.objects.filter(is_active=True).select_related("site").order_by("device_type", "name")
            ]
        except Exception:  # noqa: BLE001 - module optional
            devices = []

    infra = health.get("infra") or {}
    type_labels = {"camera": "Cameras", "nvr": "NVRs", "ups": "UPS", "inverter": "Inverters", "network": "Network", "server": "Servers"}
    by_type = [
        {"type": type_labels.get(t, t), "total": v.get("total", 0), "online": v.get("online", 0)}
        for t, v in (infra.get("by_type") or {}).items()
    ]
    heartbeat = None
    if not is_hq_row and age is not None:
        heartbeat = _when_dict(now - timedelta(seconds=age))
    return {
        "health": health,
        "state": state,
        "heartbeat": heartbeat,
        "by_type": by_type,
        "devices": devices,
        "devices_listed": is_hq_row,
    }


# ----------------------------------------------------------------------- charts & reports


def _charts(code: str, hq: str, days: int, now) -> dict[str, Any]:
    from cameras.models import DetectionEvent
    from detentions.models import DetentionMemo
    from seizure_management.models import NoteSheet
    from visitors.models import Visitor

    n = max(days, 7)
    loc = _loc_q(code, hq)
    day_list = _day_list(n)
    trend = {d: {"day": _day_label(d, n), "people": 0, "vehicles": 0, "weapons": 0, "plates": 0} for d in day_list}
    for r in (
        DetectionEvent.objects.filter(loc, _tracked_q(), created_at__date__gte=day_list[0])
        .annotate(d=TruncDate("created_at"))
        .values("d", "class_name")
        .annotate(n=Count("pk"))
    ):
        group = detection_group(r["class_name"])
        if group and r["d"] in trend:
            trend[r["d"]][TREND_KEYS[group]] += r["n"]

    ops = _daily(
        n,
        {
            "seizures": NoteSheet.objects.filter(loc, is_deleted=False),
            "detentions": DetentionMemo.objects.filter(loc, is_deleted=False),
            "visitors": Visitor.objects.filter(loc, is_deleted=False),
        },
    )
    return {
        "trend": [trend[d] for d in day_list],
        "ops": ops,
        "comparison": _comparison(days, now),
    }


def _comparison(days: int, now) -> list[dict[str, Any]]:
    """Every city side by side for the same period (Comparative Report)."""
    from cameras.models import DetectionEvent
    from detentions.models import DetentionMemo
    from seizure_management.models import NoteSheet
    from ops_central.synced_board import synced_city_rows

    start, _prev = _window(days, now)
    hq = location_code()
    out = []
    for row in synced_city_rows(with_detail=False):
        loc = _loc_q(row["location_code"], hq)
        out.append(
            {
                "id": row["id"],
                "city": row["city"],
                "is_hq": row["is_hq"],
                "status": row["status"],
                "cameras_online": row["cameras_online"],
                "cameras_total": row["cameras_total"],
                # Same four types as the AI screen (people, vehicles, weapons, plates).
                "detections": DetectionEvent.objects.filter(loc, _tracked_q(), created_at__gte=start).count(),
                "seizures": NoteSheet.objects.filter(loc, is_deleted=False, created_at__gte=start).count(),
                "detentions": DetentionMemo.objects.filter(loc, is_deleted=False, created_at__gte=start).count(),
                "staff_total": row.get("staff_total", 0),
            }
        )
    return out


# ----------------------------------------------------------------------- entry point


def build_city_section(city_id: str, section: str, params) -> dict[str, Any]:
    """Payload for one menu screen. Raises UnknownCity / ValueError (unknown section)."""
    if section not in SECTIONS:
        raise ValueError(section)
    if not is_hq():
        raise UnknownCity(city_id)
    code, name, is_hq_row = resolve_city(city_id)
    hq = location_code()
    now = timezone.now()
    days = parse_days(params.get("days"))

    if section == "cameras":
        data = _cameras(code, hq, is_hq_row, now, params)
    elif section == "ai":
        data = _ai(code, hq, days, now, params)
    elif section == "seizures":
        data = _seizures(code, hq, days, now, params)
    elif section == "customs":
        data = _customs(code, hq, days, now, params)
    elif section == "infra":
        data = _infra(code, hq, is_hq_row, now)
    else:
        data = _charts(code, hq, days, now)

    return {
        "city": {"id": city_id, "location_code": code, "name": name, "is_hq": is_hq_row},
        "section": section,
        "days": days,
        "range_label": RANGES[days],
        "generated_at": now.isoformat(),
        **data,
    }
