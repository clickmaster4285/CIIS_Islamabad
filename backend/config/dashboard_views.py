"""GET /api/dashboard/glance/ — live numbers for the dashboard "At a Glance" cards."""

from __future__ import annotations

from datetime import timedelta

from django.db.models import Q
from django.utils import timezone
from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView

PEAK_WINDOW_HOURS = 4
ANALYTICS_DAYS = 30


def _hour_label(hour: int) -> str:
    hour %= 24
    suffix = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}{suffix}"


def _peak_hours(entry_times) -> str | None:
    """Busiest PEAK_WINDOW_HOURS-hour window of the day (local time), e.g. "10AM - 2PM"."""
    per_hour = [0] * 24
    for ts in entry_times:
        per_hour[timezone.localtime(ts).hour] += 1
    if not any(per_hour):
        return None
    # Windows stay inside one day (no wrap past midnight)
    best_start = max(
        range(24 - PEAK_WINDOW_HOURS + 1),
        key=lambda h: sum(per_hour[h : h + PEAK_WINDOW_HOURS]),
    )
    return f"{_hour_label(best_start)} - {_hour_label(best_start + PEAK_WINDOW_HOURS)}"


def _warehouse_stats() -> dict:
    from warehouse.models import MemoDistribution, WarehouseStockItem

    return {
        # Stock still held in the warehouse
        "active_operations": WarehouseStockItem.objects.filter(
            is_deleted=False, status__iexact="In Custody"
        ).count(),
        # Distributions started but not finished yet
        "pending_items": MemoDistribution.objects.filter(
            status=MemoDistribution.STATUS_RECORDING
        ).count(),
    }


def _analytics_stats() -> dict:
    from visitors.models import ZoneAccessLog

    since = timezone.now() - timedelta(days=ANALYTICS_DAYS)
    scans = ZoneAccessLog.objects.filter(allowed=True, scanned_at__gte=since)
    entries = scans.filter(scan_type="entry")
    entered = set(entries.values_list("visitor_id", flat=True).distinct())
    exited = set(scans.filter(scan_type="exit").values_list("visitor_id", flat=True).distinct())
    efficiency = round(100 * len(entered & exited) / len(entered)) if entered else None
    return {
        "peak_hours": _peak_hours(entries.values_list("scanned_at", flat=True).iterator()),
        # Visits completed: visitors who checked out ÷ visitors who checked in
        "efficiency_pct": efficiency,
        "window_days": ANALYTICS_DAYS,
    }


def _hr_stats() -> dict:
    from users.models import Attendance, Staff

    today = timezone.localdate()
    present = Attendance.objects.filter(
        date=today,
        is_deleted=False,
        status__in=[Attendance.STATUS_PRESENT, Attendance.STATUS_LATE, Attendance.STATUS_HALF_DAY],
    )
    # One person counts once, whether the row is linked by staff profile or by user
    people = {
        f"s{staff_id}" if staff_id else f"u{user_id}"
        for staff_id, user_id in present.values_list("staff_id", "user_id")
        if staff_id or user_id
    }
    return {
        "employees": Staff.objects.filter(is_deleted=False).count(),
        "present_today": len(people),
    }


class DashboardGlanceAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response(
            {
                "warehouse": _warehouse_stats(),
                "analytics": _analytics_stats(),
                "hr": _hr_stats(),
                "generated_at": timezone.now().isoformat(),
            }
        )
