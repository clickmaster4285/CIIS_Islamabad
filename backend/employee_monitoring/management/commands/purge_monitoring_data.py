"""Delete monitoring data past its retention deadline. Run from cron / a scheduled task."""

from __future__ import annotations

import os
from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from employee_monitoring.models import (
    EmployeeAppSession,
    EmployeeAppUsage,
    EmployeeCallRecord,
    EmployeeCallRecording,
    EmployeeScreenCapture,
    MonitoringAudit,
)
from employee_monitoring.services import retention_days, safe_storage_path


def audit_retention_days() -> int:
    """MONITORING_AUDIT_RETENTION_DAYS if set, else the deployment retention window (0 = keep)."""
    try:
        value = getattr(settings, "MONITORING_AUDIT_RETENTION_DAYS", None)
        return int(value) if value not in (None, "") else retention_days()
    except (TypeError, ValueError):
        return retention_days()


class Command(BaseCommand):
    help = "Delete monitoring data (and any stored files) past its retention deadline."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Report what would be deleted without deleting.")

    def _purge_with_files(self, qs, label: str, dry_run: bool) -> None:
        """Delete rows that carry a private file, removing the file first."""
        count = qs.count()
        if dry_run:
            self.stdout.write(f"Would delete {count} expired {label}.")
            return
        removed_files = 0
        for row in qs.iterator():
            # safe_storage_path refuses keys that resolve outside the private root.
            path = safe_storage_path(row.storage_key)
            if path:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        removed_files += 1
                except OSError:
                    pass
        deleted, _ = qs.delete()
        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} {label} row(s); removed {removed_files} file(s)."))

    def handle(self, *args, **options):
        now = timezone.now()
        dry_run = options["dry_run"]

        self._purge_with_files(
            EmployeeCallRecording.objects.filter(retention_deadline__isnull=False, retention_deadline__lt=now),
            "recording", dry_run,
        )
        self._purge_with_files(
            EmployeeScreenCapture.objects.filter(retention_deadline__isnull=False, retention_deadline__lt=now),
            "screen capture", dry_run,
        )

        # Metadata rows have no per-row deadline; fall back to the deployment retention window.
        days = retention_days()
        if days:
            cutoff = now - timedelta(days=days)
            self._purge_rows(EmployeeAppUsage.objects.filter(received_at__lt=cutoff), "app-usage", dry_run)
            self._purge_rows(EmployeeAppSession.objects.filter(received_at__lt=cutoff), "app-session", dry_run)
            self._purge_rows(EmployeeCallRecord.objects.filter(received_at__lt=cutoff), "call-metadata", dry_run)

        audit_days = audit_retention_days()
        if audit_days:
            self._purge_rows(
                MonitoringAudit.objects.filter(created_at__lt=now - timedelta(days=audit_days)), "audit", dry_run,
            )

    def _purge_rows(self, qs, label: str, dry_run: bool) -> None:
        if dry_run:
            self.stdout.write(f"Would delete {qs.count()} expired {label} row(s).")
            return
        deleted, _ = qs.delete()
        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} {label} row(s)."))
