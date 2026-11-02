"""Delete monitoring data past its retention deadline. Run from cron / a scheduled task."""

from __future__ import annotations

import os
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from employee_monitoring.models import (
    EmployeeAppUsage,
    EmployeeCallRecording,
    EmployeeScreenCapture,
)
from employee_monitoring.services import retention_days, storage_path


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
            if row.storage_key:
                path = storage_path(row.storage_key)
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

        # App usage has no per-row deadline; fall back to the deployment retention window.
        days = retention_days()
        if days:
            old = EmployeeAppUsage.objects.filter(received_at__lt=now - timedelta(days=days))
            if dry_run:
                self.stdout.write(f"Would delete {old.count()} expired app-usage row(s).")
            else:
                deleted, _ = old.delete()
                self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} app-usage row(s)."))
