"""Delete old activity logs and mobile event receipts.

Run daily (cron / Task Scheduler):
  python manage.py purge_activity_logs
  python manage.py purge_activity_logs --days 30 --receipt-days 7
  python manage.py purge_activity_logs --dry-run

Defaults come from settings ACTIVITY_LOG_RETENTION_DAYS (90) and
MOBILE_EVENT_RECEIPT_RETENTION_DAYS (30). Receipts only de-duplicate offline
uploads, which a phone retries within hours, so a short window is enough.
"""

from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from logs.models import MobileEventReceipt, UserActivityLog

BATCH = 5000


def _purge(qs, dry_run: bool) -> int:
    if dry_run:
        return qs.count()
    total = 0
    while True:
        ids = list(qs.values_list("pk", flat=True)[:BATCH])
        if not ids:
            return total
        deleted, _ = qs.model.objects.filter(pk__in=ids).delete()
        total += deleted


class Command(BaseCommand):
    help = "Delete activity logs and mobile event receipts older than the retention window."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=None, help="Keep activity logs this many days")
        parser.add_argument("--receipt-days", type=int, default=None, help="Keep mobile event receipts this many days")
        parser.add_argument("--dry-run", action="store_true", help="Only count what would be deleted")

    def handle(self, *args, **options):
        days = options["days"]
        if days is None:
            days = int(getattr(settings, "ACTIVITY_LOG_RETENTION_DAYS", 90))
        receipt_days = options["receipt_days"]
        if receipt_days is None:
            receipt_days = int(getattr(settings, "MOBILE_EVENT_RECEIPT_RETENTION_DAYS", 30))
        if days < 1 or receipt_days < 1:
            raise CommandError("Retention must be at least 1 day.")

        now = timezone.now()
        dry = bool(options["dry_run"])
        logs = _purge(UserActivityLog.objects.filter(time__lt=now - timedelta(days=days)), dry)
        receipts = _purge(MobileEventReceipt.objects.filter(created_at__lt=now - timedelta(days=receipt_days)), dry)
        verb = "Would delete" if dry else "Deleted"
        self.stdout.write(
            self.style.SUCCESS(
                f"{verb} {logs} activity log(s) older than {days} day(s) and "
                f"{receipts} mobile event receipt(s) older than {receipt_days} day(s)."
            )
        )
