"""
Delete journey snapshot JPEGs older than the retention window.

    python manage.py purge_journey_snapshots                 # JOURNEY_SNAPSHOT_RETENTION_DAYS (default 30)
    python manage.py purge_journey_snapshots --days 14 --dry-run

Timeline rows are kept; their snapshot_path is cleared and metadata gets snapshot_purged=true so nothing tries
to re-capture them. Only files under MEDIA_ROOT/journey_snapshots/ are touched (detection clips are not).
"""

from __future__ import annotations

from datetime import date, timedelta

from django.conf import settings
from django.core.files.storage import default_storage
from django.core.management.base import BaseCommand
from django.utils import timezone

PREFIX = "journey_snapshots/"


def _dated_dirs(storage):
    """Yield (date, 'journey_snapshots/YYYY/MM/DD') for every day folder."""
    try:
        years, _ = storage.listdir(PREFIX.rstrip("/"))
    except (FileNotFoundError, OSError, NotImplementedError):
        return
    for y in years:
        try:
            months, _ = storage.listdir(f"{PREFIX}{y}")
        except OSError:
            continue
        for m in months:
            try:
                days, _ = storage.listdir(f"{PREFIX}{y}/{m}")
            except OSError:
                continue
            for d in days:
                try:
                    yield date(int(y), int(m), int(d)), f"{PREFIX}{y}/{m}/{d}"
                except ValueError:
                    continue


class Command(BaseCommand):
    help = "Delete journey snapshot images older than JOURNEY_SNAPSHOT_RETENTION_DAYS."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=None, help="Retention in days (overrides the setting).")
        parser.add_argument("--dry-run", action="store_true", help="Only report what would be deleted.")

    def handle(self, *args, **opts):
        from person_journey.jsonmeta import merge_json_field
        from person_journey.models import JourneyEvent

        days = opts["days"] if opts["days"] is not None else int(getattr(settings, "JOURNEY_SNAPSHOT_RETENTION_DAYS", 30))
        if days < 1:
            self.stderr.write("Retention must be at least 1 day.")
            return
        dry = bool(opts["dry_run"])
        cutoff_day = timezone.localdate() - timedelta(days=days)
        cutoff = timezone.now() - timedelta(days=days)

        files = 0
        for day, folder in _dated_dirs(default_storage):
            if day >= cutoff_day:
                continue
            try:
                _, names = default_storage.listdir(folder)
            except OSError:
                continue
            for name in names:
                files += 1
                if not dry:
                    try:
                        default_storage.delete(f"{folder}/{name}")
                    except OSError:
                        pass

        rows = JourneyEvent.all_objects.filter(created_at__lt=cutoff, snapshot_path__contains=PREFIX)
        row_count = rows.count()
        if not dry:
            for pk in rows.values_list("pk", flat=True).iterator():
                JourneyEvent.all_objects.filter(pk=pk).update(snapshot_path="")
                merge_json_field(
                    JourneyEvent.all_objects, pk, "metadata", {"snapshot_purged": True}, remove_keys=("full_snapshot_path",)
                )

        verb = "Would delete" if dry else "Deleted"
        self.stdout.write(
            self.style.SUCCESS(
                f"{verb} {files} journey snapshot file(s) older than {days} days; "
                f"{row_count} timeline row(s) {'would be unlinked' if dry else 'unlinked'}."
            )
        )
