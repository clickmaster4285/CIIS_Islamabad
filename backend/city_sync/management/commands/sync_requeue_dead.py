"""City: put dead-lettered Outbox events (and failed media uploads) back in the queue."""

from __future__ import annotations

from django.core.management.base import BaseCommand
from django.utils import timezone

from city_sync.identity import is_city
from city_sync.models import MediaQueue, SyncOutbox


class Command(BaseCommand):
    help = (
        "City only. Move DEAD Outbox rows back to pending (fresh attempt budget) and, with "
        "--media, FAILED media uploads too. Run once after deploying the retry fix: older code "
        "dead-lettered events on plain network / HQ outages."
    )

    def add_arguments(self, parser):
        parser.add_argument("--entity", default="", help="Only this entity_type (e.g. attendance)")
        parser.add_argument("--media", action="store_true", help="Also requeue FAILED media uploads")
        parser.add_argument("--media-only", action="store_true", help="Only requeue FAILED media uploads")
        parser.add_argument(
            "--include-missing-files",
            action="store_true",
            help="With --media: also retry uploads that failed because the local file was missing",
        )
        parser.add_argument("--dry-run", action="store_true", help="Only show how many rows would move")

    def handle(self, *args, **options):
        if not is_city():
            self.stderr.write(self.style.ERROR("SYSTEM_ROLE must be city"))
            return
        dry = options["dry_run"]

        if not options["media_only"]:
            qs = SyncOutbox.objects.filter(status=SyncOutbox.STATUS_DEAD)
            if options["entity"]:
                qs = qs.filter(entity_type=options["entity"])
            if dry:
                self.stdout.write(f"Outbox: {qs.count()} dead rows would be requeued")
            else:
                moved = qs.update(
                    status=SyncOutbox.STATUS_PENDING,
                    attempts=0,
                    done_at=None,
                    claimed_at=None,
                    schema_hold=False,
                    next_attempt_at=timezone.now(),
                )
                self.stdout.write(self.style.SUCCESS(f"Outbox: requeued {moved} dead rows"))

        if options["media"] or options["media_only"]:
            from city_sync.media_sync import requeue_failed_media

            if dry:
                qs = MediaQueue.objects.filter(status=MediaQueue.STATUS_FAILED)
                if not options["include_missing_files"]:
                    qs = qs.exclude(last_error="local_file_missing")
                self.stdout.write(f"Media: {qs.count()} failed uploads would be requeued")
            else:
                moved = requeue_failed_media(include_missing=options["include_missing_files"])
                self.stdout.write(self.style.SUCCESS(f"Media: requeued {moved} failed uploads"))
