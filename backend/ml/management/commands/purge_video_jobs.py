"""Delete old Find-in-Video / Video AI Test job folders and stale snapshot temp files."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from ml import job_store


class Command(BaseCommand):
    help = (
        "Remove media/video_search/<job> and media/video_analyze/<job> folders older than "
        "ML_VIDEO_JOB_RETENTION_DAYS (default 7) plus leftover media/detection_clips/_tmp files. "
        "Detection event snapshots themselves are never touched. Running jobs are kept."
    )

    def add_arguments(self, parser):
        parser.add_argument("--days", type=float, default=None, help="Override the retention in days.")
        parser.add_argument("--dry-run", action="store_true", help="List what would be removed.")

    def handle(self, *args, **options):
        days = options["days"]
        if days is None:
            days = float(getattr(settings, "ML_VIDEO_JOB_RETENTION_DAYS", 7))
        days = max(0.0, float(days))
        dry = bool(options["dry_run"])
        cutoff = time.time() - days * 86400
        media = Path(settings.MEDIA_ROOT)

        removed_jobs = 0
        freed = 0
        for sub in ("video_search", "video_analyze"):
            root = media / sub
            if not root.is_dir():
                continue
            for folder in root.iterdir():
                if not folder.is_dir() or not job_store.is_valid_job_id(folder.name):
                    continue
                if self._newest_mtime(folder) >= cutoff:
                    continue
                row = job_store.read_status(folder) or {}
                if str(row.get("status") or "") in job_store.ACTIVE_STATES:
                    continue  # still fresh enough not to be marked stale — leave it alone
                size = self._size(folder)
                self.stdout.write(f"{'Would remove' if dry else 'Removing'} {folder} ({size / 1e6:.1f} MB)")
                if not dry:
                    shutil.rmtree(folder, ignore_errors=True)
                removed_jobs += 1
                freed += size

        removed_tmp = 0
        tmp_root = media / "detection_clips" / "_tmp"
        if tmp_root.is_dir():
            # Snapshot temp files live for seconds; anything older than an hour is a leftover.
            tmp_cutoff = time.time() - 3600
            for path in tmp_root.iterdir():
                try:
                    if path.is_file() and path.stat().st_mtime < tmp_cutoff:
                        freed += path.stat().st_size
                        if not dry:
                            path.unlink()
                        removed_tmp += 1
                except OSError:
                    continue

        verb = "Would free" if dry else "Freed"
        self.stdout.write(
            self.style.SUCCESS(
                f"{verb} {freed / 1e6:.1f} MB: {removed_jobs} job folder(s) older than {days:g} day(s), "
                f"{removed_tmp} snapshot temp file(s)."
            )
        )

    @staticmethod
    def _newest_mtime(folder: Path) -> float:
        newest = 0.0
        for path in [folder, *folder.rglob("*")]:
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
        return newest

    @staticmethod
    def _size(folder: Path) -> int:
        total = 0
        for path in folder.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        return total
