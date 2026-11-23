"""HQ: retry replica rows whose copy into the business tables failed."""

from __future__ import annotations

import time

from django.core.management.base import BaseCommand
from django.db import close_old_connections

from city_sync.hq_receiver import retry_failed_materializations
from city_sync.identity import is_hq


class Command(BaseCommand):
    help = (
        "HQ only. Re-copy city records whose materialization failed (open materialize_failed "
        "conflicts) from their latest replica. Use --loop to keep retrying periodically."
    )

    def add_arguments(self, parser):
        parser.add_argument("--location", default="", help="Only this LOCATION_CODE")
        parser.add_argument("--limit", type=int, default=200, help="Entities per pass")
        parser.add_argument("--loop", action="store_true", help="Run forever")
        parser.add_argument("--interval", type=float, default=300.0, help="Seconds between passes with --loop")

    def handle(self, *args, **options):
        if not is_hq():
            self.stderr.write(self.style.ERROR("SYSTEM_ROLE must be hq"))
            return
        while True:
            close_old_connections()
            stats = retry_failed_materializations(loc=options["location"], limit=options["limit"])
            if stats["retried"] or not options["loop"]:
                self.stdout.write(str(stats))
            if not options["loop"]:
                break
            time.sleep(max(10.0, options["interval"]))
