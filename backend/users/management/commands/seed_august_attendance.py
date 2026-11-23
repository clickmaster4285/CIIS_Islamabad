"""
Seed August attendance for SWH Ratta Kulachi warehouse staff.

Demo data only: refuses to run unless DEBUG is on or --allow-production is given.

Usage:
  python manage.py seed_august_attendance
  python manage.py seed_august_attendance --roster C:/secure/roster.json
  python manage.py seed_august_attendance --year 2026
  python manage.py seed_august_attendance --dry-run
  python manage.py seed_august_attendance --clear
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from datetime import date, datetime, time, timedelta

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from users.models import Attendance, Staff

# Sample roster with obviously fake identifiers (real CNICs / personal numbers must never
# live in source control). For a real roster pass --roster path/to/roster.json, a JSON list
# of {"personal_number", "full_name", "cnic"} kept outside the repository.
AUGUST_STAFF = [
    {"personal_number": f"TEST{n:04d}", "full_name": f"Sample Employee {n:02d}", "cnic": f"00000-0000000-{n % 10}"}
    for n in range(1, 6)
]

MONTH = 8

# Random windows (local time)
CHECK_IN_EARLIEST = (8, 30)
CHECK_IN_LATEST = (9, 25)
CHECK_OUT_EARLIEST = (16, 30)
CHECK_OUT_LATEST = (17, 45)
LATE_AFTER = time(9, 5)


def _random_time(rng: random.Random, start: tuple[int, int], end: tuple[int, int]) -> time:
    start_m = start[0] * 60 + start[1]
    end_m = end[0] * 60 + end[1]
    picked = rng.randint(start_m, end_m)
    return time(picked // 60, picked % 60)


def _random_check_times(rng: random.Random, day: date, tz) -> tuple[datetime, datetime, str]:
    in_t = _random_time(rng, CHECK_IN_EARLIEST, CHECK_IN_LATEST)
    out_t = _random_time(rng, CHECK_OUT_EARLIEST, CHECK_OUT_LATEST)
    check_in = timezone.make_aware(datetime.combine(day, in_t), tz)
    check_out = timezone.make_aware(datetime.combine(day, out_t), tz)
    if check_out <= check_in:
        check_out = check_in + timedelta(hours=rng.randint(7, 9), minutes=rng.randint(0, 45))
    status = Attendance.STATUS_LATE if in_t > LATE_AFTER else Attendance.STATUS_PRESENT
    return check_in, check_out, status


def _normalize_cnic(value: str) -> str:
    return "".join(ch for ch in (value or "") if ch.isdigit())


def _resolve_staff(row: dict) -> Staff | None:
    pn = str(row.get("personal_number") or "").strip()
    cnic = _normalize_cnic(str(row.get("cnic") or ""))
    if pn:
        staff = Staff.objects.filter(personal_number=pn).first()
        if staff:
            return staff
    if cnic:
        for candidate in Staff.objects.exclude(cnic__isnull=True).exclude(cnic=""):
            if _normalize_cnic(candidate.cnic) == cnic:
                return candidate
    name = str(row.get("full_name") or "").strip()
    if name:
        return Staff.objects.filter(full_name__iexact=name).first()
    return None


def _august_weekdays(year: int) -> list[date]:
    days: list[date] = []
    d = date(year, MONTH, 1)
    while d.month == MONTH:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


class Command(BaseCommand):
    help = "Seed randomized August attendance (demo data) for a staff roster."

    def add_arguments(self, parser):
        parser.add_argument("--year", type=int, default=2026, help="Year for August seed (default: 2026)")
        parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
        parser.add_argument("--roster", type=str, default="", help="JSON roster file (kept outside git)")
        parser.add_argument(
            "--allow-production",
            action="store_true",
            help="Run even though DEBUG is off (writes fake attendance into a live database)",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Delete existing August attendance for matched staff before seeding",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed for reproducible check-in/out times (optional)",
        )

    def handle(self, *args, **options):
        if not settings.DEBUG and not options["allow_production"]:
            raise CommandError(
                "This writes fake attendance. Refusing with DEBUG off; pass --allow-production if you really mean it."
            )
        roster = AUGUST_STAFF
        if options["roster"]:
            path = Path(options["roster"])
            if not path.is_file():
                raise CommandError(f"Roster file not found: {path}")
            roster = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(roster, list):
                raise CommandError("Roster must be a JSON list")
        year = int(options["year"])
        dry_run = bool(options["dry_run"])
        clear = bool(options["clear"])
        tz = timezone.get_current_timezone()
        workdays = _august_weekdays(year)
        rng = random.Random(options["seed"] if options["seed"] is not None else year)

        resolved: list[tuple[Staff, dict]] = []
        missing: list[str] = []

        for row in roster:
            staff = _resolve_staff(row)
            if staff:
                resolved.append((staff, row))
            else:
                missing.append(f"{row['full_name']} (PN {row['personal_number']}, CNIC {row['cnic']})")

        if missing:
            self.stdout.write(self.style.WARNING("Staff not found in DB (skipped):"))
            for line in missing:
                self.stdout.write(f"  - {line}")

        if not resolved:
            self.stderr.write(self.style.ERROR("No staff matched. Add staff records first, then re-run."))
            return

        self.stdout.write(
            f"Matched {len(resolved)} staff × {len(workdays)} weekdays "
            f"= {len(resolved) * len(workdays)} attendance rows for August {year}"
        )

        if dry_run:
            for staff, row in resolved:
                self.stdout.write(f"  [dry-run] {staff.full_name} ({row['personal_number']})")
            return

        created = 0
        updated = 0
        deleted = 0

        with transaction.atomic():
            if clear:
                staff_ids = [s.id for s, _ in resolved]
                qs = Attendance.objects.filter(
                    staff_id__in=staff_ids,
                    date__year=year,
                    date__month=MONTH,
                )
                deleted, _ = qs.delete()
                self.stdout.write(self.style.WARNING(f"Cleared {deleted} existing August {year} rows"))

            for staff, _row in resolved:
                for day in workdays:
                    check_in, check_out, status = _random_check_times(rng, day, tz)
                    defaults = {
                        "check_in": check_in,
                        "check_out": check_out,
                        "status": status,
                        "source": Attendance.SOURCE_MANUAL,
                        "notes": f"Seeded August {year} attendance (randomized times)",
                    }
                    if staff.user_id:
                        defaults["user"] = staff.user

                    obj, was_created = Attendance.objects.update_or_create(
                        staff=staff,
                        date=day,
                        defaults=defaults,
                    )
                    if was_created:
                        created += 1
                    else:
                        updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Done — created {created}, updated {updated}, cleared {deleted} "
                f"(August {year}, {len(resolved)} employees)"
            )
        )
