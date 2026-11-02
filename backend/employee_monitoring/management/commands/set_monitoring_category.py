"""Turn one monitoring category on or off in the ACTIVE policy, leaving the others untouched.

    python manage.py set_monitoring_category android_call_recording        # turn on
    python manage.py set_monitoring_category app_usage --off               # turn off

Adds the category (with its canonical label/description) if the active policy predates it.
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from employee_monitoring.models import MonitoringPolicy
from employee_monitoring.services import retention_days

from .seed_monitoring_policy import default_categories


class Command(BaseCommand):
    help = "Enable or disable one category in the active monitoring policy."

    def add_arguments(self, parser):
        parser.add_argument("key", help="Category key, e.g. android_call_recording.")
        parser.add_argument("--off", action="store_true", help="Disable instead of enable.")

    @transaction.atomic
    def handle(self, *args, **options):
        key = options["key"]
        collected = not options["off"]

        policy = MonitoringPolicy.objects.filter(is_active=True).order_by("-effective_at").first()
        if policy is None:
            raise CommandError("No active policy. Run seed_monitoring_policy first.")

        canonical = {c["key"]: c for c in default_categories(retention_days() or None)}
        if key not in canonical:
            raise CommandError(f"Unknown category '{key}'. Known: {', '.join(sorted(canonical))}.")

        cats = policy.categories or []
        found = False
        for c in cats:
            if isinstance(c, dict) and c.get("key") == key:
                c["collected"] = collected
                found = True
        if not found:
            entry = dict(canonical[key])
            entry["collected"] = collected
            cats.append(entry)

        policy.categories = cats
        policy.save(update_fields=["categories"])
        self.stdout.write(self.style.SUCCESS(
            f"{'Enabled' if collected else 'Disabled'} '{key}' in active policy {policy.version}."
        ))
