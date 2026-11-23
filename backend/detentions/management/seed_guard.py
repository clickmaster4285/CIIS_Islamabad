"""Guard for sample-data seed commands: never run against production by accident."""

from __future__ import annotations

from django.conf import settings
from django.core.management.base import CommandError


def add_guard_argument(parser) -> None:
    parser.add_argument(
        "--allow-production",
        action="store_true",
        help="Allow seeding when DEBUG is off (writes sample records into a live database).",
    )


def check_seed_allowed(options) -> None:
    if getattr(settings, "DEBUG", False) or options.get("allow_production"):
        return
    raise CommandError(
        "Refusing to seed sample data: DEBUG is off. "
        "Re-run with --allow-production if you really mean to write sample records here."
    )
