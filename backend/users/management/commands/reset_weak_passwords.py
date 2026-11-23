"""
Find active accounts that still use a known default/weak password and (optionally) reset them.

    python manage.py reset_weak_passwords            # list only (no changes)
    python manage.py reset_weak_passwords --reset    # give each a random password, printed ONCE

Staff logins used to be created with the fixed password "123456". Run this on every server
(HQ and each city), hand the new passwords to their owners, and keep the output private.
"""

from __future__ import annotations

import secrets
import string

from django.core.management.base import BaseCommand

WEAK_PASSWORDS = (
    "123456",
    "1234567",
    "12345678",
    "123456789",
    "password",
    "admin",
    "admin123",
    "pass1234",
    "000000",
)


def _random_password(length: int = 14) -> str:
    alphabet = string.ascii_letters + string.digits
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        if any(c.isdigit() for c in pwd) and any(c.isalpha() for c in pwd):
            return pwd


class Command(BaseCommand):
    help = "List (or with --reset, randomise) active accounts that use a known weak/default password."

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true", help="Set a random password on each weak account.")

    def handle(self, *args, **options):
        from users.models import User

        weak = []
        for user in User.objects.filter(is_active=True).order_by("username"):
            if not user.has_usable_password():
                continue
            if any(user.check_password(p) for p in WEAK_PASSWORDS):
                weak.append(user)

        if not weak:
            self.stdout.write(self.style.SUCCESS("No active account uses a known weak password."))
            return

        self.stdout.write(f"{len(weak)} active account(s) use a weak/default password:")
        for user in weak:
            if options["reset"]:
                new = _random_password()
                user.set_password(new)
                user.save(update_fields=["password"])
                # Existing API tokens keep working until they expire; revoke them so the
                # old password can't be used to keep a session alive.
                try:
                    from rest_framework.authtoken.models import Token

                    Token.objects.filter(user=user).delete()
                except Exception:
                    pass
                self.stdout.write(f"  {user.username:<30} {user.role or '-':<20} new password: {new}")
            else:
                self.stdout.write(f"  {user.username:<30} {user.role or '-'}")
        if not options["reset"]:
            self.stdout.write("Run again with --reset to give each a random password (printed once).")
