"""Who may change Visitor Management (VMS) data.

Mirrors the frontend: Guards and Receptionists run the VMS screens, collectorate officers and
location admins have full site access, and any user granted the "Visitor Management" sidebar
module may also use it. Everyone else can still read (lists stay location-scoped).
"""

from __future__ import annotations

from django.db.models import Q
from rest_framework import permissions

from users.permissions import GLOBAL_ADMIN_ROLE, SITE_FULL_ACCESS_ROLES


def site_rows_q(location: str | None, field: str = "location_code") -> Q:
    """Rows that originate from one site, using the city-sync origin code.

    Rows created on this server carry this server's LOCATION_CODE (or are untagged), so when the
    requested site is this server they match too. Empty location = no restriction.
    """
    from city_sync.identity import is_city, local_rows_q, location_code

    loc = (location or "").strip()
    if not loc:
        return Q()
    q = Q(**{f"{field}__iexact": loc})
    here = location_code()
    # A city server only holds its own city's rows; on HQ compare with HQ's own code.
    if is_city() or not here or here == loc.upper():
        prefix = field[: -len("location_code")] if field.endswith("location_code") else ""
        q |= local_rows_q(prefix)
    return q

VMS_MODULE_KEY = "Visitor Management"

# Roles that operate the VMS (registration, approvals, gate scans, lists, alerts).
VMS_WRITE_ROLES = frozenset({GLOBAL_ADMIN_ROLE, "RECEPTIONIST", "GUARD"} | SITE_FULL_ACCESS_ROLES)


def can_write_vms(user) -> bool:
    if not user or not getattr(user, "is_authenticated", False):
        return False
    if getattr(user, "is_superuser", False):
        return True
    if getattr(user, "role", None) in VMS_WRITE_ROLES:
        return True
    modules = getattr(user, "allowed_modules", None) or []
    return VMS_MODULE_KEY in modules


class CanWriteVms(permissions.BasePermission):
    """Any write (or any request, for write-only endpoints) needs a VMS role or module grant."""

    message = "Your role is not allowed to change visitor management records."

    def has_permission(self, request, view):
        return can_write_vms(request.user)


class ReadAnyWriteVms(permissions.BasePermission):
    """Logged-in users may read; only VMS operators may write."""

    message = CanWriteVms.message

    def has_permission(self, request, view):
        user = request.user
        if not user or not user.is_authenticated:
            return False
        if request.method in permissions.SAFE_METHODS:
            return True
        return can_write_vms(user)
