"""Warehouse / destruction permissions and location scoping."""

from __future__ import annotations

from django.db.models import Q
from rest_framework import permissions

from users.permissions import GLOBAL_ADMIN_ROLE, LOCATION_ADMIN_ROLE, SITE_FULL_ACCESS_ROLES, get_location_scope

WAREHOUSE_MODULE_KEY = "Warehouse Management"

# Who is alerted about (and may acknowledge) smoke/fire during destruction.
DESTRUCTION_ALERT_ROLES = frozenset(
    {
        GLOBAL_ADMIN_ROLE,
        LOCATION_ADMIN_ROLE,
        "WAREHOUSE_OFFICER",
        "WAREHOUSE_IN_CHARGE",
        "WAREHOUSE_SUPERINTENDENT",
    }
)

# Who may record destructions and push stock changes (matches the frontend warehouse screens).
WAREHOUSE_WRITE_ROLES = frozenset(
    {
        GLOBAL_ADMIN_ROLE,
        "WAREHOUSE_OFFICER",
        "WAREHOUSE_IN_CHARGE",
        "WAREHOUSE_SUPERINTENDENT",
        "STOCK_CONTROLLER",
    }
    | SITE_FULL_ACCESS_ROLES
)


def can_write_warehouse(user) -> bool:
    if not user or not getattr(user, "is_authenticated", False):
        return False
    if getattr(user, "is_superuser", False):
        return True
    if getattr(user, "role", None) in WAREHOUSE_WRITE_ROLES:
        return True
    modules = getattr(user, "allowed_modules", None) or []
    return WAREHOUSE_MODULE_KEY in modules


class CanWriteWarehouse(permissions.BasePermission):
    message = "Your role is not allowed to change warehouse stock or destruction records."

    def has_permission(self, request, view):
        return can_write_warehouse(request.user)


def _scope_is_this_server(scope: str) -> bool:
    """A city server only holds its own city's rows; on HQ compare with HQ's LOCATION_CODE."""
    from city_sync.identity import is_city, location_code

    here = location_code()
    return is_city() or not here or here == scope.strip().upper()


def site_rows_q(user, field: str = "location_code") -> Q:
    """Restrict city-sync rows (origin code in `field`) to the user's site. Q() for unscoped users."""
    from city_sync.identity import local_rows_q

    scope = (get_location_scope(user) or "").strip()
    if not scope:
        return Q()
    q = Q(**{f"{field}__iexact": scope})
    if _scope_is_this_server(scope):
        q |= local_rows_q(field[: -len("location_code")] if field.endswith("location_code") else "")
    return q


def memo_in_scope(user, memo) -> bool:
    """Destructions are recorded only on the server that owns the memo, by users of that site."""
    from city_sync.identity import location_code

    origin = (getattr(memo, "location_code", "") or "").strip().upper()
    here = location_code()
    if origin and here and origin != here:
        # HQ copy of another server's memo: that server owns the goods and the destruction.
        return False
    scope = (get_location_scope(user) or "").strip()
    return not scope or _scope_is_this_server(scope)


def cameras_in_scope(user, camera_ids) -> bool:
    from cameras.models import Camera
    from voice_agent.policy import camera_scope_q

    ids = {int(c) for c in camera_ids}
    if not ids:
        return True
    if not get_location_scope(user):
        return True
    allowed = set(
        Camera.objects.filter(camera_scope_q(user), pk__in=ids).values_list("pk", flat=True).distinct()
    )
    return ids <= allowed


def session_in_scope(user, session) -> bool:
    """Distribution session belongs to the user's site (by its resolved location or its cameras)."""
    scope = (get_location_scope(user) or "").strip()
    if not scope:
        return True
    loc = (session.location_code or "").strip()
    if loc:
        return loc.upper() == scope.upper()
    return cameras_in_scope(user, session.camera_ids or ([session.camera_id] if session.camera_id else []))


def session_scope_q(user) -> Q:
    scope = (get_location_scope(user) or "").strip()
    if not scope:
        return Q()
    return Q(location_code__iexact=scope) | Q(location_code="", performed_by=getattr(user, "username", "") or "")
