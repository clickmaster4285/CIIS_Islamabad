"""Who may read object-tracking data, and which cameras they see.

Mirrors the web client's route rules for the "AI Monitoring & Analytics" pages
(frontend/src/lib/role-access.ts): Super Admin always; users with custom module grants only
when that module is granted; collectorate / operations officers without grants keep full
app access. Everyone except the Super Admin is limited to cameras at their own location.
"""

from __future__ import annotations

from django.db.models import Exists, OuterRef
from rest_framework.permissions import BasePermission

from users.permissions import SITE_FULL_ACCESS_ROLES, LOCATION_ADMIN_ROLE, get_location_scope, is_global_admin

MODULE_KEY = "AI Monitoring & Analytics"

# Site officers that see every module when no custom grants are set (Location Admin excluded,
# exactly as in the web client).
_DEFAULT_VIEWER_ROLES = SITE_FULL_ACCESS_ROLES - {LOCATION_ADMIN_ROLE}


def can_view_object_tracking(user) -> bool:
    if not user or not getattr(user, "is_authenticated", False):
        return False
    if is_global_admin(user) or getattr(user, "is_superuser", False):
        return True
    modules = [str(m).strip() for m in (getattr(user, "allowed_modules", None) or []) if str(m).strip()]
    if modules:
        return MODULE_KEY in modules
    return (getattr(user, "role", None) or "").strip().upper() in _DEFAULT_VIEWER_ROLES


class CanViewObjectTracking(BasePermission):
    message = "You do not have access to object tracking."

    def has_permission(self, request, view):
        return can_view_object_tracking(request.user)


def camera_location(user) -> str | None:
    """Location the user's camera view is limited to (None = all cameras)."""
    if getattr(user, "is_superuser", False):
        return None
    return get_location_scope(user)


def scope_by_camera(qs, user, field: str = "camera"):
    """Rows whose camera (via `field`) sits at the user's location."""
    loc = camera_location(user)
    if not loc:
        return qs
    return qs.filter(**{f"{field}__location__iexact": loc})


def scope_global_objects(qs, user):
    """Global objects seen (any visit) on a camera at the user's location."""
    from .models import ObjectVisit

    loc = camera_location(user)
    if not loc:
        return qs
    seen_here = ObjectVisit.objects.filter(global_object=OuterRef("pk"), camera__location__iexact=loc)
    return qs.filter(Exists(seen_here))
