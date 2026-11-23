"""Who may see / change Person Journey data."""

from __future__ import annotations

from django.db.models import Q
from rest_framework import permissions

from users.permissions import (
    PRIVILEGED_ROLES,
    SITE_FULL_ACCESS_ROLES,
    get_location_scope,
    is_ops_viewer,
)


def can_view_journeys(user) -> bool:
    """Camera-stream viewers (admins, IT super admin, collectors) plus site full-access roles."""
    if is_ops_viewer(user):
        return True
    return bool(
        user
        and getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in SITE_FULL_ACCESS_ROLES
    )


def can_edit_journeys(user) -> bool:
    """Renaming a person or linking them to a visitor record: admin roles only."""
    return bool(
        user
        and getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in PRIVILEGED_ROLES
    )


class CanViewPersonJourney(permissions.BasePermission):
    message = "Person Journey is limited to camera operations roles."

    def has_permission(self, request, view):
        return can_view_journeys(request.user)


class CanEditPersonJourney(permissions.BasePermission):
    message = "Only administrators may change journey identities."

    def has_permission(self, request, view):
        return can_edit_journeys(request.user)


def person_scope_q(user, prefix: str = "") -> Q:
    """
    JourneyPerson rows (or a relation to one via ``prefix``) inside the user's site.

    A person belongs to a site through the server that recorded them (location_code; this server's own rows
    are untagged or carry its LOCATION_CODE) or through the camera that last saw them.
    """
    scope = get_location_scope(user)
    if not scope:
        return Q()
    from city_sync.identity import local_rows_q, location_code

    q = Q(**{f"{prefix}location_code__iexact": scope}) | Q(**{f"{prefix}latest_camera__location__iexact": scope})
    if scope.strip().upper() == location_code():
        q |= local_rows_q(prefix)
    return q
