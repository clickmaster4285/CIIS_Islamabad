"""Who may see which staff / face data: this server's own rows, narrowed to the viewer's site."""

from __future__ import annotations

from django.db.models import Q

from city_sync.identity import local_rows_q
from users.permissions import get_location_scope


def staff_scope_q(user, prefix: str = "", *, local_only: bool = True) -> Q:
    """
    Q for Staff rows (or a relation to Staff via ``prefix``, e.g. "staff__") the user may work with.

    local_only: this server's own staff (HQ copies of city staff are recognised on the city, not here).
    Location-scoped users only see staff whose login is at their site; staff without a login cannot be
    placed, so they stay visible to the site that owns the row.
    """
    q = local_rows_q(prefix) if local_only else Q()
    q &= Q(**{f"{prefix}is_deleted": False})
    scope = get_location_scope(user)
    if scope:
        q &= Q(**{f"{prefix}user__location": scope}) | Q(**{f"{prefix}user__isnull": True})
    return q


def scoped_staff(user, *, local_only: bool = True):
    from users.models import Staff

    return Staff.objects.filter(staff_scope_q(user, local_only=local_only))


def face_sample_scope_q(user) -> Q:
    """Samples whose confirmed (or, if none, suggested) person the user may see; unassigned crops stay visible."""
    unassigned = Q(staff__isnull=True) & (Q(suggested_staff__isnull=True) | staff_scope_q(user, "suggested_staff__"))
    return unassigned | (Q(staff__isnull=False) & staff_scope_q(user, "staff__"))
