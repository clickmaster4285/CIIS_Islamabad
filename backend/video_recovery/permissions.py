"""Who may use video recovery, and whose jobs they see.

The page sits in the "AI Monitoring & Analytics" module, so access follows the same rule as
the object-tracking pages. Uploaded footage is evidence: ordinary users only see their own
jobs; the Super Admin and investigation officers see every job.
"""

from __future__ import annotations

from rest_framework.permissions import BasePermission

from object_tracking.permissions import can_view_object_tracking
from users.permissions import is_global_admin

FORENSIC_ROLES = frozenset({"INVESTIGATION_OFFICER"})


def can_use_video_recovery(user) -> bool:
    return can_view_object_tracking(user)


def can_see_all_jobs(user) -> bool:
    if not user or not getattr(user, "is_authenticated", False):
        return False
    if getattr(user, "is_superuser", False) or is_global_admin(user):
        return True
    return (getattr(user, "role", None) or "").strip().upper() in FORENSIC_ROLES


def scope_jobs(qs, user):
    return qs if can_see_all_jobs(user) else qs.filter(uploaded_by=user)


class CanUseVideoRecovery(BasePermission):
    message = "You do not have access to video recovery."

    def has_permission(self, request, view):
        return can_use_video_recovery(request.user)
