from rest_framework import permissions

from users.permissions import is_global_admin, is_it_superadmin


class IsInfraAdmin(permissions.BasePermission):
    """IT Super Admin and Super Admin may manage infrastructure monitoring."""

    def has_permission(self, request, view):
        user = request.user
        if not user or not user.is_authenticated:
            return False
        return is_it_superadmin(user) or is_global_admin(user)


def may_run_destructive_nvr_command(user) -> bool:
    """
    Reboot / format / factory reset / manual recording start-stop on an NVR.

    Explicit role check only (Super Admin or IT Super Admin) — no is_staff or
    free-text role fallbacks.
    """
    return bool(
        user
        and getattr(user, "is_authenticated", False)
        and (is_global_admin(user) or is_it_superadmin(user))
    )
