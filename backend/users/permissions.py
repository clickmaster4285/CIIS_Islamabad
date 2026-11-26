from django.db.models import Q
from rest_framework import permissions

GLOBAL_ADMIN_ROLE = "ADMIN"
IT_SUPERADMIN_ROLE = "IT_SUPERADMIN"
LOCATION_ADMIN_ROLE = "LOCATION_ADMIN"

PRIVILEGED_ROLES = frozenset({GLOBAL_ADMIN_ROLE, LOCATION_ADMIN_ROLE, IT_SUPERADMIN_ROLE})


def is_global_admin(user) -> bool:
    return bool(
        user
        and getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) == GLOBAL_ADMIN_ROLE
    )


def is_it_superadmin(user) -> bool:
    return bool(
        user
        and getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) == IT_SUPERADMIN_ROLE
    )


def is_ops_viewer(user) -> bool:
    """Super Admin, IT Super Admin, location admin, or collectorate officers may view camera streams."""
    if not user or not getattr(user, "is_authenticated", False):
        return False
    if is_global_admin(user) or is_it_superadmin(user):
        return True
    role = getattr(user, "role", None)
    return role in {
        LOCATION_ADMIN_ROLE,
        "COLLECTOR",
        "DEPUTY_COLLECTOR",
        "ASSISTANT_COLLECTOR",
    }


# Islamabad is the head office and the only site in this deployment.
HEAD_OFFICE_LOCATION = "ISLAMABAD"
THIS_SITE_LOCATION = "ISLAMABAD"

_LOCATION_NAME_ALIASES = {
    "ISLAMABAD": ("Islamabad",),
}

_ALL_CITIES_VIEWER_ROLES = frozenset(
    {
        GLOBAL_ADMIN_ROLE,
        IT_SUPERADMIN_ROLE,
        LOCATION_ADMIN_ROLE,
        "COLLECTOR",
        "DEPUTY_COLLECTOR",
        "ASSISTANT_COLLECTOR",
    }
)


def _normalize_location_code(value) -> str:
    return (value or "").strip().upper().replace(" ", "_").replace("-", "_")


def can_see_all_cities_cameras(user) -> bool:
    """Head-office (Islamabad) Super Admin, IT Super Admin, admin and collectors see every city."""
    if not user or not getattr(user, "is_authenticated", False):
        return False
    role = getattr(user, "role", None)
    if role not in _ALL_CITIES_VIEWER_ROLES:
        return False
    loc = _normalize_location_code(getattr(user, "location", None))
    if loc == HEAD_OFFICE_LOCATION:
        return True
    if role in {GLOBAL_ADMIN_ROLE, IT_SUPERADMIN_ROLE} and THIS_SITE_LOCATION == HEAD_OFFICE_LOCATION and not loc:
        return True
    return False


def ops_camera_location_scope(user) -> str | None:
    """None = all connected cities; otherwise only this location's servers."""
    if can_see_all_cities_cameras(user):
        return None
    loc = (getattr(user, "location", None) or "").strip()
    return loc or THIS_SITE_LOCATION


def apply_remote_server_scope(queryset, user):
    """Restrict remote servers to the viewer's site unless they may see all cities."""
    scope = ops_camera_location_scope(user)
    if not scope:
        return queryset
    aliases = _LOCATION_NAME_ALIASES.get(scope.upper(), ())
    match = Q(location_code__iexact=scope)
    for alias in aliases:
        match |= Q(location_code__iexact=alias)
        match |= Q(name__icontains=alias)
    pretty = scope.replace("_", " ")
    if pretty.lower() != scope.lower():
        match |= Q(name__icontains=pretty)
    return queryset.filter(match)


def is_location_admin(user) -> bool:
    return bool(
        user
        and getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) == LOCATION_ADMIN_ROLE
    )


def is_admin_user(user) -> bool:
    return is_global_admin(user) or is_location_admin(user)


def get_location_scope(user) -> str | None:
    """
    Return the location code the user is restricted to, or None if they may see all sites.
    Global admins are never scoped; everyone else uses their assigned location when set.
    """
    if not user or not getattr(user, "is_authenticated", False):
        return None
    if is_global_admin(user):
        return None
    loc = (getattr(user, "location", None) or "").strip()
    return loc or None


def get_effective_location(user, query_param: str | None = None) -> str | None:
    """Location used for list filtering. Scoped users ignore query_param overrides."""
    scope = get_location_scope(user)
    if scope:
        return scope
    qp = (query_param or "").strip()
    return qp or None


def apply_location_filter(queryset, user, field: str = "location", query_param: str | None = None):
    loc = get_effective_location(user, query_param)
    if loc:
        return queryset.filter(**{field: loc})
    return queryset


def apply_source_location_filter(queryset, user, code: str | None):
    """
    Filter synced rows by the server they came from (Locations & Sync).
    Only unscoped users (Super Admin) choose; scoped users keep their existing filter.
    The HQ's own code also matches untagged local rows.
    """
    from city_sync.identity import local_rows_q, location_code

    code = (code or "").strip().upper()
    if not code or get_location_scope(user):
        return queryset
    if code == location_code():
        return queryset.filter(local_rows_q())
    return queryset.filter(location_code=code)


def resolve_location_for_write(user, requested_location: str = "") -> str:
    """On create/update: location-scoped users always write to their own site."""
    scope = get_location_scope(user)
    if scope:
        return scope
    return (requested_location or "").strip()


# Roles that carry cross-city visibility or camera-admin power. Only the Super Admin
# may hand these out (see _ALL_CITIES_VIEWER_ROLES and CAMERA_ADMIN_ROLES below).
_SUPER_ADMIN_ONLY_ROLES = frozenset(
    {
        GLOBAL_ADMIN_ROLE,
        IT_SUPERADMIN_ROLE,
        LOCATION_ADMIN_ROLE,
        "IT_ADMIN",
        "COLLECTOR",
        "DEPUTY_COLLECTOR",
        "ASSISTANT_COLLECTOR",
    }
)

# Site-wide staff overview (HR data, salaries, every employee's GPS) inside one location.
SITE_MANAGER_ROLES = frozenset({"HR", "OPERATION_MANAGER"})

# Explicit allow-list: the only roles HR / module-granted HR users may assign.
# Keep in sync with users.models.User.ROLE_CHOICES; a new role is Super-Admin-only
# until it is added here on purpose.
LOW_ASSIGNABLE_ROLES = frozenset(
    {
        "INSPECTOR",
        "RECEPTIONIST",
        "GUARD",
        "WAREHOUSE_OFFICER",
        "WAREHOUSE_SUPERINTENDENT",
        "WAREHOUSE_IN_CHARGE",
        "EXAMINATION_OFFICER",
        "STOCK_CONTROLLER",
        # SUPPORT / DEVELOPER run the Requests & Support desk for every site: Super Admin only.
        "AUDITOR",
        "PRAL",
        "DETECTION_OFFICER",
        "FIR_OFFICER",
        "INVESTIGATION_OFFICER",
        "SEIZING_OFFICER",
    }
)

# Location admins may additionally staff their own site's HR / operations desk.
LOCATION_ADMIN_ASSIGNABLE_ROLES = LOW_ASSIGNABLE_ROLES | SITE_MANAGER_ROLES

_LEVEL_SUPER = 100
_LEVEL_IT_SUPER = 90
_LEVEL_SITE_ADMIN = 80
_LEVEL_SITE_MANAGER = 50
_LEVEL_LOW = 10


def account_level(user) -> int:
    """Rough privilege rank used to decide who may edit whose account."""
    role = getattr(user, "role", None)
    if role == GLOBAL_ADMIN_ROLE:
        return _LEVEL_SUPER
    if role == IT_SUPERADMIN_ROLE:
        return _LEVEL_IT_SUPER
    if role in _SUPER_ADMIN_ONLY_ROLES:
        return _LEVEL_SITE_ADMIN
    if role in SITE_MANAGER_ROLES:
        return _LEVEL_SITE_MANAGER
    if role in LOW_ASSIGNABLE_ROLES:
        # A module grant of HR gives the same reach as the HR role.
        modules = getattr(user, "allowed_modules", None) or []
        return _LEVEL_SITE_MANAGER if HR_MODULE_KEY in modules else _LEVEL_LOW
    # Unknown / future role: treat as privileged so only the Super Admin touches it.
    return _LEVEL_SITE_ADMIN


def assignable_roles_for(actor) -> frozenset | None:
    """Roles the actor may hand out. None = every role (Super Admin)."""
    if is_global_admin(actor):
        return None
    if is_location_admin(actor):
        return LOCATION_ADMIN_ASSIGNABLE_ROLES
    return LOW_ASSIGNABLE_ROLES


def location_admin_may_assign_role(actor, role: str) -> bool:
    """Only the Super Admin may hand out admin / cross-city / camera-admin roles.

    Everyone else may assign only an explicit allow-list of low roles
    (location admins additionally HR / Operation Manager for their own site).
    """
    allowed = assignable_roles_for(actor)
    if allowed is None:
        return True
    return role in allowed


def may_manage_account(actor, target) -> bool:
    """May the actor edit / reset the password of / deactivate this account?

    Super Admin: anyone. Everyone else: their own account, or accounts at or below
    their own privilege level (and never an admin-role account).
    """
    if is_global_admin(actor):
        return True
    if getattr(target, "pk", None) is not None and target.pk == getattr(actor, "pk", None):
        return True
    if getattr(target, "role", None) in PRIVILEGED_ROLES or getattr(target, "role", None) == GLOBAL_ADMIN_ROLE:
        return False
    return account_level(target) <= account_level(actor)


def staff_location_q(scope: str, prefix: str = "") -> Q:
    """Staff rows belonging to one location: by the linked login's location, or for staff
    without a login by the server they belong to (location_code; untagged rows are this
    server's own) or their branch_location."""
    from django.conf import settings

    code = (scope or "").strip()
    here = str(getattr(settings, "LOCATION_CODE", "") or "").strip().upper()
    unlinked = Q(**{f"{prefix}location_code__iexact": code}) | Q(**{f"{prefix}branch_location__iexact": code})
    if here and here == code.upper():
        unlinked |= Q(**{f"{prefix}location_code": ""})
    return Q(**{f"{prefix}user__location": code}) | (Q(**{f"{prefix}user__isnull": True}) & unlinked)


def staff_in_scope(actor, staff) -> bool:
    """True when a location-scoped actor may touch this staff row (always for unscoped)."""
    scope = get_location_scope(actor)
    if not scope or staff is None:
        return True
    from users.models import Staff

    return Staff.objects.filter(staff_location_q(scope), pk=staff.pk).exists()


HR_MODULE_KEY = "Human Resource"

# Matches frontend site-full-access roles that already show Attendance / HR in the sidebar.
SITE_FULL_ACCESS_ROLES = frozenset(
    {
        LOCATION_ADMIN_ROLE,
        "OPERATION_MANAGER",
        "COLLECTOR",
        "DEPUTY_COLLECTOR",
        "ASSISTANT_COLLECTOR",
    }
)

HR_API_ROLES = frozenset(
    {GLOBAL_ADMIN_ROLE, "HR", "IT_ADMIN"} | SITE_FULL_ACCESS_ROLES
)

# PWA: these roles may see every employee’s location, attendance, and mobile logs.
STAFF_OVERVIEW_ROLES = frozenset(
    {
        GLOBAL_ADMIN_ROLE,
        IT_SUPERADMIN_ROLE,
        LOCATION_ADMIN_ROLE,
        "HR",
        "IT_ADMIN",
        "OPERATION_MANAGER",
        "COLLECTOR",
        "DEPUTY_COLLECTOR",
        "ASSISTANT_COLLECTOR",
    }
)


def can_view_all_staff(user) -> bool:
    """True when the user may view other employees’ GPS, attendance, and logs."""
    if not user or not getattr(user, "is_authenticated", False):
        return False
    if getattr(user, "role", None) in STAFF_OVERVIEW_ROLES:
        return True
    modules = getattr(user, "allowed_modules", None) or []
    return HR_MODULE_KEY in modules


def has_hr_api_access(user) -> bool:
    """True when the user may call staff / attendance / leave / recognition APIs."""
    if not user or not getattr(user, "is_authenticated", False):
        return False
    if getattr(user, "role", None) in HR_API_ROLES:
        return True
    modules = getattr(user, "allowed_modules", None) or []
    return HR_MODULE_KEY in modules


class IsAdminOrHR(permissions.BasePermission):
    """Allow HR APIs to the same people who can open Attendance in the UI."""

    allowed_roles = tuple(HR_API_ROLES)

    def has_permission(self, request, view):
        return has_hr_api_access(request.user)


class IsGlobalAdmin(permissions.BasePermission):
    """Allow access only to the global super administrator."""

    def has_permission(self, request, view):
        return is_global_admin(request.user)


class IsAdminUser(permissions.BasePermission):
    """Allow global or location administrators."""

    def has_permission(self, request, view):
        return is_admin_user(request.user)


CAMERA_ADMIN_ROLES = frozenset({GLOBAL_ADMIN_ROLE, IT_SUPERADMIN_ROLE, LOCATION_ADMIN_ROLE, "IT_ADMIN"})


class ReadOnlyUnlessCameraAdmin(permissions.BasePermission):
    """Any logged-in user may view sites/NVRs/cameras; only admin roles may change them."""

    def has_permission(self, request, view):
        user = request.user
        if not user or not user.is_authenticated:
            return False
        if request.method in permissions.SAFE_METHODS:
            return True
        return getattr(user, "role", None) in CAMERA_ADMIN_ROLES
