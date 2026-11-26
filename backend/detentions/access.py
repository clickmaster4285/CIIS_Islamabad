"""Location scope, HQ read-only rule and role checks for detention / seizure documents.

Shared by ``detentions`` and ``seizure_management``.

* Reads: a user sees rows of their own location; the global Super Admin sees every
  location (including the city copies HQ received through sync). A non-admin user
  with no location set sees only this server's own rows.
* Writes: only rows this server owns (``local_rows_q``) may be changed. On HQ the
  synced copies of a city's documents are read-only (403).
* Roles: writes need an operational role (or a "Seizure Management" / "Warehouse
  Management" module grant); deletes need a higher official.
"""

from __future__ import annotations

import re

from django.db.models import Q
from rest_framework import permissions
from rest_framework.response import Response

from users.permissions import GLOBAL_ADMIN_ROLE, LOCATION_ADMIN_ROLE, get_location_scope, is_global_admin

# Same location code may appear in its long or short form (city_sync allows both).
_LOCATION_ALIASES = {
    "ISLAMABAD": {"ISB"},
    # "PESHAWAR": {"PEW"},
    "ABBOTTABAD": {"ABT"},
    "DI_KHAN": {"DIK"},
    "MARDAN": {"MRD"},
    "NOWSHERA": {"NOW"},
    "KOHAT": {"KHT"},
}

# Super Admin + Location Admin may edit documents in any status (matches the
# frontend FULL_EDIT_ROLES in lib/seizure-management-api.ts).
FULL_EDIT_ROLES = frozenset({GLOBAL_ADMIN_ROLE, LOCATION_ADMIN_ROLE})

HIGHER_OFFICIAL_ROLES = frozenset(
    {
        GLOBAL_ADMIN_ROLE,
        LOCATION_ADMIN_ROLE,
        "COLLECTOR",
        "DEPUTY_COLLECTOR",
        "ASSISTANT_COLLECTOR",
    }
)

# Roles that create / edit detention + seizure documents.
DOCUMENT_WRITE_ROLES = HIGHER_OFFICIAL_ROLES | frozenset(
    {
        "OPERATION_MANAGER",
        "INSPECTOR",
        "WAREHOUSE_OFFICER",
        "WAREHOUSE_SUPERINTENDENT",
        "EXAMINATION_OFFICER",
        "DETECTION_OFFICER",
        "FIR_OFFICER",
        "INVESTIGATION_OFFICER",
        "SEIZING_OFFICER",
    }
)

# Deposit account register is also kept by the warehouse roles.
DEPOSIT_WRITE_ROLES = DOCUMENT_WRITE_ROLES | frozenset({"WAREHOUSE_IN_CHARGE", "STOCK_CONTROLLER"})

# Detention memo edit page is only offered to these roles in the UI.
MEMO_EDIT_ROLES = HIGHER_OFFICIAL_ROLES

DOCUMENT_DELETE_ROLES = HIGHER_OFFICIAL_ROLES
DEPOSIT_DELETE_ROLES = HIGHER_OFFICIAL_ROLES | frozenset({"WAREHOUSE_SUPERINTENDENT"})

# Never write, whatever modules they were granted.
READ_ONLY_ROLES = frozenset({"AUDITOR", "PRAL"})

SEIZURE_MODULE = "Seizure Management"
WAREHOUSE_MODULE = "Warehouse Management"


def normalize_role(role: str | None) -> str:
    return re.sub(r"[\s-]+", "_", (role or "").strip()).upper()


def user_role(user) -> str:
    return normalize_role(getattr(user, "role", None))


def _authenticated(user) -> bool:
    return bool(user and getattr(user, "is_authenticated", False))


def _has_module(user, *modules: str) -> bool:
    granted = getattr(user, "allowed_modules", None) or []
    if not isinstance(granted, (list, tuple)):
        return False
    names = {str(m).strip() for m in granted}
    return any(m in names for m in modules)


def can_write_documents(user, *, deposit: bool = False) -> bool:
    if not _authenticated(user):
        return False
    role = user_role(user)
    if role in READ_ONLY_ROLES:
        return False
    allowed = DEPOSIT_WRITE_ROLES if deposit else DOCUMENT_WRITE_ROLES
    if role in allowed:
        return True
    modules = (SEIZURE_MODULE, WAREHOUSE_MODULE) if deposit else (SEIZURE_MODULE,)
    return _has_module(user, *modules)


def can_delete_documents(user, *, deposit: bool = False) -> bool:
    if not _authenticated(user):
        return False
    return user_role(user) in (DEPOSIT_DELETE_ROLES if deposit else DOCUMENT_DELETE_ROLES)


def can_full_edit(user) -> bool:
    return _authenticated(user) and user_role(user) in FULL_EDIT_ROLES


def can_edit_memo(user) -> bool:
    return _authenticated(user) and user_role(user) in MEMO_EDIT_ROLES


class DocumentWritePermission(permissions.BasePermission):
    """Any logged-in user may read; writes need a document-writer role or module grant."""

    message = "Your role cannot create or change detention / seizure documents."

    def has_permission(self, request, view):
        user = request.user
        if not _authenticated(user):
            return False
        if request.method in permissions.SAFE_METHODS:
            return True
        return can_write_documents(user)


class DepositWritePermission(DocumentWritePermission):
    message = "Your role cannot create or change deposit account entries."

    def has_permission(self, request, view):
        user = request.user
        if not _authenticated(user):
            return False
        if request.method in permissions.SAFE_METHODS:
            return True
        return can_write_documents(user, deposit=True)


class DocumentDeletePermission(permissions.BasePermission):
    message = "Only Super Admin, Location Admin or Collector officers can delete this record."

    def has_permission(self, request, view):
        return can_delete_documents(request.user)


class DepositDeletePermission(permissions.BasePermission):
    message = "Only higher officials can delete deposit account entries."

    def has_permission(self, request, view):
        return can_delete_documents(request.user, deposit=True)


class MemoEditPermission(permissions.BasePermission):
    message = "Only Super Admin, Location Admin or Collector officers can edit a detention memo."

    def has_permission(self, request, view):
        return can_edit_memo(request.user)


# --------------------------------------------------------------------------- scope


def _site_code() -> str:
    from city_sync.identity import location_code

    return location_code()


def _codes_for(code: str) -> set[str]:
    code = (code or "").strip().upper().replace(" ", "_").replace("-", "_")
    if not code:
        return set()
    out = {code}
    out |= _LOCATION_ALIASES.get(code, set())
    for long_code, shorts in _LOCATION_ALIASES.items():
        if code in shorts:
            out.add(long_code)
            out |= shorts
    return out


def location_codes_for(code: str) -> set[str]:
    """``code`` plus its long/short aliases (e.g. DI_KHAN <-> DIK), upper-case."""
    return _codes_for(code)


def location_scope_q(user, prefix: str = "") -> Q:
    """Q for rows ``user`` may read. Empty Q = everything (global admin)."""
    from city_sync.identity import local_rows_q

    if is_global_admin(user):
        return Q()
    scope = get_location_scope(user)
    if not scope:
        # No location assigned: only this server's own rows.
        return local_rows_q(prefix)
    codes = _codes_for(scope)
    field = f"{prefix}location_code"
    q = Q(**{f"{field}__in": sorted(codes)})
    if _site_code() in codes:
        q |= local_rows_q(prefix)
    return q


def scoped(queryset, user, prefix: str = ""):
    return queryset.filter(location_scope_q(user, prefix))


def is_local_row(obj) -> bool:
    code = (getattr(obj, "location_code", "") or "").strip().upper()
    return code == "" or code == _site_code()


def readonly_copy_response(obj) -> Response | None:
    """403 when ``obj`` is a city copy held by HQ (owned by another server)."""
    if is_local_row(obj):
        return None
    return Response(
        {
            "detail": (
                "This record belongs to another location and is read-only here. "
                "Change it on that location's server."
            )
        },
        status=403,
    )
