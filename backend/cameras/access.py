"""Location scoping and role checks for camera / NVR / detection endpoints."""

from __future__ import annotations

import ipaddress
import re
import threading
from collections import Counter

from django.conf import settings
from django.db.models import Q

from users.permissions import (
    CAMERA_ADMIN_ROLES,
    can_see_all_cities_cameras,
    get_location_scope,
    is_global_admin,
)


def camera_location_scope(user) -> str | None:
    """None = every site; otherwise the site code this user is restricted to.

    Global admins (and the HQ roles that already see every city in Ops) are never scoped;
    everyone else with an assigned location only sees that site.
    """
    if user is None or not getattr(user, "is_authenticated", False):
        return None
    if is_global_admin(user) or can_see_all_cities_cameras(user):
        return None
    return get_location_scope(user)


def _pretty(scope: str) -> str:
    return scope.replace("_", " ")


def site_scope_q(scope: str | None, prefix: str = "") -> Q:
    if not scope:
        return Q()
    return Q(**{f"{prefix}code__iexact": scope}) | Q(**{f"{prefix}name__iexact": _pretty(scope)})


def camera_scope_q(scope: str | None, prefix: str = "") -> Q:
    if not scope:
        return Q()
    return Q(**{f"{prefix}location__iexact": scope}) | site_scope_q(scope, f"{prefix}nvr__site__")


def detection_scope_q(scope: str | None) -> Q:
    """Local events by their camera's site; HQ copies of city events by location_code."""
    if not scope:
        return Q()
    return (Q(camera__isnull=False) & camera_scope_q(scope, "camera__")) | Q(
        camera__isnull=True, location_code__iexact=scope
    )


def site_in_scope(site, scope: str | None) -> bool:
    if not scope:
        return True
    if site is None:
        return False
    wanted = scope.strip().lower()
    return (site.code or "").strip().lower() == wanted or (site.name or "").strip().lower() == _pretty(
        scope
    ).lower()


def is_camera_admin(user) -> bool:
    return bool(
        user
        and getattr(user, "is_authenticated", False)
        and getattr(user, "role", None) in CAMERA_ADMIN_ROLES
    )


# ——— NVR address validation ———

_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*\.?$"
)


def validate_nvr_host(value: str) -> str:
    """Return a cleaned IP / hostname or raise ValueError (no scheme, path, @, port or spaces)."""
    text = (value or "").strip()
    if not text:
        raise ValueError("IP address is required.")
    if any(ch in text for ch in ("@", "/", "\\", "?", "#", " ", "\t")) or "://" in text:
        raise ValueError("Enter only an IP address or hostname (no scheme, path, credentials or spaces).")
    candidate = text[1:-1] if text.startswith("[") and text.endswith("]") else text
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        pass
    if ":" in text:
        raise ValueError("Enter the port in the Port field, not in the IP address.")
    if not _HOSTNAME_RE.match(text):
        raise ValueError("Enter a valid IP address or hostname.")
    return text.lower()


# ——— Concurrent RTSP preview slots (each preview is one ffmpeg process) ———

_preview_guard = threading.Lock()
_preview_total = 0
_preview_by_user: Counter = Counter()


def preview_limits() -> tuple[int, int]:
    total = int(getattr(settings, "CAMERA_PREVIEW_MAX_CONCURRENT", 8) or 8)
    per_user = int(getattr(settings, "CAMERA_PREVIEW_MAX_PER_USER", 4) or 4)
    return max(1, total), max(1, per_user)


def acquire_preview_slot(user_id) -> str | None:
    """Reserve a preview slot. Returns None on success, or a reason string when over the cap."""
    global _preview_total
    total_cap, user_cap = preview_limits()
    with _preview_guard:
        if _preview_total >= total_cap:
            return f"Too many live previews running on the server (limit {total_cap}). Close one and retry."
        if _preview_by_user[user_id] >= user_cap:
            return f"You already have {user_cap} live previews open. Close one and retry."
        _preview_total += 1
        _preview_by_user[user_id] += 1
    return None


def release_preview_slot(user_id) -> None:
    global _preview_total
    with _preview_guard:
        _preview_total = max(0, _preview_total - 1)
        _preview_by_user[user_id] -= 1
        if _preview_by_user[user_id] <= 0:
            del _preview_by_user[user_id]
