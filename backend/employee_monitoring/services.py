"""Helpers shared by the monitoring views: masking, audit, scoping, storage keys."""

from __future__ import annotations

import os
import re
from datetime import timedelta

from django.conf import settings
from django.core.signing import BadSignature, SignatureExpired, TimestampSigner
from django.utils import timezone

from users.permissions import can_view_all_staff, get_location_scope

from .models import MonitoringAudit

_SIGNER_SALT = "employee_monitoring.recording.playback"

# Reserved so only the signed-in handset can create these rows. PBX ingest cannot use it.
ANDROID_CALL_SOURCE = "android_call_log"
HANDSET_CATEGORY = "android_call_history"

# Audio recordings the phone's own dialer made, uploaded by the signed-in handset (no PBX).
# A separate, more invasive opt-in than call-history metadata.
ANDROID_CALL_RECORDING_SOURCE = "android_call_recording"
CALL_RECORDING_CATEGORY = "android_call_recording"

# Sources/categories for the handset usage + screen features (signed-in device only).
APP_USAGE_SOURCE = "android_app_usage"
APP_USAGE_CATEGORY = "app_usage"
SCREEN_SOURCE = "mobile_screen"
IN_APP_SCREEN_CATEGORY = "in_app_screens"
SCREEN_SHARE_CATEGORY = "screen_share"


def feature_enabled() -> bool:
    return str(getattr(settings, "EMPLOYEE_MONITORING_ENABLED", "")).lower() in ("true", "1", "yes")


def deployment_id() -> str:
    return getattr(settings, "TEKEYE_DEPLOYMENT_ID", "") or "unknown"


def retention_days() -> int:
    try:
        return int(getattr(settings, "MONITORING_RETENTION_DAYS", 0) or 0)
    except (TypeError, ValueError):
        return 0


def policy_allows_category(policy, key: str) -> bool:
    """True only when the published policy explicitly collects the given category."""
    if not policy:
        return False
    for category in policy.categories or []:
        if not isinstance(category, dict):
            continue
        if category.get("key") == key and category.get("collected", True):
            return True
    return False


def policy_allows_handset_calls(policy) -> bool:
    """True only when the published policy explicitly collects Android call history."""
    return policy_allows_category(policy, HANDSET_CATEGORY)


def latest_consent(user):
    """The current consent row for a user (latest wins), or None."""
    from .models import MonitoringConsent

    return MonitoringConsent.objects.filter(user=user).select_related("policy").first()


def category_consented(consent, policy, key: str) -> bool:
    """Per-category opt-in gate used for BYOD features.

    True only when the user's latest consent is an acknowledgment of THIS active policy and
    that acknowledgment explicitly disclosed the given category. Withdrawing, acknowledging a
    different policy, or acknowledging without this category in ``disclosed_categories`` all
    revoke collection for the category independently of the others.
    """
    from .models import MonitoringConsent

    if not consent or not policy:
        return False
    if consent.status != MonitoringConsent.ACKNOWLEDGED or consent.policy_id != policy.id:
        return False
    return key in (consent.disclosed_categories or [])


def normalize_number(raw: str | None) -> str:
    """Keep a leading plus and digits. Drop contact names and other non-number text."""
    if not raw:
        return ""
    text = raw.strip()
    plus = text.startswith("+")
    digits = re.sub(r"\D", "", text)[:20]
    if not digits:
        return ""
    return f"+{digits}" if plus else digits


def mask_number(raw: str | None) -> str:
    """+923001234567 -> "+92 3** *** **67". Keeps country prefix and last two digits only."""
    if not raw:
        return ""
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 4:
        return "***"
    plus = "+" if raw.strip().startswith("+") else ""
    head = digits[:2]
    tail = digits[-2:]
    middle = "*" * max(0, len(digits) - 4)
    return f"{plus}{head}{middle}{tail}"


def client_ip(request) -> str | None:
    fwd = request.META.get("HTTP_X_FORWARDED_FOR", "")
    if fwd:
        return fwd.split(",")[0].strip() or None
    return request.META.get("REMOTE_ADDR") or None


def record_audit(request, action: str, *, target_type: str = "", target_id: str = "", reason: str = "", **metadata) -> None:
    MonitoringAudit.objects.create(
        actor=getattr(request, "user", None) if getattr(request, "user", None) and request.user.is_authenticated else None,
        action=action,
        target_type=target_type,
        target_id=str(target_id),
        reason=reason,
        request_ip=client_ip(request),
        metadata=metadata or {},
    )


def scope_queryset(qs, user, *, own_field: str = "user", location_field: str = "user__location"):
    """Own records for ordinary staff; location-scoped (or all) records for overview roles."""
    if not can_view_all_staff(user):
        return qs.filter(**{own_field: user})
    loc = get_location_scope(user)
    if loc:
        return qs.filter(**{location_field: loc})
    return qs


# --- recording storage (private, local by default) -----------------------------------------

def recording_root() -> str:
    root = getattr(settings, "MONITORING_RECORDING_ROOT", "") or os.path.join(settings.BASE_DIR, "private_media", "monitoring")
    os.makedirs(root, exist_ok=True)
    return root


def storage_path(storage_key: str) -> str:
    # storage_key is a server-chosen relative name; never a client path.
    safe = os.path.normpath(storage_key).lstrip(os.sep).replace("..", "")
    return os.path.join(recording_root(), safe)


def sign_playback(recording_id: str) -> str:
    return TimestampSigner(salt=_SIGNER_SALT).sign(str(recording_id))


def unsign_playback(token: str, max_age_seconds: int = 300) -> str | None:
    try:
        return TimestampSigner(salt=_SIGNER_SALT).unsign(token, max_age=max_age_seconds)
    except (BadSignature, SignatureExpired):
        return None


def playback_expiry(seconds: int = 300):
    return timezone.now() + timedelta(seconds=seconds)
