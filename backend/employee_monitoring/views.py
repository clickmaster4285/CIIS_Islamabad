"""Employee-monitoring REST API.

Mounted at /api/v1/employee-monitor/. Every endpoint requires authentication (the project's
ExpiringTokenAuthentication) and returns 501 when the deployment has monitoring disabled, so the
mobile client can show an honest "not set up" state instead of guessing.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.db.models import Sum
from django.http import Http404
from django.utils.dateparse import parse_date
from rest_framework import permissions, status
from rest_framework.exceptions import APIException, ValidationError
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from users.permissions import can_view_all_staff

from .models import (
    EmployeeAppSession,
    EmployeeAppUsage,
    EmployeeCallRecord,
    EmployeeCallRecording,
    EmployeeScreenCapture,
    MobileVisibility,
    MonitoredAppIcon,
    MonitoringConsent,
    MonitoringPolicy,
)
from .serializers import (
    AppIconSyncSerializer,
    AppSessionSyncSerializer,
    AppUsageSyncSerializer,
    CallMetadataBatchSerializer,
    ConsentInSerializer,
    DeviceCallSyncSerializer,
    DeviceRecordingUploadSerializer,
    RecordingUploadRequestSerializer,
    ScreenUploadSerializer,
    app_session_payload,
    app_usage_payload,
    call_payload,
    policy_payload,
    recording_payload,
    screen_capture_payload,
)
from .services import (
    ANDROID_CALL_RECORDING_SOURCE,
    ANDROID_CALL_SOURCE,
    APP_USAGE_CATEGORY,
    APP_USAGE_SOURCE,
    AUDIO_CONTENT_TYPES,
    CALL_RECORDING_CATEGORY,
    HANDSET_CATEGORY,
    IN_APP_SCREEN_CATEGORY,
    SCREEN_CONTENT_TYPES,
    SCREEN_SHARE_CATEGORY,
    SCREEN_SOURCE,
    category_consented,
    deployment_id,
    feature_enabled,
    latest_consent,
    mask_number,
    new_storage_key,
    normalize_content_type,
    normalize_number,
    playback_expiry,
    policy_allows_category,
    policy_allows_handset_calls,
    record_audit,
    retention_days,
    probe_audio_duration,
    protected_file_response,
    safe_storage_path,
    scope_queryset,
    sign_playback,
    storage_path,
    unsign_playback,
)

User = get_user_model()


class FeatureDisabled(APIException):
    status_code = status.HTTP_501_NOT_IMPLEMENTED
    default_detail = "Employee monitoring is not enabled on this deployment."
    default_code = "feature_disabled"


class MonitorView(APIView):
    """Base: authenticated + feature-flag gate shared by every endpoint."""

    permission_classes = [permissions.IsAuthenticated]

    def initial(self, request, *args, **kwargs):
        super().initial(request, *args, **kwargs)
        # Deployment-wide switch; raising here keeps DRF's renderer/exception handling intact.
        if not feature_enabled():
            raise FeatureDisabled()


# Page sizing keeps each request small so the list endpoints never load the server with a huge
# scan + serialize. page_size is clamped; totals are computed with a cheap DB aggregate so a
# page stays correct without fetching every row.
PAGE_SIZE_DEFAULT = 50
PAGE_SIZE_MAX = 100


def match_call_for_recording(user, recorded_at, duration_seconds):
    """Best-effort link a handset recording to one of the user's calls by time. The recording's
    modified time (recorded_at) lands near the call: usually at the call's end, i.e. about
    started_at + duration. Pick the call whose start is closest to the recording, within 15 min.
    Returns None when no call is near (the recording then shows unlinked, which is honest)."""
    if not recorded_at:
        return None
    est_start = recorded_at - timedelta(seconds=duration_seconds or 0)
    tol = timedelta(minutes=15)
    lo = min(recorded_at, est_start) - tol
    hi = max(recorded_at, est_start) + tol
    best, best_delta = None, None
    for c in EmployeeCallRecord.objects.filter(user=user, started_at__gte=lo, started_at__lte=hi):
        if not c.started_at:
            continue
        delta = min(
            abs((c.started_at - recorded_at).total_seconds()),
            abs((c.started_at - est_start).total_seconds()),
        )
        if best_delta is None or delta < best_delta:
            best, best_delta = c, delta
    return best


def paginate(request, qs, default_size: int = PAGE_SIZE_DEFAULT, max_size: int = PAGE_SIZE_MAX):
    """Slice `qs` to one page using ?page & ?page_size. Returns (rows, meta). Caller must have
    applied any select_related/prefetch_related to `qs` before passing it in."""
    try:
        page = max(1, int(request.query_params.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    try:
        page_size = int(request.query_params.get("page_size", default_size))
    except (TypeError, ValueError):
        page_size = default_size
    page_size = max(1, min(page_size, max_size))
    total = qs.count()
    start = (page - 1) * page_size
    rows = list(qs[start:start + page_size])
    return rows, {
        "count": total,
        "page": page,
        "page_size": page_size,
        "has_next": start + page_size < total,
        "has_prev": page > 1,
    }


# Sources only the signed-in handset may create; server-side ingest (PBX etc.) must not use them.
RESERVED_SOURCES = frozenset({ANDROID_CALL_SOURCE, ANDROID_CALL_RECORDING_SOURCE, APP_USAGE_SOURCE, SCREEN_SOURCE})


def date_param(request):
    """Parse ?date=YYYY-MM-DD. Returns a date, None when absent; raises a 400 when invalid."""
    raw = (request.query_params.get("date") or "").strip()
    if not raw:
        return None
    try:
        value = parse_date(raw)
    except ValueError:
        value = None
    if value is None:
        raise ValidationError({"date": "Use YYYY-MM-DD."})
    return value


def parse_bool(value) -> bool:
    """JSON booleans as-is; strings only for clear truthy words ("false"/"0"/"" are False)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def remove_quietly(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def save_upload(upload, path: str):
    """Stream an upload to `path`; returns (size, sha256 hex). Removes a partial file on error."""
    digest = hashlib.sha256()
    size = 0
    try:
        with open(path, "wb") as fh:
            for chunk in upload.chunks():
                digest.update(chunk)
                size += len(chunk)
                fh.write(chunk)
    except BaseException:
        remove_quietly(path)
        raise
    return size, digest.hexdigest()


def _scoped_uuid_lookup(qs, user, raw_id):
    """One row by UUID through scope_queryset; None for malformed ids or out-of-scope rows."""
    try:
        obj_id = uuid.UUID(str(raw_id))
    except (TypeError, ValueError):
        return None
    return scope_queryset(qs, user).filter(id=obj_id).first()


class IngestView(MonitorView):
    """Endpoints that accept data from an approved source — restricted to overview roles."""

    def initial(self, request, *args, **kwargs):
        super().initial(request, *args, **kwargs)
        if not can_view_all_staff(request.user):
            self.permission_denied(request, message="Not permitted to ingest monitoring data.")


class MeView(MonitorView):
    def get(self, request):
        overview = can_view_all_staff(request.user)
        return Response({
            "username": request.user.get_username(),
            "deployment_id": deployment_id(),
            "is_overview": overview,
            "allowed_features": {
                "work_location": True,
                "call_metadata": True,
                "recording_playback": True,
                "activity_summary": False,
            },
        })


class PolicyCurrentView(MonitorView):
    def get(self, request):
        policy = MonitoringPolicy.objects.filter(is_active=True).order_by("-effective_at").first()
        if not policy:
            return Response(
                {"error": {"code": "NO_POLICY", "message": "No active monitoring policy is published."}},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(policy_payload(policy))


class ConsentView(MonitorView):
    """GET current consent state; POST acknowledge/withdraw."""

    def get(self, request):
        latest = MonitoringConsent.objects.filter(user=request.user).select_related("policy").first()
        if not latest:
            return Response({
                "consent_id": None, "status": "none", "policy_version": None, "effective_at": None, "disclosed_categories": [],
            })
        return Response({
            "consent_id": str(latest.id),
            "status": latest.status,
            "policy_version": latest.policy.version,
            "effective_at": latest.created_at.isoformat(),
            "disclosed_categories": latest.disclosed_categories or [],
        })

    def post(self, request):
        ser = ConsentInSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        policy = MonitoringPolicy.objects.filter(id=data["policy_id"]).first()
        if not policy:
            return Response(
                {"error": {"code": "NO_POLICY", "message": "Unknown policy."}}, status=status.HTTP_404_NOT_FOUND
            )
        consent = MonitoringConsent.objects.create(
            user=request.user,
            policy=policy,
            status=MonitoringConsent.ACKNOWLEDGED if data["action"] == "acknowledge" else MonitoringConsent.WITHDRAWN,
            disclosed_categories=data.get("disclosed_categories") or [],
        )
        record_audit(
            request,
            action=f"consent_{data['action']}",
            target_type="policy",
            target_id=policy.version,
        )
        return Response(
            {
                "consent_id": str(consent.id),
                "status": consent.status,
                "policy_version": policy.version,
                "effective_at": consent.created_at.isoformat(),
                "disclosed_categories": consent.disclosed_categories or [],
            },
            status=status.HTTP_201_CREATED,
        )


class DataCategoriesView(MonitorView):
    def get(self, request):
        policy = MonitoringPolicy.objects.filter(is_active=True).order_by("-effective_at").first()
        categories = []
        for c in (policy.categories if policy else []) or []:
            categories.append({
                "key": c.get("key", ""),
                "label": c.get("label", ""),
                "purpose": c.get("description", ""),
                "collected": c.get("collected", True),
                "retention_days": c.get("retention_days"),
                "measurement": c.get("measurement"),
            })
        return Response({"categories": categories})


class CallsView(MonitorView):
    def get(self, request):
        qs = scope_queryset(EmployeeCallRecord.objects.all(), request.user)
        date = date_param(request)
        if date:
            qs = qs.filter(started_at__date=date)
        username = (request.query_params.get("username") or "").strip()
        if username and can_view_all_staff(request.user):
            qs = qs.filter(user__username=username)
        qs = qs.select_related("user").prefetch_related("recordings")
        rows, meta = paginate(request, qs)
        # Full numbers only for overview roles, and every such disclosure is audited.
        full_numbers = can_view_all_staff(request.user)
        if full_numbers and rows:
            record_audit(
                request,
                action="call_numbers_viewed",
                target_type="calls",
                target_id=str(date or "all"),
                count=len(rows),
                username=username or "",
            )
        return Response({
            "results": [call_payload(c, include_number=full_numbers) for c in rows],
            **meta,
            # True total across all matching rows (cheap DB aggregate, not just this page).
            "total_duration_seconds": qs.aggregate(s=Sum("duration_seconds"))["s"] or 0,
            "next_cursor": None,
        })


class CallMetadataBatchView(IngestView):
    def post(self, request):
        ser = CallMetadataBatchSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        source = ser.validated_data["source"]
        if source in RESERVED_SOURCES:
            return Response(
                {"error": {"code": "SOURCE_RESERVED", "message": "Handset data must be uploaded by the signed-in device."}},
                status=status.HTTP_400_BAD_REQUEST,
            )
        source_label = ser.validated_data.get("source_label", "")
        accepted, duplicates = [], []
        for ev in ser.validated_data["events"]:
            user = User.objects.filter(username=ev["username"]).first() if ev.get("username") else None
            try:
                with transaction.atomic():
                    EmployeeCallRecord.objects.create(
                        user=user,
                        direction=ev["direction"],
                        number=ev.get("number", ""),
                        masked_number=mask_number(ev.get("number", "")),
                        started_at=ev["started_at"],
                        ended_at=ev.get("ended_at"),
                        duration_seconds=ev.get("duration_seconds"),
                        source=source,
                        source_label=source_label,
                        source_event_id=ev["source_event_id"],
                        recording_status=ev["recording_status"],
                    )
                accepted.append(ev["source_event_id"])
            except IntegrityError:
                duplicates.append(ev["source_event_id"])
        record_audit(request, action="call_metadata_ingest", target_type="call_batch",
                     reason=source, accepted=len(accepted), duplicates=len(duplicates))
        return Response({"accepted": accepted, "duplicates": duplicates, "rejected": []},
                        status=status.HTTP_201_CREATED)


class DeviceCallSyncView(MonitorView):
    """Android handset metadata for the signed-in user. No audio, no other employee's calls."""

    def post(self, request):
        policy = MonitoringPolicy.objects.filter(is_active=True).order_by("-effective_at").first()
        if policy is None or not policy_allows_handset_calls(policy):
            return Response(
                {
                    "error": {
                        "code": "POLICY_SCOPE",
                        "message": "The active monitoring policy does not include Android call history.",
                    }
                },
                status=status.HTTP_403_FORBIDDEN,
            )
        consent = MonitoringConsent.objects.filter(user=request.user).select_related("policy").first()
        if not category_consented(consent, policy, HANDSET_CATEGORY):
            return Response(
                {
                    "error": {
                        "code": "CONSENT_REQUIRED",
                        "message": "Opt in to Android call history under the current monitoring policy before it can be uploaded.",
                    }
                },
                status=status.HTTP_403_FORBIDDEN,
            )

        ser = DeviceCallSyncSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        device_uuid = ser.validated_data["device_uuid"]
        # Five minutes covers a slow phone clock without accepting older history.
        cutoff = consent.created_at - timedelta(minutes=5)
        accepted, duplicates, rejected = [], [], []
        for ev in ser.validated_data["events"]:
            client_id = ev["source_event_id"]
            if ev["started_at"] < cutoff:
                rejected.append({"source_event_id": client_id, "reason": "before_consent"})
                continue
            number = normalize_number(ev.get("number", ""))
            duration = ev.get("duration_seconds")
            try:
                with transaction.atomic():
                    EmployeeCallRecord.objects.create(
                        user=request.user,
                        direction=ev["direction"],
                        number=number,
                        masked_number=mask_number(number),
                        started_at=ev["started_at"],
                        ended_at=ev["started_at"] + timedelta(seconds=duration) if duration else None,
                        duration_seconds=duration,
                        source=ANDROID_CALL_SOURCE,
                        source_label="Android call history",
                        source_event_id=f"{device_uuid}:{client_id}",
                        recording_status=EmployeeCallRecord.UNAVAILABLE,
                        device_uuid=device_uuid,
                        sim_slot=(ev.get("sim_slot") or "")[:120],
                    )
                accepted.append(client_id)
            except IntegrityError:
                duplicates.append(client_id)
        record_audit(
            request,
            action="android_call_log_sync",
            target_type="call_batch",
            reason=ANDROID_CALL_SOURCE,
            accepted=len(accepted),
            duplicates=len(duplicates),
            rejected=len(rejected),
        )
        return Response(
            {"accepted": accepted, "duplicates": duplicates, "rejected": rejected},
            status=status.HTTP_201_CREATED,
        )


class DeviceCallRecordingUploadView(MonitorView):
    """Upload one call recording the phone's own dialer made, for the signed-in user only.

    This is the no-PBX path: Android cannot let the app tap the cellular audio stream, but an
    OEM dialer that records calls writes an audio file the officer owns. With this category opted
    in, the app uploads those files here. Gated exactly like the handset call log: the active
    policy must collect ``android_call_recording``, the user's latest consent must disclose that
    category, and only recordings from at/after the acknowledgment are accepted. The recording is
    always attributed to the uploading user and never to anyone else.
    """

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        policy = MonitoringPolicy.objects.filter(is_active=True).order_by("-effective_at").first()
        if policy is None or not policy_allows_category(policy, CALL_RECORDING_CATEGORY):
            return Response(
                {"error": {"code": "POLICY_SCOPE", "message": "The active monitoring policy does not include call recordings."}},
                status=status.HTTP_403_FORBIDDEN,
            )
        consent = latest_consent(request.user)
        if not category_consented(consent, policy, CALL_RECORDING_CATEGORY):
            return Response(
                {"error": {"code": "CONSENT_REQUIRED", "message": "Opt in to call recordings before they can be uploaded."}},
                status=status.HTTP_403_FORBIDDEN,
            )

        ser = DeviceRecordingUploadSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        # Reject recordings of calls that happened before the officer acknowledged (5-min clock slack).
        if data["recorded_at"] < consent.created_at - timedelta(minutes=5):
            return Response(
                {"error": {"code": "BEFORE_CONSENT", "message": "Recording predates your consent and was not stored."}},
                status=status.HTTP_409_CONFLICT,
            )

        upload = data["file"]
        if upload.size and upload.size > 50 * 1024 * 1024:
            return Response(
                {"error": {"code": "TOO_LARGE", "message": "Recording exceeds 50 MB."}},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        content_type = normalize_content_type(data.get("content_type") or "audio/mpeg")
        if content_type not in AUDIO_CONTENT_TYPES:
            return Response(
                {"error": {"code": "UNSUPPORTED_TYPE", "message": "Only audio recordings are accepted."}},
                status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            )

        device_uuid = data["device_uuid"]
        source_recording_id = f"{device_uuid}:{data['source_recording_id']}"
        # Link to a call-history row from the same device if the app matched one.
        call = None
        if data.get("source_call_event_id"):
            call = EmployeeCallRecord.objects.filter(
                source=ANDROID_CALL_SOURCE,
                source_event_id=f"{device_uuid}:{data['source_call_event_id']}",
                user=request.user,
            ).first()
        # Otherwise match by time: a dialer writes the recording when the call ends, so its
        # modified time sits near the call. Attach the closest call for this user.
        if call is None:
            call = match_call_for_recording(request.user, data["recorded_at"], data.get("duration_seconds"))

        storage_key = new_storage_key(ANDROID_CALL_RECORDING_SOURCE, request.user.id, AUDIO_CONTENT_TYPES[content_type])
        path = storage_path(storage_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        size, checksum = save_upload(upload, path)

        try:
            # Exact length read from the stored audio itself wins; then the device's value, then
            # the linked call's connected time. (Call connected time differs from the recording.)
            duration_seconds = (
                probe_audio_duration(path)
                or data.get("duration_seconds")
                or (call.duration_seconds if call else None)
            )
            with transaction.atomic():
                rec = EmployeeCallRecording.objects.create(
                    user=request.user,
                    call=call,
                    source=ANDROID_CALL_RECORDING_SOURCE,
                    source_label="Android call recording",
                    source_recording_id=source_recording_id,
                    storage_key=storage_key,
                    content_type=content_type,
                    size_bytes=size,
                    checksum_sha256=checksum,
                    duration_seconds=duration_seconds,
                    recorded_at=data["recorded_at"],
                    status=EmployeeCallRecording.AVAILABLE,
                    retention_deadline=playback_expiry(retention_days() * 86400) if retention_days() else None,
                )
                if call:
                    EmployeeCallRecord.objects.filter(id=call.id).update(
                        recording_status=EmployeeCallRecord.AVAILABLE
                    )
        except IntegrityError:
            remove_quietly(path)
            return Response({"status": "duplicate"}, status=status.HTTP_200_OK)
        except BaseException:
            remove_quietly(path)  # never leave an orphan file behind
            raise

        record_audit(request, action="call_recording_uploaded", target_type="recording", target_id=rec.id)
        return Response({"id": str(rec.id), "status": rec.status}, status=status.HTTP_201_CREATED)


class RecordingsView(MonitorView):
    def get(self, request):
        qs = scope_queryset(EmployeeCallRecording.objects.all(), request.user)
        date = date_param(request)
        if date:
            qs = qs.filter(recorded_at__date=date)
        username = (request.query_params.get("username") or "").strip()
        if username and can_view_all_staff(request.user):
            qs = qs.filter(user__username=username)
        # select_related so per-row user/call details don't each hit the DB.
        qs = qs.select_related("user", "call")
        rows, meta = paginate(request, qs)
        overview = can_view_all_staff(request.user)
        # Full numbers only for overview roles, and the disclosure is audited (as on CallsView).
        if overview and rows:
            record_audit(
                request, action="recording_numbers_viewed", target_type="recordings",
                target_id=str(date or "all"), count=len(rows), username=username or "",
            )
        results = [
            recording_payload(r, can_play=overview or r.user_id == request.user.id, include_number=overview)
            for r in rows
        ]
        return Response({"results": results, **meta, "next_cursor": None})


class RecordingUploadRequestView(IngestView):
    """Authorize a single private upload. The app never receives storage credentials."""

    def post(self, request):
        ser = RecordingUploadRequestSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        if data["source"] in RESERVED_SOURCES:
            return Response(
                {"error": {"code": "SOURCE_RESERVED", "message": "This source is reserved for the signed-in handset."}},
                status=status.HTTP_400_BAD_REQUEST,
            )
        content_type = normalize_content_type(data.get("content_type") or "audio/mpeg")
        if content_type not in AUDIO_CONTENT_TYPES:
            return Response(
                {"error": {"code": "UNSUPPORTED_TYPE", "message": "Only audio recordings are accepted."}},
                status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            )
        user = User.objects.filter(username=data["username"]).first() if data.get("username") else None
        call = None
        if data.get("source_call_event_id"):
            call = EmployeeCallRecord.objects.filter(
                source=data["source"], source_event_id=data["source_call_event_id"]
            ).first()
        # The storage key is always generated here, never built from client text. A repeated
        # request for the same recording keeps the (safe) key it was first given.
        existing = EmployeeCallRecording.objects.filter(
            source=data["source"], source_recording_id=data["source_recording_id"]
        ).first()
        storage_key = (
            existing.storage_key
            if existing and safe_storage_path(existing.storage_key)
            else new_storage_key(data["source"], None, AUDIO_CONTENT_TYPES[content_type])
        )
        rec, _ = EmployeeCallRecording.objects.update_or_create(
            source=data["source"],
            source_recording_id=data["source_recording_id"],
            defaults={
                "user": user or (call.user if call else None),
                "call": call,
                "content_type": content_type,
                "size_bytes": data.get("size_bytes"),
                "checksum_sha256": data.get("checksum_sha256", ""),
                "duration_seconds": data.get("duration_seconds"),
                "storage_key": storage_key,
                "status": EmployeeCallRecording.PENDING,
                "retention_deadline": playback_expiry(retention_days() * 86400) if retention_days() else None,
            },
        )
        record_audit(request, action="recording_upload_requested", target_type="recording", target_id=rec.id)
        return Response({
            "upload_id": str(rec.id),
            "status": "upload_authorized",
            "upload_method": "server_side_ingest",
            "storage_key": storage_key,
        }, status=status.HTTP_201_CREATED)


class RecordingUploadCompleteView(IngestView):
    def post(self, request, recording_id):
        rec = EmployeeCallRecording.objects.filter(id=recording_id).first()
        if not rec:
            raise Http404
        path = safe_storage_path(rec.storage_key)
        exists = bool(path) and os.path.isfile(path)
        rec.status = EmployeeCallRecording.AVAILABLE if exists else EmployeeCallRecording.FAILED
        rec.save(update_fields=["status"])
        if rec.call_id and exists:
            EmployeeCallRecord.objects.filter(id=rec.call_id).update(recording_status=EmployeeCallRecord.AVAILABLE)
        record_audit(request, action="recording_upload_completed", target_type="recording",
                     target_id=rec.id, verified=exists)
        return Response({"id": str(rec.id), "status": rec.status})


class RecordingPlaybackView(MonitorView):
    """Issue a short-lived signed URL. Playback itself is audited at stream time."""

    def get(self, request, recording_id):
        # Looked up through the caller's scope: own rows for staff, site rows for overview roles.
        rec = _scoped_uuid_lookup(EmployeeCallRecording.objects.all(), request.user, recording_id)
        if not rec:
            raise Http404
        if rec.status != EmployeeCallRecording.AVAILABLE:
            return Response(
                {"error": {"code": "RECORDING_UNAVAILABLE", "message": "Recording is not available."}},
                status=status.HTTP_409_CONFLICT,
            )
        token = sign_playback(str(rec.id))
        record_audit(request, action="recording_playback_authorized", target_type="recording", target_id=rec.id)
        url = request.build_absolute_uri(f"/api/v1/employee-monitor/recordings/stream/?t={token}")
        return Response({"url": url, "expires_at": playback_expiry().isoformat()})


class RecordingStreamView(MonitorView):
    """Stream the audio for a valid, unexpired signed token belonging to a permitted user."""

    def get(self, request):
        token = request.query_params.get("t", "")
        rec_id = unsign_playback(token)
        # Screen tokens carry a "screen:" prefix; only a bare UUID is a recording token.
        if not rec_id or rec_id.startswith("screen:"):
            raise Http404
        rec = _scoped_uuid_lookup(EmployeeCallRecording.objects.all(), request.user, rec_id)
        if not rec or rec.status != EmployeeCallRecording.AVAILABLE:
            raise Http404
        path = safe_storage_path(rec.storage_key)
        if not path or not os.path.isfile(path):
            raise Http404
        record_audit(request, action="recording_streamed", target_type="recording", target_id=rec.id)
        return protected_file_response(path, rec.content_type or "audio/mpeg", AUDIO_CONTENT_TYPES)


# --- App usage (Android handset, signed-in user) -------------------------------------------

class DeviceAppUsageSyncView(MonitorView):
    """Per-app foreground usage for the signed-in user only.

    Gated exactly like the handset call log: the active policy must include the ``app_usage``
    category, the user's latest consent must acknowledge this policy AND disclose that category,
    and only windows at/after the acknowledgment are accepted. Collects app names and durations,
    never the content shown inside those apps.
    """

    def post(self, request):
        policy = MonitoringPolicy.objects.filter(is_active=True).order_by("-effective_at").first()
        if policy is None or not policy_allows_category(policy, APP_USAGE_CATEGORY):
            return Response(
                {"error": {"code": "POLICY_SCOPE", "message": "The active monitoring policy does not include app usage."}},
                status=status.HTTP_403_FORBIDDEN,
            )
        consent = latest_consent(request.user)
        if not category_consented(consent, policy, APP_USAGE_CATEGORY):
            return Response(
                {"error": {"code": "CONSENT_REQUIRED", "message": "Opt in to app-usage collection before it can be uploaded."}},
                status=status.HTTP_403_FORBIDDEN,
            )

        ser = AppUsageSyncSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        device_uuid = ser.validated_data["device_uuid"]
        cutoff = consent.created_at - timedelta(minutes=5)
        accepted, duplicates, rejected = [], [], []
        for ev in ser.validated_data["events"]:
            client_id = ev["source_event_id"]
            if ev["window_start"] < cutoff:
                rejected.append({"source_event_id": client_id, "reason": "before_consent"})
                continue
            try:
                with transaction.atomic():
                    # Upsert on (source, source_event_id). The event id is derived from a fixed
                    # per-day window, so a day re-synced later (today's total keeps growing until
                    # midnight) UPDATES the same row rather than appending a duplicate. Appending
                    # was the cause of per-app totals ballooning on every sync.
                    # user is part of the lookup, so another user's row is never overwritten
                    # (the insert hits the unique constraint and is reported as a duplicate).
                    _, created = EmployeeAppUsage.objects.update_or_create(
                        source=APP_USAGE_SOURCE,
                        source_event_id=f"{device_uuid}:{client_id}"[:160],
                        user=request.user,
                        defaults=dict(
                            package_name=ev["package_name"][:160],
                            app_label=(ev.get("app_label") or "")[:200],
                            foreground_seconds=ev["foreground_seconds"],
                            launch_count=ev.get("launch_count", 0),
                            window_start=ev["window_start"],
                            window_end=ev["window_end"],
                            last_used_at=ev.get("last_used_at"),
                            device_uuid=device_uuid,
                        ),
                    )
                (accepted if created else duplicates).append(client_id)
            except IntegrityError:
                duplicates.append(client_id)
        record_audit(
            request, action="app_usage_sync", target_type="app_usage_batch", reason=APP_USAGE_SOURCE,
            accepted=len(accepted), duplicates=len(duplicates), rejected=len(rejected),
        )
        return Response(
            {"accepted": accepted, "duplicates": duplicates, "rejected": rejected},
            status=status.HTTP_201_CREATED,
        )


class AppUsageView(MonitorView):
    """List app-usage rows: own for ordinary staff, location-scoped for overview roles."""

    def get(self, request):
        qs = scope_queryset(EmployeeAppUsage.objects.all(), request.user)
        date = date_param(request)
        if date:
            qs = qs.filter(window_start__date=date)
        username = (request.query_params.get("username") or "").strip()
        if username and can_view_all_staff(request.user):
            qs = qs.filter(user__username=username)
        qs = qs.select_related("user")
        rows, meta = paginate(request, qs)
        packages = {r.package_name for r in rows}
        icons = dict(
            MonitoredAppIcon.objects.filter(package_name__in=packages, icon_base64__gt="")
            .values_list("package_name", "icon_base64")
        )
        return Response({
            "results": [app_usage_payload(r, icons) for r in rows],
            **meta,
            # True total across all matching rows (cheap DB aggregate, not just this page).
            "total_foreground_seconds": qs.aggregate(s=Sum("foreground_seconds"))["s"] or 0,
        })


class MobileVisibilityView(MonitorView):
    """What ordinary staff may see in the mobile app. Any signed-in user may read it (the app
    needs it to build its tab bar); only overview roles may change it. Admins are never
    restricted by these flags — they always see Calls and Logs."""

    def get(self, request):
        v = MobileVisibility.load()
        return Response({"staff_see_calls": v.staff_see_calls, "staff_see_logs": v.staff_see_logs})

    def patch(self, request):
        if not can_view_all_staff(request.user):
            return Response(
                {"error": {"code": "FORBIDDEN", "message": "Only supervisors can change staff visibility."}},
                status=status.HTTP_403_FORBIDDEN,
            )
        v = MobileVisibility.load()
        for field in ("staff_see_calls", "staff_see_logs"):
            if field in request.data:
                setattr(v, field, parse_bool(request.data[field]))
        v.save()
        return Response({"staff_see_calls": v.staff_see_calls, "staff_see_logs": v.staff_see_logs})


class AppIconSyncView(MonitorView):
    """Store launcher icons for app packages so the dashboard can show real icons. Gated like
    app-usage: active policy includes app_usage and the user has consented to it. Icons carry
    no personal content; they are deduplicated per package across the whole deployment."""

    def post(self, request):
        policy = MonitoringPolicy.objects.filter(is_active=True).order_by("-effective_at").first()
        if policy is None or not policy_allows_category(policy, APP_USAGE_CATEGORY):
            return Response(
                {"error": {"code": "POLICY_SCOPE", "message": "The active monitoring policy does not include app usage."}},
                status=status.HTTP_403_FORBIDDEN,
            )
        consent = latest_consent(request.user)
        if not category_consented(consent, policy, APP_USAGE_CATEGORY):
            return Response(
                {"error": {"code": "CONSENT_REQUIRED", "message": "Opt in to app-usage collection first."}},
                status=status.HTTP_403_FORBIDDEN,
            )
        ser = AppIconSyncSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        stored = []
        for ic in ser.validated_data["icons"]:
            MonitoredAppIcon.objects.update_or_create(
                package_name=ic["package_name"][:160],
                defaults={"app_label": (ic.get("app_label") or "")[:200], "icon_base64": ic["icon_base64"]},
            )
            stored.append(ic["package_name"])
        return Response({"stored": stored}, status=status.HTTP_200_OK)


class DeviceAppSessionSyncView(MonitorView):
    """Foreground-session timeline from the signed-in Android phone. Same consent/policy gate
    as app-usage totals, and only sessions starting at/after the acknowledgment are kept. Usage
    metadata only (which app, when, how long) — never what was shown or done inside the app."""

    def post(self, request):
        policy = MonitoringPolicy.objects.filter(is_active=True).order_by("-effective_at").first()
        if policy is None or not policy_allows_category(policy, APP_USAGE_CATEGORY):
            return Response(
                {"error": {"code": "POLICY_SCOPE", "message": "The active monitoring policy does not include app usage."}},
                status=status.HTTP_403_FORBIDDEN,
            )
        consent = latest_consent(request.user)
        if not category_consented(consent, policy, APP_USAGE_CATEGORY):
            return Response(
                {"error": {"code": "CONSENT_REQUIRED", "message": "Opt in to app-usage collection first."}},
                status=status.HTTP_403_FORBIDDEN,
            )
        ser = AppSessionSyncSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        device_uuid = ser.validated_data["device_uuid"]
        cutoff = consent.created_at - timedelta(minutes=5)
        accepted, duplicates, rejected = [], [], []
        for ev in ser.validated_data["events"]:
            client_id = ev["source_event_id"]
            if ev["started_at"] < cutoff:
                rejected.append({"source_event_id": client_id, "reason": "before_consent"})
                continue
            try:
                with transaction.atomic():
                    # user in the lookup: never overwrite another user's row.
                    _, created = EmployeeAppSession.objects.update_or_create(
                        source=APP_USAGE_SOURCE,
                        source_event_id=f"{device_uuid}:{client_id}"[:160],
                        user=request.user,
                        defaults=dict(
                            package_name=ev["package_name"][:160],
                            app_label=(ev.get("app_label") or "")[:200],
                            started_at=ev["started_at"],
                            ended_at=ev["ended_at"],
                            duration_seconds=ev["duration_seconds"],
                            device_uuid=device_uuid,
                        ),
                    )
                (accepted if created else duplicates).append(client_id)
            except IntegrityError:
                duplicates.append(client_id)
        record_audit(
            request, action="app_session_sync", target_type="app_session_batch", reason=APP_USAGE_SOURCE,
            accepted=len(accepted), duplicates=len(duplicates), rejected=len(rejected),
        )
        return Response(
            {"accepted": accepted, "duplicates": duplicates, "rejected": rejected},
            status=status.HTTP_201_CREATED,
        )


class AppSessionView(MonitorView):
    """List foreground sessions: own for ordinary staff, location-scoped for overview roles."""

    def get(self, request):
        qs = scope_queryset(EmployeeAppSession.objects.all(), request.user)
        date = date_param(request)
        if date:
            qs = qs.filter(started_at__date=date)
        username = (request.query_params.get("username") or "").strip()
        if username and can_view_all_staff(request.user):
            qs = qs.filter(user__username=username)
        package = (request.query_params.get("package") or "").strip()
        if package:
            qs = qs.filter(package_name=package)
        # Same 1000-row default as before, so callers that never page see no change.
        rows, meta = paginate(request, qs.select_related("user"), default_size=1000, max_size=1000)
        return Response({
            "results": [app_session_payload(r) for r in rows],
            **meta,
        })


# --- Screen captures (in-app screenshots + consented, visible screen share) -----------------

class ScreenUploadView(MonitorView):
    """Accept one screenshot / screen-share chunk from the signed-in device.

    Consent is per-kind: in-app screenshots need the ``in_app_screens`` category, screen-share
    chunks need ``screen_share``. The bytes are written to private storage immediately; the row
    never holds the image itself. iOS cannot capture other apps, so only CIIS's own screens and
    employee-started screen-share sessions ever reach here.
    """

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        ser = ScreenUploadSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        kind = data["kind"]
        category = IN_APP_SCREEN_CATEGORY if kind == EmployeeScreenCapture.IN_APP else SCREEN_SHARE_CATEGORY

        policy = MonitoringPolicy.objects.filter(is_active=True).order_by("-effective_at").first()
        if policy is None or not policy_allows_category(policy, category):
            return Response(
                {"error": {"code": "POLICY_SCOPE", "message": "The active monitoring policy does not include this capture."}},
                status=status.HTTP_403_FORBIDDEN,
            )
        consent = latest_consent(request.user)
        if not category_consented(consent, policy, category):
            return Response(
                {"error": {"code": "CONSENT_REQUIRED", "message": "Opt in to this capture before it can be uploaded."}},
                status=status.HTTP_403_FORBIDDEN,
            )

        upload = data["file"]
        if upload.size and upload.size > 25 * 1024 * 1024:
            return Response(
                {"error": {"code": "TOO_LARGE", "message": "Capture exceeds 25 MB."}},
                status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )
        content_type = normalize_content_type(data.get("content_type") or "image/jpeg")
        if content_type not in SCREEN_CONTENT_TYPES:
            return Response(
                {"error": {"code": "UNSUPPORTED_TYPE", "message": "Only JPEG/PNG/WebP images or MP4 video are accepted."}},
                status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            )
        storage_key = new_storage_key(SCREEN_SOURCE, request.user.id, SCREEN_CONTENT_TYPES[content_type])
        path = storage_path(storage_key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        size, checksum = save_upload(upload, path)

        try:
            with transaction.atomic():
                cap = EmployeeScreenCapture.objects.create(
                    user=request.user,
                    kind=kind,
                    session_id=(data.get("session_id") or "")[:64],
                    route=(data.get("route") or "")[:160],
                    storage_key=storage_key,
                    content_type=content_type,
                    size_bytes=size,
                    checksum_sha256=checksum,
                    captured_at=data.get("captured_at"),
                    status=EmployeeScreenCapture.AVAILABLE,
                    source=SCREEN_SOURCE,
                    source_capture_id=f"{request.user.id}:{data['source_capture_id']}"[:160],
                    retention_deadline=playback_expiry(retention_days() * 86400) if retention_days() else None,
                )
        except IntegrityError:
            remove_quietly(path)
            return Response({"status": "duplicate"}, status=status.HTTP_200_OK)
        except BaseException:
            remove_quietly(path)
            raise

        record_audit(request, action="screen_capture_uploaded", target_type="screen", target_id=cap.id, kind=kind)
        return Response({"id": str(cap.id), "status": cap.status}, status=status.HTTP_201_CREATED)


class ScreensView(MonitorView):
    def get(self, request):
        qs = scope_queryset(EmployeeScreenCapture.objects.all(), request.user)
        date = date_param(request)
        if date:
            qs = qs.filter(created_at__date=date)
        kind = (request.query_params.get("kind") or "").strip()
        if kind:
            qs = qs.filter(kind=kind)
        username = (request.query_params.get("username") or "").strip()
        if username and can_view_all_staff(request.user):
            qs = qs.filter(user__username=username)
        # Same 200-row default as before; ?page / ?page_size now reach older captures.
        rows, meta = paginate(request, qs.select_related("user"), default_size=200, max_size=200)
        overview = can_view_all_staff(request.user)
        results = [screen_capture_payload(c, can_view=overview or c.user_id == request.user.id) for c in rows]
        return Response({"results": results, **meta})


class ScreenPlaybackView(MonitorView):
    """Issue a short-lived signed URL for one capture; the stream itself is audited."""

    def get(self, request, capture_id):
        cap = _scoped_uuid_lookup(EmployeeScreenCapture.objects.all(), request.user, capture_id)
        if not cap:
            raise Http404
        if cap.status != EmployeeScreenCapture.AVAILABLE:
            return Response(
                {"error": {"code": "CAPTURE_UNAVAILABLE", "message": "Capture is not available."}},
                status=status.HTTP_409_CONFLICT,
            )
        token = sign_playback(f"screen:{cap.id}")
        record_audit(request, action="screen_playback_authorized", target_type="screen", target_id=cap.id)
        url = request.build_absolute_uri(f"/api/v1/employee-monitor/screens/stream/?t={token}")
        return Response({"url": url, "expires_at": playback_expiry().isoformat()})


class ScreenStreamView(MonitorView):
    def get(self, request):
        raw = unsign_playback(request.query_params.get("t", ""))
        if not raw or not raw.startswith("screen:"):
            raise Http404
        cap = _scoped_uuid_lookup(EmployeeScreenCapture.objects.all(), request.user, raw.split(":", 1)[1])
        if not cap or cap.status != EmployeeScreenCapture.AVAILABLE:
            raise Http404
        path = safe_storage_path(cap.storage_key)
        if not path or not os.path.isfile(path):
            raise Http404
        record_audit(request, action="screen_streamed", target_type="screen", target_id=cap.id)
        return protected_file_response(path, cap.content_type or "image/jpeg", SCREEN_CONTENT_TYPES)
