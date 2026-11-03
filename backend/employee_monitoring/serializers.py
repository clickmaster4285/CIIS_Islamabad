from __future__ import annotations

from rest_framework import serializers

from .models import (
    EmployeeAppSession,
    EmployeeAppUsage,
    EmployeeCallRecord,
    EmployeeCallRecording,
    EmployeeScreenCapture,
    MonitoredAppIcon,
    MonitoringPolicy,
)


def policy_payload(policy: MonitoringPolicy) -> dict:
    return {
        "policy_id": str(policy.id),
        "policy_version": policy.version,
        "effective_at": policy.effective_at.isoformat(),
        "summary": policy.summary,
        "work_hours": policy.work_hours,
        "contact": policy.contact,
        "categories": policy.categories or [],
    }


def call_payload(call: EmployeeCallRecord, *, include_number: bool = False) -> dict:
    """``include_number`` adds the decrypted full number; callers must restrict and audit it."""
    user = call.user
    recording = call.recordings.first()
    name = ""
    if user:
        name = (getattr(user, "full_name", "") or "").strip() or user.get_username()
    return {
        "id": str(call.id),
        "username": user.get_username() if user else None,
        "name": name or None,
        "direction": call.direction,
        "masked_number": call.masked_number or None,
        "started_at": call.started_at.isoformat() if call.started_at else None,
        "ended_at": call.ended_at.isoformat() if call.ended_at else None,
        "duration_seconds": call.duration_seconds,
        "source": call.source,
        "source_label": call.source_label,
        "sim_slot": call.sim_slot or None,
        "recording_status": call.recording_status,
        "recording_id": str(recording.id) if recording else None,
        **({"number": call.number or None} if include_number else {}),
    }


def recording_payload(rec: EmployeeCallRecording, *, can_play: bool, include_number: bool = False) -> dict:
    """``include_number`` adds the linked call's decrypted full number; callers must restrict
    and audit it, exactly as call_payload does."""
    user = rec.user
    name = ""
    if user:
        name = (getattr(user, "full_name", "") or "").strip() or user.get_username()
    call = rec.call
    return {
        "id": str(rec.id),
        "call_id": str(rec.call_id) if rec.call_id else None,
        # Whose call this recording belongs to.
        "username": user.get_username() if user else None,
        "name": name or None,
        # The linked call's details (null when the recording isn't tied to a call row).
        "direction": call.direction if call else None,
        "masked_number": (call.masked_number or None) if call else None,
        "call_started_at": (call.started_at.isoformat() if call.started_at else None) if call else None,
        "source": rec.source,
        "source_label": rec.source_label,
        "recorded_at": rec.recorded_at.isoformat() if rec.recorded_at else None,
        # The dialer file's own length isn't read on upload, so fall back to the linked call's
        # duration (they match) rather than showing 0s.
        "duration_seconds": rec.duration_seconds or (call.duration_seconds if call else None),
        "status": rec.status,
        "can_play": bool(can_play and rec.status == EmployeeCallRecording.AVAILABLE),
        **({"number": (call.number or None) if call else None} if include_number else {}),
    }


class ConsentInSerializer(serializers.Serializer):
    policy_id = serializers.UUIDField()
    action = serializers.ChoiceField(choices=["acknowledge", "withdraw"])
    disclosed_categories = serializers.ListField(child=serializers.CharField(), required=False, default=list)


class CallMetadataEventSerializer(serializers.Serializer):
    source_event_id = serializers.CharField(max_length=120)
    direction = serializers.ChoiceField(
        choices=[c[0] for c in EmployeeCallRecord.DIRECTION_CHOICES], default=EmployeeCallRecord.UNKNOWN
    )
    number = serializers.CharField(max_length=64, required=False, allow_blank=True, default="")
    username = serializers.CharField(max_length=150, required=False, allow_blank=True, default="")
    started_at = serializers.DateTimeField()
    ended_at = serializers.DateTimeField(required=False, allow_null=True)
    duration_seconds = serializers.IntegerField(required=False, allow_null=True, min_value=0)
    recording_status = serializers.ChoiceField(
        choices=[c[0] for c in EmployeeCallRecord.REC_STATUS_CHOICES], default=EmployeeCallRecord.UNAVAILABLE
    )


class CallMetadataBatchSerializer(serializers.Serializer):
    source = serializers.CharField(max_length=60)
    source_label = serializers.CharField(max_length=120, required=False, allow_blank=True, default="")
    events = CallMetadataEventSerializer(many=True)


class DeviceCallEventSerializer(serializers.Serializer):
    source_event_id = serializers.RegexField(r"^[A-Za-z0-9._:-]{1,40}$")
    direction = serializers.ChoiceField(choices=[c[0] for c in EmployeeCallRecord.DIRECTION_CHOICES])
    number = serializers.CharField(max_length=64, required=False, allow_blank=True, default="")
    started_at = serializers.DateTimeField()
    duration_seconds = serializers.IntegerField(required=False, allow_null=True, min_value=0, max_value=24 * 60 * 60)
    sim_slot = serializers.CharField(max_length=120, required=False, allow_blank=True, default="")


class DeviceCallSyncSerializer(serializers.Serializer):
    device_uuid = serializers.RegexField(r"^[A-Za-z0-9._:-]{8,64}$")
    events = DeviceCallEventSerializer(many=True)

    def validate_events(self, value):
        if len(value) > 100:
            raise serializers.ValidationError("At most 100 calls per sync.")
        return value


class RecordingUploadRequestSerializer(serializers.Serializer):
    source = serializers.CharField(max_length=60)
    source_recording_id = serializers.CharField(max_length=120)
    source_call_event_id = serializers.CharField(max_length=120, required=False, allow_blank=True, default="")
    username = serializers.CharField(max_length=150, required=False, allow_blank=True, default="")
    content_type = serializers.CharField(max_length=60, required=False, default="audio/mpeg")
    size_bytes = serializers.IntegerField(required=False, allow_null=True, min_value=0)
    checksum_sha256 = serializers.CharField(max_length=64, required=False, allow_blank=True, default="")
    duration_seconds = serializers.IntegerField(required=False, allow_null=True, min_value=0)


class DeviceRecordingUploadSerializer(serializers.Serializer):
    """Multipart upload of one call recording the phone's own dialer produced.

    ``file`` carries the audio bytes. ``recorded_at`` is when the call happened (used to reject
    recordings from before consent). ``source_call_event_id`` optionally links it to a call-history
    row already synced from the same device.
    """

    source_recording_id = serializers.RegexField(r"^[A-Za-z0-9._:-]{1,120}$")
    source_call_event_id = serializers.CharField(max_length=120, required=False, allow_blank=True, default="")
    device_uuid = serializers.RegexField(r"^[A-Za-z0-9._:-]{8,64}$")
    recorded_at = serializers.DateTimeField()
    duration_seconds = serializers.IntegerField(required=False, allow_null=True, min_value=0, max_value=24 * 60 * 60)
    content_type = serializers.CharField(max_length=60, required=False, allow_blank=True, default="audio/mpeg")
    file = serializers.FileField()


# --- app usage (Android handset, signed-in user) -------------------------------------------

class AppUsageEventSerializer(serializers.Serializer):
    source_event_id = serializers.RegexField(r"^[A-Za-z0-9._:-]{1,120}$")
    package_name = serializers.CharField(max_length=160)
    app_label = serializers.CharField(max_length=200, required=False, allow_blank=True, default="")
    foreground_seconds = serializers.IntegerField(min_value=0, max_value=36 * 60 * 60)
    launch_count = serializers.IntegerField(required=False, min_value=0, max_value=100000, default=0)
    window_start = serializers.DateTimeField()
    window_end = serializers.DateTimeField()
    last_used_at = serializers.DateTimeField(required=False, allow_null=True)


class AppUsageSyncSerializer(serializers.Serializer):
    device_uuid = serializers.RegexField(r"^[A-Za-z0-9._:-]{8,64}$")
    events = AppUsageEventSerializer(many=True)

    def validate_events(self, value):
        if len(value) > 500:
            raise serializers.ValidationError("At most 500 usage rows per sync.")
        return value


def app_usage_payload(row: EmployeeAppUsage, icons: dict[str, str] | None = None) -> dict:
    user = row.user
    name = ""
    if user:
        name = (getattr(user, "full_name", "") or "").strip() or user.get_username()
    icon = (icons or {}).get(row.package_name) or ""
    return {
        "id": str(row.id),
        "username": user.get_username() if user else None,
        "name": name or None,
        "package_name": row.package_name,
        "app_label": row.app_label or row.package_name,
        "foreground_seconds": row.foreground_seconds,
        "launch_count": row.launch_count,
        "window_start": row.window_start.isoformat() if row.window_start else None,
        "window_end": row.window_end.isoformat() if row.window_end else None,
        "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
        # data: URI ready for an <Image>, or "" when the phone has not uploaded this icon yet.
        "icon": f"data:image/png;base64,{icon}" if icon else "",
    }


# --- app icons (uploaded once per package so the dashboard can show real icons) -------------

class AppIconSerializer(serializers.Serializer):
    package_name = serializers.CharField(max_length=160)
    app_label = serializers.CharField(max_length=200, required=False, allow_blank=True, default="")
    # Raw base64 of a small PNG (no data: prefix). ~a few KB each; cap generously.
    icon_base64 = serializers.CharField(max_length=200_000)


class AppIconSyncSerializer(serializers.Serializer):
    icons = AppIconSerializer(many=True)

    def validate_icons(self, value):
        if len(value) > 300:
            raise serializers.ValidationError("At most 300 icons per sync.")
        return value


# --- app sessions (foreground usage timeline) ----------------------------------------------

class AppSessionEventSerializer(serializers.Serializer):
    source_event_id = serializers.RegexField(r"^[A-Za-z0-9._:-]{1,120}$")
    package_name = serializers.CharField(max_length=160)
    app_label = serializers.CharField(max_length=200, required=False, allow_blank=True, default="")
    started_at = serializers.DateTimeField()
    ended_at = serializers.DateTimeField()
    duration_seconds = serializers.IntegerField(min_value=0, max_value=36 * 60 * 60)


class AppSessionSyncSerializer(serializers.Serializer):
    device_uuid = serializers.RegexField(r"^[A-Za-z0-9._:-]{8,64}$")
    events = AppSessionEventSerializer(many=True)

    def validate_events(self, value):
        if len(value) > 1000:
            raise serializers.ValidationError("At most 1000 sessions per sync.")
        return value


def app_session_payload(row: EmployeeAppSession) -> dict:
    user = row.user
    name = ""
    if user:
        name = (getattr(user, "full_name", "") or "").strip() or user.get_username()
    return {
        "id": str(row.id),
        "username": user.get_username() if user else None,
        "name": name or None,
        "package_name": row.package_name,
        "app_label": row.app_label or row.package_name,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "ended_at": row.ended_at.isoformat() if row.ended_at else None,
        "duration_seconds": row.duration_seconds,
    }


# --- screen captures (in-app screenshots + consented screen share) --------------------------

class ScreenUploadSerializer(serializers.Serializer):
    """Multipart upload from the signed-in device. ``file`` carries the image/video bytes."""

    kind = serializers.ChoiceField(choices=[c[0] for c in EmployeeScreenCapture.KIND_CHOICES])
    source_capture_id = serializers.RegexField(r"^[A-Za-z0-9._:-]{1,120}$")
    session_id = serializers.CharField(max_length=64, required=False, allow_blank=True, default="")
    route = serializers.CharField(max_length=160, required=False, allow_blank=True, default="")
    captured_at = serializers.DateTimeField(required=False, allow_null=True)
    content_type = serializers.CharField(max_length=60, required=False, allow_blank=True, default="image/jpeg")
    checksum_sha256 = serializers.CharField(max_length=64, required=False, allow_blank=True, default="")
    file = serializers.FileField()


def screen_capture_payload(cap: EmployeeScreenCapture, *, can_view: bool) -> dict:
    user = cap.user
    name = ""
    if user:
        name = (getattr(user, "full_name", "") or "").strip() or user.get_username()
    return {
        "id": str(cap.id),
        "username": user.get_username() if user else None,
        "name": name or None,
        "kind": cap.kind,
        "route": cap.route or None,
        "content_type": cap.content_type,
        "size_bytes": cap.size_bytes,
        "captured_at": cap.captured_at.isoformat() if cap.captured_at else None,
        "status": cap.status,
        "can_view": bool(can_view and cap.status == EmployeeScreenCapture.AVAILABLE),
    }
