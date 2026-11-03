"""Employee-monitoring data models.

Call metadata arrives from two approved sources: the company phone system, and an Android
work phone after the employee has acknowledged the active policy that lists handset call
history. iOS cannot supply cellular call history. Audio bytes never live in these rows;
the handset path does not accept recordings at all.
"""

from __future__ import annotations

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone

from config.field_crypto import EncryptedCharField


class MonitoringPolicy(models.Model):
    """A versioned statement of what is collected, why, and for how long."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.CharField(max_length=40, unique=True)
    effective_at = models.DateTimeField(default=timezone.now)
    summary = models.TextField(blank=True, help_text="Purpose, who can view records, and how.")
    work_hours = models.CharField(max_length=200, blank=True)
    contact = models.CharField(max_length=200, blank=True)
    # [{key, label, description, retention_days}]
    categories = models.JSONField(default=list, blank=True)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-effective_at"]

    def __str__(self) -> str:
        return f"Policy {self.version}"


class MonitoringConsent(models.Model):
    """One acknowledgment or withdrawal event. The latest row per user is the current state."""

    ACKNOWLEDGED = "acknowledged"
    WITHDRAWN = "withdrawn"
    STATUS_CHOICES = [(ACKNOWLEDGED, "Acknowledged"), (WITHDRAWN, "Withdrawn")]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="monitoring_consents")
    policy = models.ForeignKey(MonitoringPolicy, on_delete=models.PROTECT, related_name="consents")
    status = models.CharField(max_length=16, choices=STATUS_CHOICES)
    disclosed_categories = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["user", "-created_at"])]

    def __str__(self) -> str:
        return f"{self.user_id} {self.status} {self.policy.version}"


class EmployeeCallRecord(models.Model):
    """Call metadata from the company phone system or a consented Android handset."""

    INBOUND, OUTBOUND, MISSED, UNKNOWN = "inbound", "outbound", "missed", "unknown"
    DIRECTION_CHOICES = [(INBOUND, "Inbound"), (OUTBOUND, "Outbound"), (MISSED, "Missed"), (UNKNOWN, "Unknown")]

    AVAILABLE, NOT_RECORDED, PENDING, FAILED, UNAVAILABLE = (
        "available", "not_recorded", "pending", "failed", "unavailable",
    )
    REC_STATUS_CHOICES = [
        (AVAILABLE, "Available"), (NOT_RECORDED, "Not recorded"),
        (PENDING, "Pending"), (FAILED, "Failed"), (UNAVAILABLE, "Unavailable"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="monitoring_calls"
    )
    direction = models.CharField(max_length=10, choices=DIRECTION_CHOICES, default=UNKNOWN)
    # Full number kept encrypted; a masked form is stored for list display.
    number = EncryptedCharField(max_length=64, blank=True, default="")
    masked_number = models.CharField(max_length=40, blank=True, default="")
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.PositiveIntegerField(null=True, blank=True)
    source = models.CharField(max_length=60)
    source_label = models.CharField(max_length=120, blank=True, default="")
    source_event_id = models.CharField(max_length=120)
    device_uuid = models.CharField(max_length=64, blank=True, default="")
    sim_slot = models.CharField(max_length=120, blank=True, default="")
    recording_status = models.CharField(max_length=16, choices=REC_STATUS_CHOICES, default=UNAVAILABLE)
    received_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-started_at"]
        constraints = [
            models.UniqueConstraint(fields=["source", "source_event_id"], name="uniq_call_source_event"),
        ]
        indexes = [
            models.Index(fields=["user", "-started_at"]),
            models.Index(fields=["-started_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.direction} {self.masked_number} @ {self.started_at:%Y-%m-%d %H:%M}"


class EmployeeCallRecording(models.Model):
    """Recording metadata + a private storage key. Audio bytes are never stored in the row."""

    AVAILABLE, PENDING, FAILED, NOT_RECORDED, UNAVAILABLE = (
        "available", "pending", "failed", "not_recorded", "unavailable",
    )
    STATUS_CHOICES = [
        (AVAILABLE, "Available"), (PENDING, "Pending"), (FAILED, "Failed"),
        (NOT_RECORDED, "Not recorded"), (UNAVAILABLE, "Unavailable"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    call = models.ForeignKey(
        EmployeeCallRecord, null=True, blank=True, on_delete=models.SET_NULL, related_name="recordings"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="monitoring_recordings"
    )
    source = models.CharField(max_length=60)
    source_label = models.CharField(max_length=120, blank=True, default="")
    source_recording_id = models.CharField(max_length=120)
    storage_key = models.CharField(max_length=300, blank=True, default="")
    content_type = models.CharField(max_length=60, blank=True, default="audio/mpeg")
    size_bytes = models.BigIntegerField(null=True, blank=True)
    checksum_sha256 = models.CharField(max_length=64, blank=True, default="")
    duration_seconds = models.PositiveIntegerField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=PENDING)
    recorded_at = models.DateTimeField(null=True, blank=True)
    retention_deadline = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-recorded_at", "-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["source", "source_recording_id"], name="uniq_recording_source_id"),
        ]
        indexes = [models.Index(fields=["user", "-created_at"])]

    def __str__(self) -> str:
        return f"Recording {self.source}:{self.source_recording_id} ({self.status})"


class MonitoringAudit(models.Model):
    """Append-only record of sensitive access — playback, export, policy and consent changes."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="monitoring_audits"
    )
    action = models.CharField(max_length=60)
    target_type = models.CharField(max_length=60, blank=True, default="")
    target_id = models.CharField(max_length=64, blank=True, default="")
    reason = models.CharField(max_length=200, blank=True, default="")
    request_ip = models.GenericIPAddressField(null=True, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["-created_at"]), models.Index(fields=["actor", "-created_at"])]

    def __str__(self) -> str:
        return f"{self.action} by {self.actor_id} @ {self.created_at:%Y-%m-%d %H:%M}"


class EmployeeAppUsage(models.Model):
    """Per-app foreground usage from an Android phone, after per-category consent.

    App package names and durations only — never the content shown inside those apps,
    and never keystrokes or message text. iOS cannot supply this. Rows are only accepted
    for windows that start at or after the employee's acknowledgment of the active policy.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="monitoring_app_usage"
    )
    package_name = models.CharField(max_length=160)
    app_label = models.CharField(max_length=200, blank=True, default="")
    foreground_seconds = models.PositiveIntegerField(default=0)
    launch_count = models.PositiveIntegerField(default=0)
    window_start = models.DateTimeField()
    window_end = models.DateTimeField()
    last_used_at = models.DateTimeField(null=True, blank=True)
    source = models.CharField(max_length=60, default="android_app_usage")
    source_event_id = models.CharField(max_length=160)
    device_uuid = models.CharField(max_length=64, blank=True, default="")
    received_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-window_start"]
        constraints = [
            models.UniqueConstraint(fields=["source", "source_event_id"], name="uniq_app_usage_source_event"),
        ]
        indexes = [
            models.Index(fields=["user", "-window_start"]),
            models.Index(fields=["-window_start"]),
        ]

    def __str__(self) -> str:
        return f"{self.package_name} {self.foreground_seconds}s @ {self.window_start:%Y-%m-%d %H:%M}"


class EmployeeScreenCapture(models.Model):
    """A screenshot of the CIIS app, or a chunk of a consented, visible screen-share session.

    The image/video bytes live in private storage (storage_key); the row holds only metadata.
    Capture is never silent: in-app screenshots are of CIIS's own UI, and screen-share runs
    only while the operating system shows its own recording indicator and the employee started it.
    """

    IN_APP, SCREEN_SHARE = "in_app_screen", "screen_share"
    KIND_CHOICES = [(IN_APP, "In-app screen"), (SCREEN_SHARE, "Screen share")]

    AVAILABLE, PENDING, FAILED, UNAVAILABLE = "available", "pending", "failed", "unavailable"
    STATUS_CHOICES = [
        (AVAILABLE, "Available"), (PENDING, "Pending"), (FAILED, "Failed"), (UNAVAILABLE, "Unavailable"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="monitoring_screens"
    )
    kind = models.CharField(max_length=20, choices=KIND_CHOICES, default=IN_APP)
    session_id = models.CharField(max_length=64, blank=True, default="")
    route = models.CharField(max_length=160, blank=True, default="", help_text="CIIS screen name for in-app captures.")
    storage_key = models.CharField(max_length=300, blank=True, default="")
    content_type = models.CharField(max_length=60, blank=True, default="image/jpeg")
    size_bytes = models.BigIntegerField(null=True, blank=True)
    checksum_sha256 = models.CharField(max_length=64, blank=True, default="")
    captured_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=PENDING)
    source = models.CharField(max_length=60, default="mobile_screen")
    source_capture_id = models.CharField(max_length=160)
    device_uuid = models.CharField(max_length=64, blank=True, default="")
    retention_deadline = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-captured_at", "-created_at"]
        constraints = [
            models.UniqueConstraint(fields=["source", "source_capture_id"], name="uniq_screen_source_id"),
        ]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["kind", "-created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.kind} {self.source_capture_id} ({self.status})"


class MonitoredAppIcon(models.Model):
    """Launcher icon for an app package, uploaded by a monitored phone so the dashboard can
    show it. Admins view from a different device that may not have the app installed, so the
    icon bytes must travel with the data. A small PNG, stored base64; no content, just the icon.
    """

    package_name = models.CharField(max_length=160, unique=True)
    app_label = models.CharField(max_length=200, blank=True, default="")
    icon_base64 = models.TextField(blank=True, default="")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["package_name"]

    def __str__(self) -> str:
        return self.package_name


class EmployeeAppSession(models.Model):
    """One foreground session: an app was in front from started_at to ended_at. This is usage
    metadata (which app, when, how long) only — never what was shown or done inside the app.
    Accepted on the same consent/policy gate as app-usage totals.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
        related_name="monitoring_app_sessions",
    )
    package_name = models.CharField(max_length=160)
    app_label = models.CharField(max_length=200, blank=True, default="")
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField()
    duration_seconds = models.PositiveIntegerField(default=0)
    source = models.CharField(max_length=60, default="android_app_usage")
    source_event_id = models.CharField(max_length=160)
    device_uuid = models.CharField(max_length=64, blank=True, default="")
    received_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-started_at"]
        constraints = [
            models.UniqueConstraint(fields=["source", "source_event_id"], name="uniq_app_session_source_event"),
        ]
        indexes = [
            models.Index(fields=["user", "-started_at"]),
            models.Index(fields=["-started_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.package_name} {self.duration_seconds}s @ {self.started_at:%Y-%m-%d %H:%M}"
