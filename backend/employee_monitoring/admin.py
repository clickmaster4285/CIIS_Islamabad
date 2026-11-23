import os

from django.contrib import admin
from django.http import Http404
from django.urls import path, reverse
from django.utils.html import format_html

from .models import (
    EmployeeAppUsage,
    EmployeeCallRecord,
    EmployeeCallRecording,
    EmployeeScreenCapture,
    MonitoringAudit,
    MonitoringConsent,
    MonitoringPolicy,
)
from .services import (
    SCREEN_CONTENT_TYPES,
    protected_file_response,
    record_audit,
    safe_storage_path,
    scope_queryset,
)


def _admin_scope(qs, request):
    """Django superusers see everything; other admin users get the same own/site scope as the
    API (scope_queryset), so a location-scoped staff account cannot browse other sites."""
    if request.user.is_superuser:
        return qs
    return scope_queryset(qs, request.user)


@admin.register(MonitoringPolicy)
class MonitoringPolicyAdmin(admin.ModelAdmin):
    list_display = ("version", "is_active", "effective_at", "created_at")
    list_filter = ("is_active",)
    search_fields = ("version",)


@admin.register(MonitoringConsent)
class MonitoringConsentAdmin(admin.ModelAdmin):
    list_display = ("user", "status", "policy", "created_at")
    list_filter = ("status",)
    search_fields = ("user__username",)
    raw_id_fields = ("user", "policy")


@admin.register(EmployeeCallRecord)
class EmployeeCallRecordAdmin(admin.ModelAdmin):
    # Read-only evidence. The full number is encrypted and never shown here (excluded from the
    # form too, which would otherwise render it decrypted); only the masked form is visible.
    list_display = ("masked_number", "direction", "user", "started_at", "duration_seconds", "source", "recording_status")
    list_filter = ("direction", "recording_status", "source")
    search_fields = ("masked_number", "user__username", "source_event_id")
    exclude = ("number",)

    def get_queryset(self, request):
        return _admin_scope(super().get_queryset(request), request)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(EmployeeCallRecording)
class EmployeeCallRecordingAdmin(admin.ModelAdmin):
    list_display = ("source_recording_id", "user", "status", "source", "recorded_at", "retention_deadline")
    list_filter = ("status", "source")
    search_fields = ("source_recording_id", "user__username")

    def get_queryset(self, request):
        return _admin_scope(super().get_queryset(request), request)


class DeviceDataAdmin(admin.ModelAdmin):
    """Rows arrive from consented devices; admins can inspect and delete, never add or edit."""

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


@admin.register(EmployeeAppUsage)
class EmployeeAppUsageAdmin(DeviceDataAdmin):
    list_display = ("app_name", "user", "foreground_minutes", "launch_count", "window_start", "window_end", "device_uuid")
    list_filter = ("source", "window_start")
    search_fields = ("package_name", "app_label", "user__username", "device_uuid")
    date_hierarchy = "window_start"
    list_select_related = ("user",)

    @admin.display(description="App", ordering="app_label")
    def app_name(self, obj):
        return obj.app_label or obj.package_name

    @admin.display(description="Minutes", ordering="foreground_seconds")
    def foreground_minutes(self, obj):
        return round(obj.foreground_seconds / 60, 1)


@admin.register(EmployeeScreenCapture)
class EmployeeScreenCaptureAdmin(DeviceDataAdmin):
    list_display = ("captured_at", "user", "kind", "route", "status", "size_bytes", "image_link")
    list_filter = ("kind", "status", "captured_at")
    search_fields = ("route", "user__username", "device_uuid", "source_capture_id")
    date_hierarchy = "captured_at"
    list_select_related = ("user",)
    readonly_fields = ("preview",)

    def get_queryset(self, request):
        return _admin_scope(super().get_queryset(request), request)

    def get_urls(self):
        view = self.admin_site.admin_view(self.image_view)
        return [path("<uuid:capture_id>/image/", view, name="employee_monitoring_screen_image")] + super().get_urls()

    def image_view(self, request, capture_id):
        # Same audit trail as the API stream: every admin view of a capture is recorded.
        cap = self.get_queryset(request).filter(id=capture_id).first()
        if not cap or cap.status != EmployeeScreenCapture.AVAILABLE or not self.has_view_permission(request, cap):
            raise Http404
        file_path = safe_storage_path(cap.storage_key)
        if not file_path or not os.path.isfile(file_path):
            raise Http404
        record_audit(request, action="screen_streamed", target_type="screen", target_id=cap.id, via="admin")
        return protected_file_response(file_path, cap.content_type or "image/jpeg", SCREEN_CONTENT_TYPES)

    def _image_url(self, obj):
        if obj.status != EmployeeScreenCapture.AVAILABLE:
            return ""
        return reverse("admin:employee_monitoring_screen_image", args=[obj.id])

    @admin.display(description="Image")
    def image_link(self, obj):
        url = self._image_url(obj)
        return format_html('<a href="{}" target="_blank">Open</a>', url) if url else "—"

    @admin.display(description="Preview")
    def preview(self, obj):
        url = self._image_url(obj)
        if not url:
            return f"Not available ({obj.get_status_display()})"
        return format_html('<a href="{0}" target="_blank"><img src="{0}" style="max-height:480px;border:1px solid #ccc"></a>', url)


@admin.register(MonitoringAudit)
class MonitoringAuditAdmin(admin.ModelAdmin):
    list_display = ("action", "actor", "target_type", "target_id", "created_at")
    list_filter = ("action",)
    search_fields = ("actor__username", "action", "target_id")
    readonly_fields = ("actor", "action", "target_type", "target_id", "reason", "request_ip", "metadata", "created_at")

    def has_add_permission(self, request):
        return False
