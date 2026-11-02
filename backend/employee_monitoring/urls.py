from django.urls import path

from .views import (
    AppUsageView,
    CallMetadataBatchView,
    CallsView,
    ConsentView,
    DataCategoriesView,
    DeviceAppUsageSyncView,
    DeviceCallRecordingUploadView,
    DeviceCallSyncView,
    MeView,
    PolicyCurrentView,
    RecordingPlaybackView,
    RecordingStreamView,
    RecordingUploadCompleteView,
    RecordingUploadRequestView,
    RecordingsView,
    ScreenPlaybackView,
    ScreenStreamView,
    ScreenUploadView,
    ScreensView,
)

# Mounted under /api/v1/employee-monitor/
urlpatterns = [
    path("me/", MeView.as_view(), name="monitor-me"),
    path("policy/current/", PolicyCurrentView.as_view(), name="monitor-policy-current"),
    path("consents/", ConsentView.as_view(), name="monitor-consents"),
    path("consents/current/", ConsentView.as_view(), name="monitor-consents-current"),
    path("privacy/data-categories/", DataCategoriesView.as_view(), name="monitor-data-categories"),
    path("calls/", CallsView.as_view(), name="monitor-calls"),
    path("calls/metadata/batch/", CallMetadataBatchView.as_view(), name="monitor-calls-ingest"),
    path("calls/device-sync/", DeviceCallSyncView.as_view(), name="monitor-calls-device-sync"),
    path("recordings/", RecordingsView.as_view(), name="monitor-recordings"),
    # Handset upload of OEM-dialer call recordings for the signed-in user (no PBX).
    path("recordings/device-upload/", DeviceCallRecordingUploadView.as_view(), name="monitor-recording-device-upload"),
    path("recordings/uploads/", RecordingUploadRequestView.as_view(), name="monitor-recording-upload"),
    path("recordings/uploads/<uuid:recording_id>/complete/", RecordingUploadCompleteView.as_view(), name="monitor-recording-upload-complete"),
    path("recordings/stream/", RecordingStreamView.as_view(), name="monitor-recording-stream"),
    path("recordings/<uuid:recording_id>/playback/", RecordingPlaybackView.as_view(), name="monitor-recording-playback"),
    # App usage (Android handset, signed-in user).
    path("app-usage/", AppUsageView.as_view(), name="monitor-app-usage"),
    path("app-usage/device-sync/", DeviceAppUsageSyncView.as_view(), name="monitor-app-usage-device-sync"),
    # Screen captures: in-app screenshots + consented, visible screen share.
    path("screens/", ScreensView.as_view(), name="monitor-screens"),
    path("screens/upload/", ScreenUploadView.as_view(), name="monitor-screen-upload"),
    path("screens/stream/", ScreenStreamView.as_view(), name="monitor-screen-stream"),
    path("screens/<uuid:capture_id>/playback/", ScreenPlaybackView.as_view(), name="monitor-screen-playback"),
]
