from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    CameraOfflineAlertAcknowledgeView,
    CameraOfflineAlertListView,
    CameraPreviewMjpegView,
    CameraStreamListView,
    CameraViewSet,
    NvrViewSet,
    SiteViewSet,
)

router = DefaultRouter()
router.register(r"sites", SiteViewSet, basename="site")
router.register(r"nvrs", NvrViewSet, basename="nvr")
router.register(r"cameras", CameraViewSet, basename="camera")

urlpatterns = [
    path("cameras/streams/", CameraStreamListView.as_view(), name="camera-stream-list"),
    path("cameras/offline-alerts/", CameraOfflineAlertListView.as_view(), name="camera-offline-alerts"),
    path(
        "cameras/offline-alerts/<int:pk>/acknowledge/",
        CameraOfflineAlertAcknowledgeView.as_view(),
        name="camera-offline-alert-ack",
    ),
    path("cameras/preview/mjpeg/", CameraPreviewMjpegView.as_view(), name="camera-preview-mjpeg"),
    path("", include(router.urls)),
]
