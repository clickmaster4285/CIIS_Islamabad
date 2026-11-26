"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
"""
from django.contrib import admin
from django.urls import include, path, re_path

from config.dashboard_views import DashboardGlanceAPIView
from config.media_views import protected_media

import os

from django.conf import settings

# Django admin has its own login without the app's lockout/audit, so it is off in production
# unless explicitly enabled (DJANGO_ADMIN_ENABLED=true); DJANGO_ADMIN_PATH moves it off /admin/.
_ADMIN_ENABLED = settings.DEBUG or os.getenv("DJANGO_ADMIN_ENABLED", "").strip().lower() in ("1", "true", "yes")
_ADMIN_PATH = (os.getenv("DJANGO_ADMIN_PATH", "admin").strip().strip("/") or "admin") + "/"

urlpatterns = [
    path("api/dashboard/glance/", DashboardGlanceAPIView.as_view(), name="dashboard-glance"),
    path("api/", include("users.urls")),
    path("api/", include("visitors.urls")),
    path("api/", include("logs.urls")),
    path("api/", include("detentions.urls")),
    path("api/", include("seizure_management.urls")),
    path("api/", include("cameras.urls")),
    path("api/", include("warehouse.urls")),
    path("api/", include("ml.urls")),
    path("api/", include("person_journey.urls")),
    path("api/", include("object_tracking.urls")),
    path("api/recognition/", include("recognition.urls")),
    path("api/identity/", include("identity.urls")),
    path("api/", include("ops_central.urls")),
    path("api/", include("gps_tracking.urls")),
    path("api/", include("video_recovery.urls")),
    path("api/", include("camera_health.urls")),
    path("api/", include("infrastructure_monitoring.urls")),
    path("api/", include("voice_agent.urls")),
    path("api/", include("requests_support.urls")),
    path("api/", include("city_sync.urls")),
    path("api/v1/employee-monitor/", include("employee_monitoring.urls")),
    # Detection clips, staff photos, attendance video, etc. — login required.
    re_path(r"^media/(?P<path>.*)$", protected_media, name="protected_media"),
]
if _ADMIN_ENABLED:
    urlpatterns.insert(0, path(_ADMIN_PATH, admin.site.urls))
