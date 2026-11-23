from django.db.models import Q
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from cameras.models import Camera
from users.permissions import CAMERA_ADMIN_ROLES, get_location_scope

from .models import CameraHealthAlert, CameraHealthSnapshot
from .serializers import (
    CameraHealthAlertSerializer,
    CameraHealthSnapshotSerializer,
    CameraHealthSummarySerializer,
)
from .services import analyze_and_store, latest_summaries, start_background_scan

MAX_ID = 2**63 - 1


def camera_scope_q(user, prefix: str = "") -> Q:
    """
    Restrict Camera rows (optionally through a relation prefix) to the user's site.
    Global admins (no location scope) see everything; everyone else sees cameras whose
    location / NVR site matches their assigned location.
    """
    scope = get_location_scope(user)
    if not scope:
        return Q()
    pretty = scope.replace("_", " ")
    return (
        Q(**{f"{prefix}location__iexact": scope})
        | Q(**{f"{prefix}nvr__site__code__iexact": scope})
        | Q(**{f"{prefix}nvr__site__name__icontains": pretty})
    )


class IsCameraHealthAdmin(permissions.BasePermission):
    """Resolving alerts and triggering scans needs a camera admin role."""

    message = "A camera administrator role is required for this action."

    def has_permission(self, request, view):
        user = request.user
        return bool(
            user
            and user.is_authenticated
            and getattr(user, "role", None) in CAMERA_ADMIN_ROLES
        )


def _scoped_cameras(user):
    return Camera.objects.filter(camera_scope_q(user))


def _parse_int(raw, *, name: str, default: int | None = None, minimum: int = 1, maximum: int = MAX_ID):
    """(value, error_response). Missing -> default; invalid -> 400."""
    if raw in (None, ""):
        return default, None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None, Response({"detail": f"{name} must be an integer."}, status=status.HTTP_400_BAD_REQUEST)
    if value < minimum or value > maximum:
        return None, Response(
            {"detail": f"{name} must be between {minimum} and {maximum}."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    return value, None


class CameraHealthSummaryAPIView(APIView):
    """List each active camera with latest health score + open alert count."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        data = latest_summaries(_scoped_cameras(request.user))
        return Response(CameraHealthSummarySerializer(data, many=True).data)


class CameraHealthSnapshotListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        camera_id, err = _parse_int(request.query_params.get("camera_id"), name="camera_id")
        if err:
            return err
        limit, err = _parse_int(request.query_params.get("limit"), name="limit", default=50, maximum=200)
        if err:
            return err
        qs = CameraHealthSnapshot.objects.select_related("camera").filter(
            camera_scope_q(request.user, "camera__")
        )
        if camera_id:
            qs = qs.filter(camera_id=camera_id)
        return Response(CameraHealthSnapshotSerializer(qs[:limit], many=True).data)


class CameraHealthAlertListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        camera_id, err = _parse_int(request.query_params.get("camera_id"), name="camera_id")
        if err:
            return err
        limit, err = _parse_int(request.query_params.get("limit"), name="limit", default=100, maximum=300)
        if err:
            return err
        qs = CameraHealthAlert.objects.select_related("camera").filter(
            camera_scope_q(request.user, "camera__")
        )
        if request.query_params.get("open") in {"1", "true", "yes"}:
            qs = qs.filter(resolved=False)
        if camera_id:
            qs = qs.filter(camera_id=camera_id)
        return Response(CameraHealthAlertSerializer(qs[:limit], many=True).data)


class CameraHealthAlertResolveAPIView(APIView):
    permission_classes = [IsCameraHealthAdmin]

    def post(self, request, alert_id: int):
        from django.utils import timezone

        alert = (
            CameraHealthAlert.objects.filter(camera_scope_q(request.user, "camera__"), pk=alert_id)
            .select_related("camera")
            .first()
        )
        if alert is None:
            return Response({"detail": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        alert.resolved = True
        alert.resolved_at = timezone.now()
        alert.save(update_fields=["resolved", "resolved_at"])
        return Response(CameraHealthAlertSerializer(alert).data)


class CameraHealthScanAPIView(APIView):
    """
    Trigger a health scan.
      {"camera_id": N} — scan one camera now (synchronous, returns the snapshot).
      {} / {"limit": N} — scan all cameras in the user's scope in the background
                          (202; one scan at a time).
    """

    permission_classes = [IsCameraHealthAdmin]

    def post(self, request):
        data = request.data if isinstance(request.data, dict) else {}
        camera_id, err = _parse_int(data.get("camera_id"), name="camera_id")
        if err:
            return err
        cameras = _scoped_cameras(request.user).filter(is_active=True)
        if camera_id:
            cam = cameras.select_related("nvr", "nvr__site").filter(pk=camera_id).first()
            if cam is None:
                return Response({"detail": "Camera not found"}, status=status.HTTP_404_NOT_FOUND)
            snap = analyze_and_store(cam)
            return Response(CameraHealthSnapshotSerializer(snap).data)
        limit, err = _parse_int(data.get("limit"), name="limit", default=None, maximum=100000)
        if err:
            return err
        camera_ids = None
        if get_location_scope(request.user):
            camera_ids = list(cameras.order_by("id").values_list("id", flat=True))
        started = start_background_scan(camera_ids=camera_ids, limit=limit)
        if not started:
            return Response(
                {
                    "detail": "A health scan is already running.",
                    "queued": False,
                    "running": True,
                    "ok": 0,
                    "failed": 0,
                },
                status=status.HTTP_202_ACCEPTED,
            )
        return Response(
            {
                "detail": "Health scan started in the background; results appear as each camera finishes.",
                "queued": True,
                "running": True,
                "ok": 0,
                "failed": 0,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class CameraHealthDashboardAPIView(APIView):
    """Aggregate counts for the AI Suggestions dashboard."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        scope_q = camera_scope_q(request.user, "camera__")
        summaries = latest_summaries(_scoped_cameras(request.user))
        open_alerts = CameraHealthAlert.objects.filter(scope_q, resolved=False).count()
        by_status = {"healthy": 0, "degraded": 0, "critical": 0, "offline": 0, "unknown": 0}
        for row in summaries:
            key = str(row.get("status") or "unknown")
            by_status[key] = by_status.get(key, 0) + 1
        suggestions = (
            CameraHealthAlert.objects.filter(scope_q, resolved=False)
            .exclude(alert_type="healthy")
            .select_related("camera")
            .order_by("-last_detected")[:30]
        )
        return Response(
            {
                "camera_count": len(summaries),
                "open_alerts": open_alerts,
                "by_status": by_status,
                "cameras": CameraHealthSummarySerializer(summaries, many=True).data,
                "suggestions": CameraHealthAlertSerializer(suggestions, many=True).data,
            }
        )
