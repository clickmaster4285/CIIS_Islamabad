from django.http import HttpResponse
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from cameras.models import Camera, CameraStatus
from recognition.models import DetectionSnapshot
from recognition.scoping import staff_scope_q
from recognition.serializers import DetectionSnapshotSerializer
from recognition.services import cctv_control
from recognition.services.attendance_cameras import (
    attendance_camera_queryset,
    collect_attendance_camera_payloads,
)
from recognition.services.cctv_worker import get_cctv_manager
from recognition.services.snapshot_saver import snapshot_to_dict
from users.permissions import IsAdminOrHR

QUEUED_MESSAGE = "Sent to the attendance worker process; status updates within a few seconds."


def _emit_attendance_runtime() -> None:
    try:
        from realtime.sio_app import emit_invalidate

        emit_invalidate(["attendance", "recognition", "hr"])
    except Exception:
        pass


def _attendance_cameras():
    """All active cameras — shown on Attendance Monitor CCTV grid."""
    return attendance_camera_queryset(for_workers=False)


def _int_param(raw, *, default=None, minimum=None, maximum=None):
    """Parse an integer query param. Raises ValueError on junk or out-of-range values."""
    if raw is None or str(raw).strip() == "":
        return default
    value = int(str(raw).strip())
    if minimum is not None and value < minimum:
        raise ValueError
    if maximum is not None:
        value = min(value, maximum)
    return value


def _scoped_snapshots(user):
    return DetectionSnapshot.objects.filter(staff_scope_q(user, "staff__", local_only=False))


def _camera_payload(camera: Camera, runtime) -> dict:
    status_row = runtime.get_status(camera.id) if runtime else None
    try:
        has_stream = bool(camera.effective_stream_url())
    except Exception:
        has_stream = False
    return {
        "id": camera.id,
        "name": camera.name or camera.code or f"Camera {camera.id}",
        "location": camera.zone or "",
        "code": camera.code,
        "purpose": camera.purpose,
        "purposes": camera.purpose_list(),
        "purpose_label": camera.purpose_label,
        "status": camera.status,
        "is_active": camera.is_active and camera.status == CameraStatus.ONLINE,
        "has_stream": has_stream,
        "runtime": status_row,
    }


class CCTVControlView(APIView):
    permission_classes = [IsAdminOrHR]

    def get(self, request):
        runtime = cctv_control.get_runtime()
        cameras = list(_attendance_cameras())
        recent = _scoped_snapshots(request.user).select_related("staff", "camera").order_by("-detected_at")[:24]
        return Response({
            "cameras": [_camera_payload(c, runtime) for c in cameras],
            "events": runtime.get_events(limit=40),
            "snapshots": [snapshot_to_dict(s) for s in recent],
            "running_count": sum(1 for s in runtime.list_status() if s.get("running")),
        })

    def post(self, request):
        action_name = request.data.get("action", "start_all")
        if action_name not in ("start_all", "stop_all"):
            return Response(
                {"error": "Unknown action. Use start_all or stop_all."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        routed = cctv_control.request(action_name)
        if not routed["local"]:
            return Response(
                {"action": action_name, "queued": True, "statuses": [], "message": QUEUED_MESSAGE},
                status=status.HTTP_202_ACCEPTED,
            )

        manager = get_cctv_manager()
        if action_name == "start_all":
            cameras = collect_attendance_camera_payloads(for_workers=False)
            statuses = manager.start_all(cameras)
            _emit_attendance_runtime()
            return Response({"action": "start_all", "statuses": statuses, "count": len(statuses)})

        statuses = manager.stop_all()
        _emit_attendance_runtime()
        return Response({"action": "stop_all", "statuses": statuses})


class CCTVCameraActionView(APIView):
    permission_classes = [IsAdminOrHR]

    def post(self, request, camera_id, action_name):
        camera = Camera.objects.filter(pk=camera_id).select_related("nvr").first()
        if not camera:
            return Response({"error": "Camera not found"}, status=status.HTTP_404_NOT_FOUND)
        if action_name not in ("start", "stop"):
            return Response({"error": "Unknown action"}, status=status.HTTP_400_BAD_REQUEST)

        if action_name == "start":
            if not camera.is_active:
                return Response(
                    {"error": "Camera is inactive"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            try:
                url = camera.effective_stream_url()
            except Exception:
                return Response({"error": "Camera has no usable stream URL"}, status=status.HTTP_400_BAD_REQUEST)

        routed = cctv_control.request(action_name, camera.id)
        if not routed["local"]:
            return Response(
                {"queued": True, "message": QUEUED_MESSAGE, "camera": _camera_payload(camera, cctv_control.get_runtime())},
                status=status.HTTP_202_ACCEPTED,
            )

        manager = get_cctv_manager()
        if action_name == "start":
            runtime = manager.start_camera(
                camera.id,
                camera.name or camera.code or f"Camera {camera.id}",
                url,
                stream_key=camera.stream_key or f"cam-{camera.id}",
            )
            _emit_attendance_runtime()
            return Response({"camera": _camera_payload(camera, manager), "runtime": runtime})

        runtime = manager.stop_camera(camera.id)
        _emit_attendance_runtime()
        return Response({"runtime": runtime})

    def get(self, request, camera_id, action_name):
        camera = Camera.objects.filter(pk=camera_id).first()
        if not camera:
            return Response({"error": "Camera not found"}, status=status.HTTP_404_NOT_FOUND)

        runtime = cctv_control.get_runtime()
        if action_name == "snapshot":
            jpeg = runtime.get_snapshot_jpeg(camera.id)
            if not jpeg:
                return Response(
                    {"error": "No snapshot available. Start the camera first."},
                    status=status.HTTP_404_NOT_FOUND,
                )
            return HttpResponse(jpeg, content_type="image/jpeg")

        if action_name == "status":
            return Response({
                "camera_id": camera.id,
                "name": camera.name or camera.code,
                "runtime": runtime.get_status(camera.id),
            })

        return Response({"error": "Unknown action"}, status=status.HTTP_400_BAD_REQUEST)


class CCTVEventsView(APIView):
    permission_classes = [IsAdminOrHR]

    def get(self, request):
        try:
            camera_id = _int_param(request.query_params.get("camera_id"), minimum=1)
        except (TypeError, ValueError):
            return Response({"error": "camera_id must be a positive integer"}, status=status.HTTP_400_BAD_REQUEST)
        events = cctv_control.get_runtime().get_events(camera_id=camera_id, limit=50)
        return Response({"events": events})


class DetectionSnapshotListView(APIView):
    permission_classes = [IsAdminOrHR]

    def get(self, request):
        try:
            camera_id = _int_param(request.query_params.get("camera_id"), minimum=1)
            staff_id = _int_param(request.query_params.get("staff_id"), minimum=1)
            limit = _int_param(request.query_params.get("limit"), default=40, minimum=1, maximum=100)
        except (TypeError, ValueError):
            return Response(
                {"error": "camera_id, staff_id and limit must be positive integers"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        qs = _scoped_snapshots(request.user).select_related("staff", "camera").order_by("-detected_at")
        if camera_id:
            qs = qs.filter(camera_id=camera_id)
        if staff_id:
            qs = qs.filter(staff_id=staff_id)
        serializer = DetectionSnapshotSerializer(qs[:limit], many=True)
        return Response({"snapshots": serializer.data})
