import logging

from rest_framework import permissions, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from django.conf import settings

from .client import (
    MLServiceError,
    ml_detect_image,
    ml_health,
    ml_reload_faces,
    ml_service_enabled,
    ml_validate_human_face,
    public_error_message,
)
from .face_sync import collect_db_face_embeddings
from . import job_store

logger = logging.getLogger(__name__)


class IsCameraAdmin(permissions.BasePermission):
    """Camera admin roles only (users.permissions.CAMERA_ADMIN_ROLES)."""

    def has_permission(self, request, view):
        from cameras.access import is_camera_admin

        return is_camera_admin(request.user)


def _ml_error(exc: MLServiceError, what: str) -> Response:
    logger.warning("%s failed: %s", what, exc)
    return Response({"detail": public_error_message(exc)}, status=exc.status_code or 503)


def _job_row_response(request, job_id: str, read_status, not_found: str) -> Response:
    if not job_store.is_valid_job_id(job_id):
        return Response({"detail": "Invalid job_id."}, status=400)
    row = read_status(job_id)
    if not row or not job_store.may_access(row, request.user):
        # Same 404 for other users' jobs: job ids are not an authorization token.
        return Response({"detail": not_found}, status=404)
    return Response(job_store.public_row(row))


class MLHealthAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        if not ml_service_enabled():
            return Response(
                {
                    "status": "disabled",
                    "message": "Detection engine not configured",
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        try:
            data = ml_health()
        except MLServiceError as exc:
            # Keep technical detail in logs; UI gets a short professional message.
            logger.warning("ML health check failed: %s", exc)
            return Response(
                {
                    "status": "error",
                    "message": "Detection engine temporarily unavailable",
                },
                status=exc.status_code or status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response({"status": "ok", **data})


class MLDetectImageAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        image = request.FILES.get("image")
        if not image:
            return Response({"detail": "image file is required."}, status=400)
        try:
            result = ml_detect_image(image.read(), filename=image.name or "image.jpg")
        except MLServiceError as exc:
            return _ml_error(exc, "ML detect image")
        return Response(result)


class MLValidateHumanFaceAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        image = request.FILES.get("image")
        if not image:
            return Response({"detail": "image file is required."}, status=400)
        try:
            result = ml_validate_human_face(image.read(), filename=image.name or "face.jpg")
        except MLServiceError as exc:
            return _ml_error(exc, "ML validate face")
        return Response(result)


class MLReloadFacesAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated, IsCameraAdmin]

    def post(self, request):
        try:
            result = ml_reload_faces(embeddings=collect_db_face_embeddings())
        except MLServiceError as exc:
            return _ml_error(exc, "ML reload faces")
        return Response(result)


def _form_float(request, name: str, default: float) -> float:
    raw = request.data.get(name, request.query_params.get(name, default))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


class MLVideoSearchAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        if not ml_service_enabled():
            return Response(
                {"detail": "ML service is not configured. Set ML_SERVICE_URL and start ml_services."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        image = request.FILES.get("image")
        video = request.FILES.get("video")
        if not image:
            return Response({"detail": "image file is required."}, status=400)
        if not video:
            return Response({"detail": "video file is required."}, status=400)

        max_bytes = int(getattr(settings, "ML_VIDEO_SEARCH_MAX_BYTES", 8 * 1024 * 1024 * 1024))
        video_size = int(getattr(video, "size", 0) or 0)
        if video_size > 0 and max_bytes > 0 and video_size > max_bytes:
            max_gb = max_bytes / (1024 * 1024 * 1024)
            return Response(
                {
                    "detail": (
                        f"Video is too large ({video_size / (1024 ** 3):.1f} GB). "
                        f"Maximum allowed is {max_gb:.0f} GB for Find in Video."
                    )
                },
                status=400,
            )

        from .video_search_jobs import start_job

        payload = start_job(
            image,
            video,
            {
                "clip_seconds": min(5.0, max(2.0, _form_float(request, "clip_seconds", 4.0))),
                "face_threshold": _form_float(request, "face_threshold", 0.45),
                "reid_threshold": _form_float(request, "reid_threshold", 0.88),
                "sample_fps": _form_float(request, "sample_fps", 0.0),
            },
            owner_id=request.user.pk,
        )
        return Response(payload, status=status.HTTP_202_ACCEPTED)

    def get(self, request):
        job_id = (request.query_params.get("job_id") or "").strip()
        if not job_id:
            return Response({"detail": "job_id is required."}, status=400)
        from .video_search_jobs import read_status

        return _job_row_response(request, job_id, read_status, "Search job not found.")


def _form_bool(request, name: str, default: bool = True) -> bool:
    raw = request.data.get(name, request.query_params.get(name, default))
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if not text:
        return default
    return text in ("1", "true", "yes", "on")


class MLVideoAnalyzeAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        if not ml_service_enabled():
            return Response(
                {"detail": "ML service is not configured. Set ML_SERVICE_URL and start ml_services."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        video = request.FILES.get("video")
        if not video:
            return Response({"detail": "video file is required."}, status=400)

        max_bytes = int(getattr(settings, "ML_VIDEO_ANALYZE_MAX_BYTES", 1 * 1024 * 1024 * 1024))
        video_size = int(getattr(video, "size", 0) or 0)
        if video_size > 0 and max_bytes > 0 and video_size > max_bytes:
            max_mb = max_bytes / (1024 * 1024)
            return Response(
                {
                    "detail": (
                        f"Video is too large ({video_size / (1024 ** 2):.0f} MB). "
                        f"Maximum allowed is {max_mb:.0f} MB for Video AI Test."
                    )
                },
                status=400,
            )

        person = _form_bool(request, "person", True)
        vehicle = _form_bool(request, "vehicle", True)
        weapon = _form_bool(request, "weapon", True)
        fire = _form_bool(request, "fire", True)
        plate = _form_bool(request, "plate", False)
        if not any([person, vehicle, weapon, fire, plate]):
            return Response({"detail": "Select at least one detection type."}, status=400)

        from .video_analyze_jobs import start_job

        payload = start_job(
            video,
            {
                "person": person,
                "vehicle": vehicle,
                "weapon": weapon,
                "fire": fire,
                "match_staff": _form_bool(request, "match_staff", True),
                "plate": plate,
                "sample_fps": min(4.0, max(0.5, _form_float(request, "sample_fps", 1.0))),
            },
            owner_id=request.user.pk,
        )
        return Response(payload, status=status.HTTP_202_ACCEPTED)

    def get(self, request):
        job_id = (request.query_params.get("job_id") or "").strip()
        if not job_id:
            return Response({"detail": "job_id is required."}, status=400)
        from .video_analyze_jobs import read_status

        return _job_row_response(request, job_id, read_status, "Analyze job not found.")


class MLVideoAnalyzeFileAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, job_id: str):
        from django.http import FileResponse

        from .video_analyze_jobs import read_status, tagged_file_path

        job_id = (job_id or "").strip()
        if not job_store.is_valid_job_id(job_id):
            return Response({"detail": "Invalid job_id."}, status=400)
        row = read_status(job_id)
        if not row or not job_store.may_access(row, request.user):
            return Response({"detail": "Tagged video is not ready."}, status=404)
        path = tagged_file_path(job_id)
        if path is None:
            return Response({"detail": "Tagged video is not ready."}, status=404)
        handle = path.open("rb")
        return FileResponse(handle, as_attachment=True, filename=path.name, content_type="video/mp4")
