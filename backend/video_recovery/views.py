from rest_framework import permissions, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import JobStatus, VideoRecoveryJob
from .permissions import CanUseVideoRecovery, scope_jobs
from .serializers import PIPELINE_STAGES, VideoRecoveryJobListSerializer, VideoRecoveryJobSerializer
from .services.corruption_types import CORRUPTION_CATALOG
from .services.ffmpeg_utils import ffmpeg_available
from .services.pipeline import is_job_running, mark_stale_jobs_failed, schedule_recovery
from .services.upload_validation import precheck_upload

READ_PERMISSIONS = [permissions.IsAuthenticated, CanUseVideoRecovery]


def _scoped_job(request, job_id):
    """The job if it exists and the caller may see it (own jobs, or all for forensic roles)."""
    return scope_jobs(VideoRecoveryJob.objects.all(), request.user).filter(pk=job_id).first()


class VideoRecoveryUploadAPIView(APIView):
    """POST /api/video-recovery/upload/ — upload a damaged video and start recovery."""

    permission_classes = READ_PERMISSIONS
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        upload = request.FILES.get("file") or request.FILES.get("video")
        if not upload:
            return Response({"detail": "file is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Extension, size and container magic bytes are checked before anything is stored.
        errors = precheck_upload(upload)
        if errors:
            return Response({"detail": " ".join(errors), "errors": errors}, status=status.HTTP_400_BAD_REQUEST)

        if not ffmpeg_available():
            return Response(
                {"detail": "ffmpeg/ffprobe not available on the server."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        job = VideoRecoveryJob(
            original_filename=(getattr(upload, "name", "") or "upload.mp4")[:255],
            uploaded_by=request.user,
        )
        try:
            job.original_file = upload
            job.save()
        except Exception:
            if job.original_file:
                job.original_file.delete(save=False)
            raise
        schedule_recovery(str(job.id))
        return Response(
            VideoRecoveryJobSerializer(job, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


class VideoRecoveryJobListAPIView(APIView):
    permission_classes = READ_PERMISSIONS

    def get(self, request):
        mark_stale_jobs_failed()
        qs = scope_jobs(VideoRecoveryJob.objects.all(), request.user).order_by("-created_at")[:100]
        return Response(
            VideoRecoveryJobListSerializer(qs, many=True, context={"request": request}).data
        )


class VideoRecoveryJobDetailAPIView(APIView):
    permission_classes = READ_PERMISSIONS

    def get(self, request, job_id):
        mark_stale_jobs_failed()
        job = _scoped_job(request, job_id)
        if job is None:
            return Response({"detail": "Job not found."}, status=status.HTTP_404_NOT_FOUND)
        return Response(VideoRecoveryJobSerializer(job, context={"request": request}).data)


class VideoRecoveryRetryAPIView(APIView):
    """POST /api/video-recovery/jobs/<id>/retry/ — re-run recovery pipeline."""

    permission_classes = READ_PERMISSIONS

    def post(self, request, job_id):
        mark_stale_jobs_failed()
        job = _scoped_job(request, job_id)
        if job is None:
            return Response({"detail": "Job not found."}, status=status.HTTP_404_NOT_FOUND)
        if job.status == JobStatus.PROCESSING or is_job_running(str(job.id)):
            return Response(
                {"detail": "This job is still running. Wait for it to finish before retrying."},
                status=status.HTTP_409_CONFLICT,
            )
        if not job.original_file:
            return Response(
                {"detail": "The original upload is no longer available; upload the video again."},
                status=status.HTTP_409_CONFLICT,
            )

        job.status = JobStatus.UPLOADED
        job.stage_logs = []
        job.forensic_report = {}
        job.damage_map = {}
        job.hybrid_report = {}
        job.quality_report = {}
        job.error_message = ""
        job.completed_at = None
        if job.recovered_file:
            job.recovered_file.delete(save=False)
        job.save()
        schedule_recovery(str(job.id))
        return Response(VideoRecoveryJobSerializer(job, context={"request": request}).data)


class VideoRecoveryPipelineSchemaAPIView(APIView):
    """GET /api/video-recovery/pipeline/ — pipeline architecture metadata."""

    permission_classes = READ_PERMISSIONS

    def get(self, request):
        return Response(
            {
                "name": "Hybrid Video Recovery + Regeneration",
                "principle": "Recover what is real + Regenerate what is lost",
                "stages": PIPELINE_STAGES,
                "flow": [
                    "Video Upload",
                    "Upload Validation",
                    "Preserve Original (SHA-256)",
                    "Deep Damage Analysis",
                    "Damage Map",
                    "Recovery Engine (container → stream → frame)",
                    "Regeneration Engine (restore / interpolate / generate)",
                    "Hybrid Merger",
                    "Temporal Consistency",
                    "Audio Recovery",
                    "Video Encoding",
                    "Quality Validation",
                    "Final Output",
                ],
                "corruption_types": CORRUPTION_CATALOG,
            }
        )
