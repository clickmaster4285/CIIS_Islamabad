import hashlib
import json
from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.shortcuts import get_object_or_404
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from city_sync.identity import local_rows_q
from recognition.models import FaceEnrollment, FaceSample
from recognition.scoping import face_sample_scope_q, scoped_staff, staff_scope_q
from recognition.serializers import FaceEnrollmentDetailSerializer
from recognition.services.face_engine import get_face_engine
from recognition.services.face_samples import review_sample
from recognition.services.snapshot_saver import save_detection_snapshot, snapshot_to_dict
from users.attendance_service import AttendanceDecisionEngine, staff_is_enrolled_for_attendance
from users.models import Attendance, Staff
from users.permissions import IsAdminOrHR


def min_enrollment_images() -> int:
    from recognition.services.training import min_enrollment_images as _required

    return _required()


def ensure_enrollment(staff: Staff) -> FaceEnrollment:
    folder = f"dataset/staff_{staff.pk}"
    enrollment, created = FaceEnrollment.objects.get_or_create(
        staff=staff,
        defaults={"dataset_folder": folder},
    )
    if created:
        Path(settings.MEDIA_ROOT, folder).mkdir(parents=True, exist_ok=True)
    return enrollment


# Wanted per enrollment: mostly frontal, plus a slight turn to each side for angled CCTV views.
POSE_TARGETS = {"front": 3, "side_a": 1, "side_b": 1}
POSES_FILE = "poses.json"


def _read_poses(folder: Path) -> dict[str, str]:
    try:
        return json.loads((folder / POSES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _record_pose(folder: Path, filename: str, bucket: str) -> dict[str, str]:
    poses = _read_poses(folder)
    poses[filename] = bucket
    (folder / POSES_FILE).write_text(json.dumps(poses), encoding="utf-8")
    return poses


def pose_coverage(folder: Path) -> dict[str, int]:
    """Photos per head-pose bucket. Photos from before pose tracking are not counted."""
    coverage = {bucket: 0 for bucket in POSE_TARGETS}
    for name, bucket in _read_poses(folder).items():
        if bucket in coverage and (folder / name).is_file():
            coverage[bucket] += 1
    return coverage


def pose_hint(coverage: dict[str, int]) -> str:
    if coverage["front"] < POSE_TARGETS["front"]:
        return "Next: look straight at the camera"
    if coverage["side_a"] < POSE_TARGETS["side_a"] and coverage["side_b"] < POSE_TARGETS["side_b"]:
        return "Next: turn your head slightly to one side"
    if coverage["side_a"] < POSE_TARGETS["side_a"] or coverage["side_b"] < POSE_TARGETS["side_b"]:
        return "Next: turn your head slightly to the other side"
    return "Pose coverage complete — vary lighting or glasses if you add more"


def build_gallery() -> dict[str, list[float]]:
    gallery = {}
    enrollments = FaceEnrollment.objects.filter(
        local_rows_q("staff__"),
        is_trained=True,
        embedding__isnull=False,
    ).select_related("staff")
    for enrollment in enrollments:
        gallery[enrollment.gallery_key] = enrollment.embedding
    return gallery


class CaptureImageSerializer(serializers.Serializer):
    image = serializers.CharField(help_text="Base64-encoded JPEG image")


class IdentifySerializer(serializers.Serializer):
    image = serializers.CharField(help_text="Base64-encoded JPEG from webcam/CCTV")
    mark_attendance = serializers.BooleanField(default=False)
    source = serializers.ChoiceField(
        choices=["cctv", "webcam", "manual"],
        default="webcam",
    )


class EnrollmentStatusView(APIView):
    permission_classes = [IsAdminOrHR]

    def get(self, request, staff_id):
        staff = get_object_or_404(scoped_staff(request.user), pk=staff_id)
        enrollment = ensure_enrollment(staff)
        coverage = pose_coverage(Path(settings.MEDIA_ROOT) / enrollment.dataset_folder)
        return Response({
            **FaceEnrollmentDetailSerializer(enrollment).data,
            "pose_coverage": coverage,
            "pose_hint": pose_hint(coverage),
        })


class CaptureFaceView(APIView):
    permission_classes = [IsAdminOrHR]

    def post(self, request, staff_id):
        serializer = CaptureImageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        staff = get_object_or_404(scoped_staff(request.user), pk=staff_id)
        enrollment = ensure_enrollment(staff)
        required = min_enrollment_images()

        engine = get_face_engine()
        try:
            image = engine.decode_base64(serializer.validated_data["image"])
        except ValueError as exc:
            return Response({"error": str(exc) or "Invalid image data"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            return Response({"error": "Invalid image data"}, status=status.HTTP_400_BAD_REQUEST)

        quality = engine.check_quality(image)
        save_image = quality.pop("image", None)
        if save_image is None:
            save_image = engine.resize_max(image, 640)
        if not quality["passed"]:
            return Response({"accepted": False, "quality": quality})

        folder = Path(settings.MEDIA_ROOT) / enrollment.dataset_folder
        # Row lock: two captures for the same person (double click, two tabs) get distinct indexes and
        # do not lose each other's poses.json entries; the file name is unique either way.
        with transaction.atomic():
            enrollment = FaceEnrollment.objects.select_for_update().get(pk=enrollment.pk)
            index = enrollment.total_images + 1
            saved = engine.save_dataset_image(save_image, folder, index, unique=True)
            bucket = (quality.get("pose") or {}).get("bucket")
            if bucket:
                _record_pose(folder, saved.name, bucket)
            coverage = pose_coverage(folder)

            enrollment.total_images = index
            enrollment.is_enrolled = enrollment.total_images >= required
            enrollment.is_trained = False
            enrollment.embedding = None
            update_fields = ["total_images", "is_enrolled", "is_trained", "embedding", "updated_at"]
            if index == 1:
                enrollment.profile_image = engine.save_profile_image(save_image, staff.pk)
                update_fields.append("profile_image")
            enrollment.save(update_fields=update_fields)

        # The person is untrained now: drop their old vectors from the live gallery.
        from recognition.services.gallery_index import invalidate_gallery

        invalidate_gallery()

        return Response({
            "accepted": True,
            "quality": quality,
            "total_images": enrollment.total_images,
            "images_required": required,
            "is_enrolled": enrollment.is_enrolled,
            "pose_coverage": coverage,
            "pose_hint": pose_hint(coverage),
        })


class TrainEmbeddingsView(APIView):
    permission_classes = [IsAdminOrHR]

    def post(self, request, staff_id):
        staff = get_object_or_404(scoped_staff(request.user), pk=staff_id)
        enrollment = ensure_enrollment(staff)
        required = min_enrollment_images()

        if enrollment.total_images < required:
            return Response(
                {
                    "error": f"Need at least {required} images, currently have {enrollment.total_images}",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        engine = get_face_engine()
        folder = Path(settings.MEDIA_ROOT) / enrollment.dataset_folder
        trained = engine.train_enrollment(enrollment, folder, required=required)
        if not trained.get("trained"):
            return Response(
                {"error": trained.get("error") or "Training failed"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not staff.face_identity_label:
            staff.face_identity_label = staff.full_name
            staff.save(update_fields=["face_identity_label"])

        enrollment.refresh_from_db()
        warnings = []
        coverage = pose_coverage(folder)
        if any(coverage.values()) and (coverage["side_a"] == 0 or coverage["side_b"] == 0):
            warnings.append(
                "All photos face the camera straight on. Add one with the head turned slightly "
                "to each side so angled CCTV views match better."
            )
        return Response({
            "trained": True,
            "warnings": warnings,
            "pose_coverage": coverage,
            "embedding_dim": trained["embedding_dim"],
            "images_used": trained["images_used"],
            "enrollment": FaceEnrollmentDetailSerializer(enrollment).data,
        })


class IdentifyFaceView(APIView):
    permission_classes = [IsAdminOrHR]

    def post(self, request):
        serializer = IdentifySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        engine = get_face_engine()
        raw_image = serializer.validated_data["image"]
        try:
            image = engine.decode_base64(raw_image)
        except ValueError as exc:
            return Response({"error": str(exc) or "Invalid image data"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            return Response({"error": "Invalid image data"}, status=status.HTTP_400_BAD_REQUEST)

        source = serializer.validated_data.get("source", Attendance.SOURCE_WEBCAM)
        result = engine.identify_from_image(image, source=source)
        response = dict(result)

        if result.get("matched") and serializer.validated_data.get("mark_attendance"):
            from recognition.services.match_confirm import CONFIRM_CCTV, CONFIRM_WEBCAM, get_match_confirmer

            staff_id = result.get("staff_id")
            needed = CONFIRM_CCTV if source == Attendance.SOURCE_CCTV else CONFIRM_WEBCAM
            # The same picture sent twice is one frame, not two confirmations.
            frame_hash = hashlib.sha1(raw_image.encode("utf-8", "ignore")).hexdigest()
            confirmed = get_match_confirmer().observe(f"{source}:{staff_id}", needed=needed, frame_hash=frame_hash)
            if not confirmed:
                response["attendance"] = {
                    "action": "pending_confirm",
                    "message": f"Face seen — need {needed} matches before attendance is written",
                }
                return Response(response)

            staff = get_object_or_404(scoped_staff(request.user).select_related("user"), pk=staff_id)
            if not staff_is_enrolled_for_attendance(staff):
                return Response(
                    {"error": "Staff face not trained"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
            decision = AttendanceDecisionEngine.process_recognition(
                staff=staff,
                confidence=result["confidence"],
                source=source,
            )
            record = decision["record"]

            if record and decision["action"] in ("check_in", "check_out"):
                snapshot = save_detection_snapshot(
                    staff=staff,
                    camera_id=0,
                    camera_name="Webcam",
                    frame=image,
                    confidence=result["confidence"],
                    attendance_action=decision["action"],
                    attendance_record=record,
                )
                if snapshot:
                    response["snapshot"] = snapshot_to_dict(snapshot)

            response["staff_name"] = staff.full_name
            response["employee_id"] = staff.employee_id
            response["attendance"] = {
                "action": decision["action"],
                "message": decision["message"],
                "date": str(record.date) if record else None,
                "check_in": record.check_in if record else None,
                "check_out": record.check_out if record else None,
                "status": record.status if record else None,
                "record_id": record.id if record else None,
            }

        return Response(response)


class GalleryStatsView(APIView):
    permission_classes = [IsAdminOrHR]

    def get(self, request):
        enrollments = FaceEnrollment.objects.filter(staff_scope_q(request.user, "staff__"))
        total = enrollments.count()
        enrolled = enrollments.filter(is_enrolled=True).count()
        trained = enrollments.filter(is_trained=True).count()
        return Response({
            "total_staff_with_enrollment": total,
            "enrolled": enrolled,
            "trained": trained,
            "ready_for_recognition": trained,
        })


def selfie_punch(user, image_b64: str) -> dict:
    """Phone punch. Skips MatchConfirmer. The face must be the logged-in employee."""
    staff = getattr(user, "staff_profile", None)
    if staff is None:
        return {"matched": False, "message": "No staff profile for this user"}
    engine = get_face_engine()
    image = engine.decode_base64(image_b64)
    result = engine.identify_from_image(image, source="webcam")
    if result.get("liveness_unavailable") or result.get("spoof"):
        # Liveness is enforced for phone punches (RECOGNITION_REQUIRE_LIVENESS, default on).
        return {"matched": False, "message": result.get("message"), "confidence": 0.0}
    if not result.get("matched") or result.get("staff_id") != staff.pk:
        return {"matched": False, "message": "Face does not match the logged-in employee", "confidence": result.get("confidence", 0.0)}
    decision = AttendanceDecisionEngine.process_recognition(
        staff=staff,
        confidence=result["confidence"],
        source=Attendance.SOURCE_WEBCAM,
    )
    return {
        "matched": True,
        "staff_id": staff.pk,
        "confidence": result["confidence"],
        "attendance": {
            "action": decision["action"],
            "message": decision["message"],
            "status": decision["record"].status if decision["record"] else "",
        },
    }


class SelfiePunchView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        image = (request.data.get("image") or "").strip()
        if not image:
            return Response({"error": "image is required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            payload = selfie_punch(request.user, image)
        except ValueError as exc:
            return Response({"error": str(exc) or "Invalid image data"}, status=status.HTTP_400_BAD_REQUEST)
        except Exception:
            return Response({"error": "Invalid image data"}, status=status.HTTP_400_BAD_REQUEST)
        code = status.HTTP_200_OK if payload.get("matched") else status.HTTP_400_BAD_REQUEST
        return Response(payload, status=code)


class FaceSampleSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()
    camera_name = serializers.SerializerMethodField()
    suggested_staff_name = serializers.CharField(source="suggested_staff.full_name", read_only=True, default=None)
    staff_name = serializers.CharField(source="staff.full_name", read_only=True, default=None)

    class Meta:
        model = FaceSample
        fields = [
            "id",
            "kind",
            "status",
            "staff_id",
            "staff_name",
            "suggested_staff_id",
            "suggested_staff_name",
            "camera_id",
            "camera_name",
            "image_url",
            "confidence",
            "quality",
            "created_at",
            "reviewed_at",
        ]

    def get_image_url(self, obj):
        return f"{settings.MEDIA_URL}{Path(obj.image.name).as_posix()}" if obj.image else ""

    def get_camera_name(self, obj):
        return getattr(obj.camera, "name", "") or ""


class FaceSampleListView(APIView):
    """CCTV face crops awaiting HR review (or reviewed, with ?status=)."""

    permission_classes = [IsAdminOrHR]

    def get(self, request):
        sample_status = request.query_params.get("status") or FaceSample.STATUS_PENDING
        if sample_status not in (FaceSample.STATUS_PENDING, FaceSample.STATUS_APPROVED, FaceSample.STATUS_REJECTED):
            return Response({"error": "status must be pending, approved or rejected"}, status=status.HTTP_400_BAD_REQUEST)
        visible = FaceSample.objects.filter(face_sample_scope_q(request.user))
        samples = (
            visible.filter(status=sample_status)
            .select_related("staff", "suggested_staff", "camera")
            .order_by("-created_at")[:100]
        )
        return Response({
            "results": FaceSampleSerializer(samples, many=True).data,
            "pending_count": visible.filter(status=FaceSample.STATUS_PENDING).count(),
        })


class FaceSampleReviewView(APIView):
    """approve (optionally as another staff member) or reject one sample."""

    permission_classes = [IsAdminOrHR]

    def post(self, request, sample_id):
        sample = get_object_or_404(FaceSample.objects.filter(face_sample_scope_q(request.user)), pk=sample_id)
        action = request.data.get("action")
        if action not in ("approve", "reject"):
            return Response({"error": "action must be approve or reject"}, status=status.HTTP_400_BAD_REQUEST)
        staff = None
        if request.data.get("staff_id"):
            try:
                staff_pk = int(request.data["staff_id"])
            except (TypeError, ValueError):
                return Response({"error": "staff_id must be an integer"}, status=status.HTTP_400_BAD_REQUEST)
            staff = get_object_or_404(scoped_staff(request.user), pk=staff_pk)
        try:
            review_sample(sample, approve=action == "approve", staff=staff, user=request.user)
        except ValueError as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(FaceSampleSerializer(sample).data)
