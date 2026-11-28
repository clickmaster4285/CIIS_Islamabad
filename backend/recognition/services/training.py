"""Train ArcFace galleries from saved enrollment photos (ported from AIAttendance)."""
from __future__ import annotations

import logging
from pathlib import Path

from django.conf import settings

from recognition.services.face_engine import get_face_engine
from recognition.services.gallery_index import invalidate_gallery

logger = logging.getLogger(__name__)

MODEL_VERSION = "InsightFace_v1_multi_light"


class TrainingError(ValueError):
    pass


def min_enrollment_images() -> int:
    return max(3, int(getattr(settings, "ATTENDANCE_MIN_ENROLLMENT_IMAGES", 5)))


def train_enrollment(enrollment) -> dict:
    """Train from the enrollment's dataset folder (same path as the Face Enrollment page's Train button)."""
    required = min_enrollment_images()
    if enrollment.total_images < required:
        raise TrainingError(
            f"Need at least {required} images, currently have {enrollment.total_images}"
        )

    engine = get_face_engine()
    folder = Path(settings.MEDIA_ROOT) / enrollment.dataset_folder
    result = engine.train_enrollment(enrollment, folder, required=required)
    if not result.get("trained"):
        raise TrainingError(result.get("error") or "Training failed")
    result["enrollment"] = enrollment
    return result


STAFF_PHOTO_PREFIX = "staffphoto_"


def enroll_from_staff_photos(staff) -> dict:
    """
    Build the attendance (InsightFace ArcFace 512-d) enrollment from the photos uploaded in Add / Edit Staff.

    Each staff photo that passes the same quality check as a webcam capture is copied into the
    enrollment folder as ``staffphoto_<hash>.jpg`` (webcam captures from the Face Enrollment page
    stay alongside it), photos removed from the staff record are dropped, and the enrollment is
    retrained. Fewer usable photos than ATTENDANCE_MIN_ENROLLMENT_IMAGES leaves it untrained.
    """
    import hashlib

    import cv2
    from django.db import transaction

    from recognition.models import FaceEnrollment
    from users.staff_photos import staff_photo_paths

    paths = staff_photo_paths(staff)
    folder_rel = f"dataset/staff_{staff.pk}"
    enrollment, _ = FaceEnrollment.objects.get_or_create(staff=staff, defaults={"dataset_folder": folder_rel})
    folder = Path(settings.MEDIA_ROOT) / enrollment.dataset_folder
    folder.mkdir(parents=True, exist_ok=True)
    engine = get_face_engine()
    required = min_enrollment_images()

    wanted: set[str] = set()
    rejected: list[str] = []
    for rel in paths:
        name = f"{STAFF_PHOTO_PREFIX}{hashlib.sha1(rel.encode('utf-8')).hexdigest()[:12]}.jpg"
        dest = folder / name
        if dest.exists():
            wanted.add(name)
            continue
        image = cv2.imread(str(Path(settings.MEDIA_ROOT) / rel))
        if image is None:
            rejected.append(f"{rel}: unreadable")
            continue
        quality = engine.check_quality(image)
        save_image = quality.pop("image", None)
        if not quality.get("passed"):
            rejected.append(f"{rel}: {quality.get('message') or 'quality check failed'}")
            continue
        cv2.imwrite(str(dest), save_image if save_image is not None else engine.resize_max(image, 640))
        wanted.add(name)
        if not enrollment.profile_image:
            enrollment.profile_image = engine.save_profile_image(
                save_image if save_image is not None else image, staff.pk
            )
            enrollment.save(update_fields=["profile_image", "updated_at"])

    for old in folder.glob(f"{STAFF_PHOTO_PREFIX}*.jpg"):
        if old.name not in wanted:
            old.unlink(missing_ok=True)  # photo was removed from the staff record

    total = len(list(folder.glob("*.jpg")))
    with transaction.atomic():
        enrollment = FaceEnrollment.objects.select_for_update().get(pk=enrollment.pk)
        enrollment.total_images = total
        enrollment.save(update_fields=["total_images", "updated_at"])

    if total < required:
        if enrollment.is_trained or enrollment.is_enrolled:
            enrollment.is_trained = False
            enrollment.is_enrolled = False
            enrollment.embedding = None
            enrollment.save(update_fields=["is_trained", "is_enrolled", "embedding", "updated_at"])
            invalidate_gallery()
        logger.warning(
            "Attendance enrollment for staff %s (%s) needs %d usable photos, has %d. Rejected: %s",
            staff.pk, staff.full_name, required, total, "; ".join(rejected) or "none",
        )
        return {"trained": False, "usable_photos": total, "required": required, "rejected": rejected}

    result = engine.train_enrollment(enrollment, folder, required=required)
    if result.get("trained") and not staff.face_identity_label:
        type(staff).objects.filter(pk=staff.pk).update(face_identity_label=staff.full_name)
    logger.info(
        "Attendance enrollment from staff photos: staff %s (%s) trained=%s photos=%d",
        staff.pk, staff.full_name, result.get("trained"), total,
    )
    return {**result, "usable_photos": total, "required": required, "rejected": rejected}
