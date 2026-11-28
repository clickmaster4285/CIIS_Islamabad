"""Attendance (InsightFace) enrollment is built straight from the Add/Edit Staff photos."""

import shutil
import tempfile
from pathlib import Path

from django.conf import settings
from django.test import TestCase, override_settings

from recognition.models import FaceEnrollment
from recognition.services.training import STAFF_PHOTO_PREFIX, enroll_from_staff_photos
from users.models import Staff

# Real face photos already on this machine (any trained enrollment folder works).
SOURCE = Path(settings.MEDIA_ROOT) / "dataset" / "staff_128"


@override_settings(ATTENDANCE_MIN_ENROLLMENT_IMAGES=5)
class StaffPhotoEnrollmentTests(TestCase):
    def setUp(self):
        photos = sorted(SOURCE.glob("*.jpg"))[:5] if SOURCE.is_dir() else []
        if len(photos) < 5:
            self.skipTest("needs 5 real face photos in media/dataset/staff_128")
        self.media = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.media, ignore_errors=True)
        (self.media / "staff" / "photos").mkdir(parents=True)
        self.rel = []
        for i, p in enumerate(photos):
            rel = f"staff/photos/p{i}.jpg"
            shutil.copy(p, self.media / rel)
            self.rel.append(rel)
        override = override_settings(MEDIA_ROOT=str(self.media))
        override.enable()
        self.addCleanup(override.disable)

    def _staff(self, photos):
        staff = Staff.objects.create(full_name="Photo Test")
        Staff.objects.filter(pk=staff.pk).update(staff_photos=photos)
        return Staff.objects.get(pk=staff.pk)

    def test_trains_insightface_enrollment_from_staff_photos(self):
        staff = self._staff(self.rel)
        result = enroll_from_staff_photos(staff)
        self.assertTrue(result["trained"], result)
        enrollment = FaceEnrollment.objects.get(staff=staff)
        self.assertTrue(enrollment.is_trained and enrollment.is_enrolled)
        self.assertEqual(len(enrollment.embedding), 512)  # ArcFace, not 128-d SFace
        folder = self.media / enrollment.dataset_folder
        self.assertEqual(len(list(folder.glob(f"{STAFF_PHOTO_PREFIX}*.jpg"))), 5)

    def test_too_few_photos_stays_untrained_and_removed_photos_are_dropped(self):
        staff = self._staff(self.rel)
        self.assertTrue(enroll_from_staff_photos(staff)["trained"])
        Staff.objects.filter(pk=staff.pk).update(staff_photos=self.rel[:2])
        result = enroll_from_staff_photos(Staff.objects.get(pk=staff.pk))
        self.assertFalse(result["trained"])
        enrollment = FaceEnrollment.objects.get(staff=staff)
        self.assertFalse(enrollment.is_trained)
        self.assertEqual(enrollment.total_images, 2)
