"""Role assignment allow-list, account rank, staff link and HR workflow guards."""

import shutil
import tempfile
from datetime import date
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from users.models import LeaveRequest, LeaveType, PayrollRun, Staff
from users.permissions import location_admin_may_assign_role, may_manage_account

User = get_user_model()
PASSWORD = "S3cure-pass-123"


def _user(username, role, location="ISLAMABAD", **extra):
    user = User(username=username, role=role, location=location, is_active=True, phone="0300", **extra)
    user.set_password(PASSWORD)
    user.save()
    return user


class RoleAllowListTests(TestCase):
    def setUp(self):
        self.hr = _user("hr1", "HR")
        self.loc_admin = _user("loc1", "LOCATION_ADMIN")
        self.boss = _user("boss", "ADMIN", location="")
        self.client = APIClient()

    def _create(self, actor, role):
        self.client.force_authenticate(actor)
        return self.client.post(
            "/api/users/",
            {"username": f"new_{role.lower()}_{actor.username}", "password": PASSWORD, "full_name": "New Person",
             "role": role, "phone": "0300", "location": "ISLAMABAD"},
            format="json",
        )

    def test_hr_may_only_assign_low_roles(self):
        for role in ("ADMIN", "IT_SUPERADMIN", "LOCATION_ADMIN", "IT_ADMIN", "COLLECTOR",
                     "DEPUTY_COLLECTOR", "ASSISTANT_COLLECTOR", "HR", "OPERATION_MANAGER", "SUPPORT"):
            self.assertEqual(self._create(self.hr, role).status_code, 400, role)
        self.assertEqual(self._create(self.hr, "GUARD").status_code, 201)

    def test_location_admin_may_staff_hr_but_not_cross_city_roles(self):
        self.assertEqual(self._create(self.loc_admin, "HR").status_code, 201)
        for role in ("COLLECTOR", "IT_ADMIN", "LOCATION_ADMIN"):
            self.assertEqual(self._create(self.loc_admin, role).status_code, 400, role)

    def test_super_admin_may_assign_anything(self):
        self.assertTrue(location_admin_may_assign_role(self.boss, "IT_SUPERADMIN"))
        self.assertEqual(self._create(self.boss, "COLLECTOR").status_code, 201)

    def test_nobody_non_global_changes_own_role(self):
        guard_with_hr = _user("hrmod", "GUARD", allowed_modules=["Human Resource"])
        self.client.force_authenticate(guard_with_hr)
        resp = self.client.patch(f"/api/users/{guard_with_hr.pk}/", {"role": "INSPECTOR"}, format="json")
        self.assertEqual(resp.status_code, 400)
        guard_with_hr.refresh_from_db()
        self.assertEqual(guard_with_hr.role, "GUARD")

    def test_password_reset_only_at_or_below_own_level(self):
        it_admin = _user("itadm", "IT_ADMIN")
        guard = _user("g1", "GUARD")
        self.assertFalse(may_manage_account(self.hr, it_admin))
        self.client.force_authenticate(self.hr)
        resp = self.client.patch(f"/api/users/{it_admin.pk}/", {"password": "Hacked-pass-999"}, format="json")
        self.assertEqual(resp.status_code, 403)
        it_admin.refresh_from_db()
        self.assertTrue(it_admin.check_password(PASSWORD))
        resp = self.client.patch(f"/api/users/{guard.pk}/", {"password": "Fresh-Lamp-4821"}, format="json")
        self.assertEqual(resp.status_code, 200)

    def test_hr_cannot_delete_higher_account(self):
        it_admin = _user("itadm2", "IT_ADMIN")
        self.client.force_authenticate(self.hr)
        self.assertEqual(self.client.delete(f"/api/users/{it_admin.pk}/").status_code, 403)


class StaffLinkGuardTests(TestCase):
    def setUp(self):
        self.hr = _user("hr2", "HR")
        self.it_admin = _user("itadm3", "IT_ADMIN")
        self.staff = Staff.objects.create(full_name="Linked Person", user=self.it_admin)
        self.client = APIClient()
        self.client.force_authenticate(self.hr)

    def test_user_is_read_only_on_staff_update(self):
        other = _user("g2", "GUARD")
        resp = self.client.patch(f"/api/staff/{self.staff.pk}/", {"user": other.pk}, format="json")
        self.assertEqual(resp.status_code, 200)
        self.staff.refresh_from_db()
        self.assertEqual(self.staff.user_id, self.it_admin.pk)

    def test_unlink_cannot_deactivate_higher_account(self):
        resp = self.client.post(
            f"/api/staff/{self.staff.pk}/unlink_user/", {"deactivate_user": True}, format="json"
        )
        self.assertEqual(resp.status_code, 403)
        self.it_admin.refresh_from_db()
        self.assertTrue(self.it_admin.is_active)
        self.staff.refresh_from_db()
        self.assertEqual(self.staff.user_id, self.it_admin.pk)

    def test_delete_staff_cannot_deactivate_higher_account(self):
        self.assertEqual(self.client.delete(f"/api/staff/{self.staff.pk}/").status_code, 403)
        self.assertTrue(Staff.objects.filter(pk=self.staff.pk).exists())


class LeaveAndPayrollGuardTests(TestCase):
    def setUp(self):
        self.hr = _user("hr3", "HR")
        self.hr_staff = Staff.objects.create(full_name="HR Officer", user=self.hr)
        self.leave_type, _ = LeaveType.objects.get_or_create(code="CASUAL", defaults={"name": "Casual"})
        self.client = APIClient()
        self.client.force_authenticate(self.hr)

    def test_no_self_approval(self):
        req = LeaveRequest.objects.create(
            staff=self.hr_staff, leave_type=self.leave_type, from_date=date(2026, 1, 1), to_date=date(2026, 1, 2), days=2
        )
        resp = self.client.post(f"/api/leave-requests/{req.pk}/approve/")
        self.assertEqual(resp.status_code, 403)
        req.refresh_from_db()
        self.assertEqual(req.status, "PENDING")

    def test_status_cannot_be_patched_on_leave(self):
        other = Staff.objects.create(full_name="Other", user=_user("g3", "GUARD"))
        req = LeaveRequest.objects.create(
            staff=other, leave_type=self.leave_type, from_date=date(2026, 1, 1), to_date=date(2026, 1, 2), days=2
        )
        self.client.patch(f"/api/leave-requests/{req.pk}/", {"status": "APPROVED"}, format="json")
        req.refresh_from_db()
        self.assertEqual(req.status, "PENDING")

    def test_locked_payroll_cannot_go_back_or_change_totals(self):
        run = PayrollRun.objects.create(
            period_label="Jan", period_start=date(2026, 1, 1), period_end=date(2026, 1, 31),
            status="LOCKED", created_by=self.hr, total_net=100,
        )
        resp = self.client.patch(f"/api/payroll-runs/{run.pk}/", {"status": "DRAFT"}, format="json")
        self.assertEqual(resp.status_code, 400)
        self.client.patch(f"/api/payroll-runs/{run.pk}/", {"total_net": "999"}, format="json")
        run.refresh_from_db()
        self.assertEqual(run.status, "LOCKED")
        self.assertEqual(run.total_net, 100)


class StaffPhotoKeepTests(TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "staff").mkdir()
        for name in ("mine.jpg", "theirs.jpg"):
            (self.root / "staff" / name).write_bytes(b"x")

    def test_foreign_keep_path_is_ignored_and_never_deleted(self):
        from types import SimpleNamespace

        from django.utils.datastructures import MultiValueDict

        from users.staff_photos import apply_staff_photo_uploads

        with override_settings(MEDIA_ROOT=str(self.root)):
            mine = Staff.objects.create(full_name="Me", staff_photos=["staff/mine.jpg"])
            Staff.objects.create(full_name="Them", staff_photos=["staff/theirs.jpg"])
            request = SimpleNamespace(
                data={"staff_photos_keep": '["staff/theirs.jpg"]'}, FILES=MultiValueDict()
            )
            apply_staff_photo_uploads(mine, request)
            mine.refresh_from_db()
            self.assertNotIn("staff/theirs.jpg", mine.staff_photos)
            self.assertTrue((self.root / "staff" / "theirs.jpg").exists())


class MediaDispositionTests(TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "detections").mkdir()
        (self.root / "detections" / "a.jpg").write_bytes(b"\xff\xd8\xff")
        (self.root / "detections" / "x.html").write_bytes(b"<script>alert(1)</script>")
        (self.root / "detections" / "y.svg").write_bytes(b"<svg/>")
        self.client = APIClient()
        from users.token_auth import issue_token

        self.token = issue_token(_user("viewer", "GUARD")).key

    def _get(self, path):
        with self.settings(MEDIA_ROOT=str(self.root)):
            return self.client.get(f"/media/{path}", HTTP_AUTHORIZATION=f"Token {self.token}")

    def test_images_inline_everything_else_downloads(self):
        img = self._get("detections/a.jpg")
        self.assertEqual(img.status_code, 200)
        self.assertTrue(img["Content-Disposition"].startswith("inline"))
        for path in ("detections/x.html", "detections/y.svg"):
            res = self._get(path)
            self.assertTrue(res["Content-Disposition"].startswith("attachment"), path)
            self.assertEqual(res["X-Content-Type-Options"], "nosniff")
            self.assertEqual(res["Content-Security-Policy"], "sandbox")

    def test_recording_manifest_is_never_served(self):
        (self.root / "distribution" / "d1").mkdir(parents=True)
        (self.root / "distribution" / "d1" / "recording_manifest.json").write_text("{}")
        self.assertEqual(self._get("distribution/d1/recording_manifest.json").status_code, 403)
