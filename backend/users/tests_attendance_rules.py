"""Night shifts, half-day threshold, note merging, duplicate-row merge, exit-camera deletes."""

from datetime import datetime, time, timedelta

from django.test import TestCase, override_settings
from django.utils import timezone

from users.attendance_service import AttendanceDecisionEngine as Engine
from users.models import Attendance, Staff, User


def _at(y, m, d, hh, mm):
    return timezone.make_aware(datetime(y, m, d, hh, mm))


@override_settings(ATTENDANCE_WORK_START="09:00", ATTENDANCE_LATE_AFTER="09:30", ATTENDANCE_WORK_END="17:00")
class AttendanceRuleTests(TestCase):
    def setUp(self):
        self.night = Staff.objects.create(
            full_name="Night Guard", work_shift_start=time(22, 0), work_shift_end=time(6, 0)
        )
        self.day = Staff.objects.create(full_name="Day Clerk")

    def test_night_shift_late_check_crosses_midnight(self):
        self.assertTrue(Engine.is_on_time(_at(2026, 7, 21, 21, 50), self.night))
        self.assertTrue(Engine.is_on_time(_at(2026, 7, 21, 22, 20), self.night))
        self.assertFalse(Engine.is_on_time(_at(2026, 7, 22, 0, 30), self.night))

    def test_half_day_from_shift_and_notes_kept(self):
        self.assertEqual(Engine.half_day_hours(self.night), 4.0)
        record = Attendance(
            staff=self.day, date=_at(2026, 7, 21, 9, 0).date(), check_in=_at(2026, 7, 21, 9, 0),
            check_out=_at(2026, 7, 21, 18, 0), notes="Visited bank",
        )
        Engine.apply_checkout_rules(record, self.day, record.check_out, zone="exit")
        self.assertEqual(record.notes, "Visited bank; Overtime")
        Engine.apply_checkout_rules(record, self.day, record.check_out, zone="exit")
        self.assertEqual(record.notes, "Visited bank; Overtime")

    @override_settings(ATTENDANCE_HALF_DAY_HOURS="2")
    def test_half_day_setting_wins(self):
        self.assertEqual(Engine.half_day_hours(self.night), 2.0)

    def test_exit_camera_keeps_existing_empty_row(self):
        now = timezone.now()
        row = Attendance.objects.create(staff=self.day, date=timezone.localdate(now), notes="On leave")
        result = Engine.process_recognition(self.day, source=Attendance.SOURCE_CCTV, now=now, camera_zone="exit")
        self.assertEqual(result["action"], "ignored")
        self.assertTrue(Attendance.objects.filter(pk=row.pk).exists())

    def test_duplicate_user_and_staff_rows_are_merged(self):
        now = timezone.now()
        today = timezone.localdate(now)
        user = User.objects.create_user(username="clerk", password="pass12345", role="GUARD", phone="0")
        staff_row = Attendance.objects.create(staff=self.day, date=today)
        user_row = Attendance.objects.create(user=user, date=today, check_in=now - timedelta(hours=1))
        self.day.user = user
        self.day.save()
        record, created = Engine._get_or_create_today(self.day, Attendance.SOURCE_CCTV, now)
        self.assertFalse(created)
        self.assertEqual(record.pk, staff_row.pk)
        self.assertEqual(record.user_id, user.pk)
        self.assertIsNotNone(record.check_in)
        self.assertFalse(Attendance.objects.filter(pk=user_row.pk).exists())
