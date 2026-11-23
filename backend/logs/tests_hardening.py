from datetime import timedelta
from io import StringIO

from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from logs.enforcement import clamp_battery_level, remember_event_id
from logs.models import MobileAlert, MobileEventReceipt, UserActivityLog
from users.models import User


def _user(username, role="INSPECTOR", location="PESHAWAR"):
    return User.objects.create_user(
        username=username, password="pass12345", role=role, phone="0300", location=location
    )


class ActivityLogVolumeTests(TestCase):
    def setUp(self):
        self.user = _user("reader", role="ADMIN", location="")
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def test_get_polls_are_not_logged(self):
        before = UserActivityLog.objects.count()
        self.client.get("/api/activity-logs/")
        self.assertEqual(UserActivityLog.objects.count(), before)

    def test_client_report_is_marked(self):
        self.client.post("/api/activity-logs/report/", {"action": "LOGIN", "source": "evil"}, format="json")
        row = UserActivityLog.objects.latest("pk")
        self.assertEqual(row.action, "[client] LOGIN")
        self.assertEqual(row.source, "web")


class ReceiptAndBatteryTests(TestCase):
    def test_event_ids_are_per_user(self):
        a, b = _user("a1"), _user("b1")
        self.assertTrue(remember_event_id(a, "e-1", "gps"))
        self.assertFalse(remember_event_id(a, "e-1", "gps"))
        self.assertTrue(remember_event_id(b, "e-1", "gps"))

    def test_battery_is_clamped(self):
        self.assertEqual(clamp_battery_level("150"), 100)
        self.assertEqual(clamp_battery_level(-5), 0)
        self.assertEqual(clamp_battery_level(0.42), 42)
        self.assertIsNone(clamp_battery_level("full"))

    def test_purge_command(self):
        user = _user("p1")
        old = UserActivityLog.objects.create(user=user, action="old")
        UserActivityLog.objects.filter(pk=old.pk).update(time=timezone.now() - timedelta(days=200))
        UserActivityLog.objects.create(user=user, action="new")
        receipt = MobileEventReceipt.objects.create(user=user, event_id="r1", event_type="gps")
        MobileEventReceipt.objects.filter(pk=receipt.pk).update(created_at=timezone.now() - timedelta(days=60))
        call_command("purge_activity_logs", stdout=StringIO())
        self.assertEqual(list(UserActivityLog.objects.values_list("action", flat=True)), ["new"])
        self.assertFalse(MobileEventReceipt.objects.exists())


class AlertAckScopeTests(TestCase):
    def test_location_admin_cannot_ack_other_city_alert(self):
        kohat_user = _user("k1", location="KOHAT")
        alert = MobileAlert.objects.create(
            user=kohat_user, alert_type="GPS_OFFLINE", message="x", detected_at=timezone.now()
        )
        client = APIClient()
        client.force_authenticate(_user("pesh_admin", role="LOCATION_ADMIN", location="PESHAWAR"))
        self.assertEqual(client.post(f"/api/mobile-alerts/{alert.pk}/acknowledge/").status_code, 404)
        alert.refresh_from_db()
        self.assertTrue(alert.is_active)
