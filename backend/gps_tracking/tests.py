from __future__ import annotations

from datetime import timedelta

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from gps_tracking.models import OfficerGpsHistory, OfficerGpsLatest
from gps_tracking.serializers import GpsPingSerializer
from users.models import User


class GpsPingValidationTests(TestCase):
    def test_non_finite_and_out_of_range_values_are_refused(self):
        for field, value in (("latitude", "NaN"), ("longitude", "Infinity"), ("speedKmh", 5000), ("altitudeM", 99999)):
            data = {"latitude": 34.01234, "longitude": 71.56789, field: value}
            self.assertFalse(GpsPingSerializer(data=data).is_valid(), field)


class GpsPingViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="officer", password="x")
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def ping(self, recorded_at, event_id=None, **extra):
        body = {"latitude": 34.012345, "longitude": 71.567891, "accuracy": 8, "recordedAt": recorded_at.isoformat()}
        if event_id:
            body["event_id"] = event_id
        body.update(extra)
        return self.client.post("/api/gps/ping/", body, format="json")

    def test_far_future_refused_and_late_fix_does_not_move_marker(self):
        now = timezone.now()
        self.assertEqual(self.ping(now + timedelta(hours=2)).status_code, 400)
        self.assertEqual(self.ping(now).status_code, 200)
        self.assertEqual(self.ping(now - timedelta(hours=1), latitude=34.5).status_code, 200)
        latest = OfficerGpsLatest.objects.get(user=self.user)
        self.assertAlmostEqual(latest.latitude, 34.012345)
        self.assertEqual(OfficerGpsHistory.objects.filter(user=self.user).count(), 2)

    def test_refused_ping_does_not_burn_its_event_id(self):
        now = timezone.now()
        self.assertEqual(self.ping(now, event_id="e-1", accuracy=900).status_code, 400)
        resp = self.ping(now, event_id="e-1")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("duplicate", resp.json())
        self.assertTrue(self.ping(now, event_id="e-1").json().get("duplicate"))
