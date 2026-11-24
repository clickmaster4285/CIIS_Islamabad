"""Camera offline monitor: marks cameras Offline, opens/closes admin alerts, scopes the API."""

from unittest import mock

from django.test import TestCase, override_settings
from rest_framework.authtoken.models import Token
from rest_framework.test import APIClient

from cameras import offline_monitor
from cameras.models import Camera, CameraOfflineAlert, CameraStatus, Nvr, Site
from users.models import User

BASE = "http://ml-node:8100"


def _user(username, role, location=""):
    return User.objects.create_user(
        username=username, password="pass12345", role=role, phone="0300", location=location
    )


@override_settings(CAMERA_OFFLINE_GRACE_SEC=60)
class OfflineMonitorTests(TestCase):
    def setUp(self):
        site = Site.objects.create(code="PEW-T", name="Peshawar test")
        nvr = Nvr.objects.create(site=site, name="NVR T", ip_address="10.0.0.9", password="x")
        self.cam = Camera.objects.create(
            nvr=nvr, name="Gate", code="gate-t", channel=1, location="PESHAWAR", status=CameraStatus.ONLINE
        )
        offline_monitor._first_bad.clear()
        self.stream = {"key": self.cam.stream_key, "connected": True, "has_frame": True}
        patcher = mock.patch("ml.client.camera_ml_base_url", return_value=BASE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, streams, clock):
        with mock.patch.object(offline_monitor, "_node_streams", return_value=streams), mock.patch.object(
            offline_monitor.time, "monotonic", return_value=clock
        ), mock.patch.object(offline_monitor, "_notify"):
            return offline_monitor.check_once()

    def test_goes_offline_after_grace_then_recovers(self):
        self._run({}, 1000.0)  # first miss: inside grace, nothing yet
        self.cam.refresh_from_db()
        self.assertEqual(self.cam.status, CameraStatus.ONLINE)
        self.assertFalse(CameraOfflineAlert.objects.exists())

        out = self._run({}, 1061.0)  # still missing after 61 s
        self.cam.refresh_from_db()
        self.assertEqual(self.cam.status, CameraStatus.OFFLINE)
        self.assertEqual(out["went_offline"], 1)
        alert = CameraOfflineAlert.objects.get()
        self.assertTrue(alert.is_open)
        self.assertIn("Not streaming", alert.reason)

        self._run({}, 1200.0)  # stays offline: still one alert
        self.assertEqual(CameraOfflineAlert.objects.count(), 1)

        out = self._run({self.cam.stream_key: self.stream}, 1300.0)  # back
        self.cam.refresh_from_db()
        alert.refresh_from_db()
        self.assertEqual(self.cam.status, CameraStatus.ONLINE)
        self.assertFalse(alert.is_open)
        self.assertIsNotNone(alert.back_online_at)
        self.assertEqual(out["came_back"], 1)

    def test_disconnected_stream_and_unreachable_server_are_offline(self):
        self._run({self.cam.stream_key: {**self.stream, "connected": False}}, 0.0)
        self._run({self.cam.stream_key: {**self.stream, "connected": False}}, 100.0)
        self.assertIn("disconnected", CameraOfflineAlert.objects.get().reason)
        self._run(None, 200.0)  # ML server down: reason updated, no second alert
        alert = CameraOfflineAlert.objects.get()
        self.assertIn("not reachable", alert.reason)

    def test_short_blip_does_not_alert(self):
        self._run({}, 0.0)
        self._run({self.cam.stream_key: self.stream}, 20.0)
        self._run({}, 30.0)  # new miss restarts the grace window
        self.assertFalse(CameraOfflineAlert.objects.exists())

    def test_unassigned_camera_is_ignored(self):
        with mock.patch("ml.client.camera_ml_base_url", return_value=""):
            self._run({}, 0.0)
            self._run({}, 1000.0)
        self.assertFalse(CameraOfflineAlert.objects.exists())


class OfflineAlertApiTests(TestCase):
    def setUp(self):
        site = Site.objects.create(code="PEW-A", name="Peshawar api")
        nvr = Nvr.objects.create(site=site, name="NVR A", ip_address="10.0.0.8", password="x")
        self.pew = Camera.objects.create(nvr=nvr, name="Pesh cam", code="p1", channel=1, location="PESHAWAR")
        self.koh = Camera.objects.create(nvr=nvr, name="Kohat cam", code="k1", channel=2, location="KOHAT")
        from django.utils import timezone

        now = timezone.now()
        self.a_pew = CameraOfflineAlert.objects.create(camera=self.pew, location="PESHAWAR", went_offline_at=now)
        CameraOfflineAlert.objects.create(camera=self.koh, location="KOHAT", went_offline_at=now)

    def _client(self, user):
        c = APIClient()
        c.credentials(HTTP_AUTHORIZATION=f"Token {Token.objects.create(user=user).key}")
        return c

    def test_super_admin_sees_all(self):
        res = self._client(_user("boss", "ADMIN")).get("/api/cameras/offline-alerts/?scope=open")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["open_count"], 2)

    def test_city_location_admin_sees_own_site_only(self):
        # (A Peshawar/HQ admin sees every city, like the camera wall; a city admin only their own.)
        res = self._client(_user("la", "LOCATION_ADMIN", "KOHAT")).get("/api/cameras/offline-alerts/?scope=open")
        self.assertEqual(res.status_code, 200)
        names = [r["camera_name"] for r in res.json()["results"]]
        self.assertEqual(names, ["Kohat cam"])

    def test_other_roles_are_refused(self):
        res = self._client(_user("guard", "GUARD", "PESHAWAR")).get("/api/cameras/offline-alerts/")
        self.assertEqual(res.status_code, 403)

    def test_acknowledge(self):
        client = self._client(_user("boss2", "ADMIN"))
        res = client.post(f"/api/cameras/offline-alerts/{self.a_pew.pk}/acknowledge/")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["acknowledged"])
