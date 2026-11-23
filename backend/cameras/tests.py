from django.test import TestCase

from .models import Camera, Nvr, Site


class CameraPassageRoleTests(TestCase):
    def test_camera_creation_defaults_passage_role_to_empty_string(self):
        site = Site.objects.create(code="TEST", name="Test Site")
        nvr = Nvr.objects.create(site=site, name="Test NVR", ip_address="127.0.0.1")

        camera = Camera.objects.create(
            nvr=nvr,
            channel=1,
            name="Entry Camera",
            location=site.code,
        )

        self.assertEqual(camera.passage_role, "")


class AuditHardeningTests(TestCase):
    def test_redact_strips_credentials_and_keys(self):
        from ml.client import MLServiceError, public_error_message, redact

        text = redact("url: /live/cam/cam-1/jpeg?key=SECRET&w=1 rtsp://admin:pw@10.0.0.1:554/x")
        self.assertNotIn("SECRET", text)
        self.assertNotIn("admin:pw", text)
        exc = MLServiceError("ML service unreachable at http://u:p@h:8000 (boom)", 503)
        self.assertNotIn("u:p", str(exc))
        self.assertEqual(public_error_message(exc), "Detection engine temporarily unavailable.")

    def test_nvr_host_validation(self):
        from cameras.access import validate_nvr_host

        self.assertEqual(validate_nvr_host(" 192.168.1.10 "), "192.168.1.10")
        self.assertEqual(validate_nvr_host("NVR-1.local"), "nvr-1.local")
        for bad in ("rtsp://10.0.0.1", "admin@10.0.0.1", "10.0.0.1/path", "10.0.0.1:554", "a b"):
            with self.assertRaises(ValueError):
                validate_nvr_host(bad)

    def test_nvr_serializer_rejects_bad_template(self):
        from cameras.serializers import NvrSerializer

        site = Site.objects.create(code="TPL", name="Tpl")
        data = {"site": site.pk, "name": "N", "ip_address": "10.0.0.2", "stream_path_template": "/x/{foo}"}
        serializer = NvrSerializer(data=data)
        self.assertFalse(serializer.is_valid())
        self.assertIn("stream_path_template", serializer.errors)

    def test_bad_saved_template_falls_back(self):
        from cameras.rtsp_utils import nvr_stream_path

        site = Site.objects.create(code="TPL2", name="Tpl2")
        nvr = Nvr.objects.create(site=site, name="N", ip_address="10.0.0.3", stream_path_template="/x/{foo}")
        self.assertEqual(nvr_stream_path(nvr, 1), "/Streaming/Channels/101")

    def test_plate_dates_mix_aware_and_naive(self):
        from cameras.plate_captures import _filter_rows_by_date

        rows = [{"timestamp": "2026-11-20T10:00:00"}, {"timestamp": "2026-11-22T10:00:00"}]
        out = _filter_rows_by_date(rows, "2026-11-21T00:00:00Z", "")
        self.assertEqual([r["timestamp"] for r in out], ["2026-11-22T10:00:00"])

    def test_file_lock_is_exclusive_and_released(self):
        import tempfile
        from pathlib import Path

        from cameras.locks import FileLockTimeout, exclusive_file_lock

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "captures.csv"
            with exclusive_file_lock(target) as lock_path:
                self.assertTrue(lock_path.is_file())
                with self.assertRaises(FileLockTimeout):
                    with exclusive_file_lock(target, timeout=0.2):
                        pass
            self.assertFalse(Path(f"{target}.lock").exists())

    def test_job_ids_are_validated(self):
        from ml import job_store

        self.assertTrue(job_store.is_valid_job_id("0123456789ab"))
        for bad in ("../../etc", "0123456789AB", "0123456789abc", ""):
            self.assertFalse(job_store.is_valid_job_id(bad))
