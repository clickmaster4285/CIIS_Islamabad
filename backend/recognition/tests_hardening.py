"""Small unit checks for the recognition hardening (no DB, no models)."""

from types import SimpleNamespace

import numpy as np
from django.test import SimpleTestCase


class ClosestFaceTests(SimpleTestCase):
    def test_picks_overlapping_face_not_most_confident(self):
        from recognition.services.preprocess import closest_face

        same = SimpleNamespace(bbox=np.array([100, 100, 200, 200]), det_score=0.6)
        other = SimpleNamespace(bbox=np.array([400, 100, 500, 200]), det_score=0.99)
        self.assertIs(closest_face([other, same], [105, 98, 205, 202]), same)

    def test_no_overlap_returns_none(self):
        from recognition.services.preprocess import closest_face

        other = SimpleNamespace(bbox=np.array([400, 100, 500, 200]), det_score=0.99)
        self.assertIsNone(closest_face([other], [0, 0, 50, 50]))


class MatchConfirmerTests(SimpleTestCase):
    def test_same_frame_twice_is_one_hit(self):
        from recognition.services.match_confirm import MatchConfirmer

        confirmer = MatchConfirmer()
        self.assertFalse(confirmer.observe("webcam:1", needed=2, frame_hash="a"))
        self.assertFalse(confirmer.observe("webcam:1", needed=2, frame_hash="a"))
        self.assertTrue(confirmer.observe("webcam:1", needed=2, frame_hash="b"))


class RedactTests(SimpleTestCase):
    def test_credentials_removed(self):
        from recognition.services.cctv_worker import redact

        text = redact("GET /x?width=1&rtsp_url=rtsp%3A%2F%2Fadmin%3Apw%40h failed; rtsp://admin:pw@10.0.0.1/Channels/1")
        self.assertNotIn("pw", text)
        self.assertNotIn("admin", text)


class DecodeLimitsTests(SimpleTestCase):
    def test_oversized_image_rejected_before_decode(self):
        import io

        from PIL import Image

        from recognition.services.face_engine import FaceEngine

        buf = io.BytesIO()
        Image.new("RGB", (5000, 10)).save(buf, format="PNG")
        with self.assertRaises(ValueError):
            FaceEngine.decode_image(buf.getvalue())

    def test_garbage_rejected(self):
        from recognition.services.face_engine import FaceEngine

        with self.assertRaises(ValueError):
            FaceEngine.decode_image(b"not an image")
