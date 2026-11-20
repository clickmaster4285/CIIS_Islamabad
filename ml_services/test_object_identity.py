"""Run: python -m unittest test_object_identity  (from ml_services/)."""

import unittest
from unittest import mock

import numpy as np

import object_identity
from object_identity import ObjectIdentityRegistry


def det(class_name: str, model: str = "coco", tid: int = 1) -> dict:
    return {
        "class_name": class_name,
        "label": class_name,
        "model": model,
        "confidence": 0.9,
        "track_id": tid,
        "bbox": [10, 10, 60, 60],
    }


class DangerDetectionPassThroughTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        import tempfile

        self.tmp = tempfile.mkdtemp(prefix="identity-")
        self.reg = ObjectIdentityRegistry(counter_file=f"{self.tmp}/counters.json")
        # Keep ReID model out of the test; identity objects get no embedding.
        patcher = mock.patch.object(
            object_identity, "extract_reid_embeddings", side_effect=lambda crops: [[] for _ in crops]
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_enrich(self, dets):
        return self.reg.enrich_detections("cam-1", self.frame, dets)

    def test_weapon_fire_smoke_are_kept(self):
        dets = [
            det("gun", model="weapon", tid=1),
            det("pistol", tid=2),
            det("fire", model="smoke", tid=3),
            det("smoke", model="smoke", tid=4),
            det("flame", tid=5),
        ]
        out = self.run_enrich(dets)
        self.assertEqual([d["class_name"] for d in out], ["gun", "pistol", "fire", "smoke", "flame"])

    def test_danger_detections_get_no_identity(self):
        out = self.run_enrich([det("gun", model="weapon")])
        self.assertNotIn("global_object_id", out[0])
        self.assertNotIn("reid_embedding", out[0])

    def test_normal_object_still_gets_identity(self):
        out = self.run_enrich([det("car", tid=7)])
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].get("global_object_id"))


class PersistentCounterTests(unittest.TestCase):
    def test_codes_never_repeat_after_restart(self):
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp(prefix="identity-")) / "counters.json"
        first = ObjectIdentityRegistry(counter_file=path)
        self.assertEqual(first._next_code("person"), "GP1001")  # new file starts above old codes
        self.assertEqual(first._next_code("person"), "GP1002")
        self.assertEqual(first._next_code("vehicle"), "GV1001")
        restarted = ObjectIdentityRegistry(counter_file=path)
        self.assertEqual(restarted._next_code("person"), "GP1003")
        self.assertEqual(restarted._next_code("vehicle"), "GV1002")


if __name__ == "__main__":
    unittest.main()
