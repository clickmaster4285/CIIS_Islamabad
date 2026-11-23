"""Run: python -m unittest test_reid_refresh  (from ml_services/)."""

import tempfile
import unittest
from unittest import mock

import numpy as np

import object_identity
from object_identity import ObjectIdentityRegistry


def person(tid: int) -> dict:
    return {
        "class_name": "person",
        "label": "person",
        "model": "coco",
        "confidence": 0.9,
        "track_id": tid,
        "bbox": [10, 10, 60, 90],
    }


class ReidTrackRefreshTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.calls: list[int] = []
        self.n = 0

        def fake_codes(crops):
            self.calls.append(len(crops))
            out = []
            for _ in crops:
                self.n += 1
                v = np.zeros(8, dtype=np.float32)
                v[self.n % 8] = 1.0
                out.append([float(x) for x in v])
            return out

        p = mock.patch.object(object_identity, "extract_reid_embeddings", side_effect=fake_codes)
        p.start()
        self.addCleanup(p.stop)
        self.clock = [1000.0]
        t = mock.patch.object(object_identity.time, "time", side_effect=lambda: self.clock[0])
        t.start()
        self.addCleanup(t.stop)

    def registry(self, refresh: str) -> ObjectIdentityRegistry:
        with mock.patch.dict("os.environ", {"ML_REID_TRACK_REFRESH_SEC": refresh}):
            return ObjectIdentityRegistry(counter_file=f"{tempfile.mkdtemp()}/c.json")

    def test_off_computes_every_frame(self):
        reg = self.registry("0")
        for _ in range(3):
            reg.enrich_detections("cam", self.frame, [person(1)])
            self.clock[0] += 0.5
        self.assertEqual(self.calls, [1, 1, 1])

    def test_bound_track_reuses_code_until_refresh(self):
        reg = self.registry("2")
        first = reg.enrich_detections("cam", self.frame, [person(1)])[0]
        self.clock[0] += 0.5
        second = reg.enrich_detections("cam", self.frame, [person(1)])[0]
        self.assertEqual(self.calls, [1])  # second frame: no OSNet call
        self.assertEqual(second["reid_embedding"], first["reid_embedding"])  # backend still gets a code
        self.assertEqual(second["global_object_id"], first["global_object_id"])
        self.clock[0] += 2.0  # refresh due
        reg.enrich_detections("cam", self.frame, [person(1)])
        self.assertEqual(self.calls, [1, 1])

    def test_new_track_always_computed(self):
        reg = self.registry("5")
        reg.enrich_detections("cam", self.frame, [person(1)])
        self.clock[0] += 0.5
        reg.enrich_detections("cam", self.frame, [person(1), person(2)])
        self.assertEqual(self.calls, [1, 1])  # only track 2 computed

    def test_reused_code_not_folded_into_gallery(self):
        reg = self.registry("5")
        out = reg.enrich_detections("cam", self.frame, [person(1)])[0]
        with mock.patch.object(reg, "_update_reid", wraps=reg._update_reid) as upd:
            self.clock[0] += 0.5
            reg.enrich_detections("cam", self.frame, [person(1)])
            upd.assert_not_called()
        self.assertTrue(out["global_object_id"])


if __name__ == "__main__":
    unittest.main()
