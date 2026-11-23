"""Run: python -m unittest test_identity_counters  (from ml_services/)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import object_identity
from object_identity import ObjectIdentityRegistry, _load_counters


class CounterFileTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="identity-"))

    def test_corrupt_file_does_not_reset_to_zero(self):
        path = self.dir / "c.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertLogs("ml.identity", level="ERROR"):
            reg = ObjectIdentityRegistry(counter_file=path)
        self.assertEqual(reg._next_code("person"), "GP1001")  # floor, never GP1
        self.assertTrue(any(p.name.startswith("c.json.corrupt-") for p in self.dir.iterdir()))

    def test_legacy_file_seeds_new_per_node_file(self):
        legacy = self.dir / "identity_counters.json"
        legacy.write_text(json.dumps({"person": 5000, "vehicle": 1200, "object": 900}), encoding="utf-8")
        node = self.dir / "identity_counters.gpu1.json"
        got = _load_counters(node, legacy=legacy)
        self.assertEqual(got, {"person": 5000, "vehicle": 1200, "object": 1000})

    def test_stride_interleaves_codes_between_nodes(self):
        env0 = {"ML_IDENTITY_CODE_STRIDE": "2", "ML_IDENTITY_CODE_OFFSET": "0"}
        env1 = {"ML_IDENTITY_CODE_STRIDE": "2", "ML_IDENTITY_CODE_OFFSET": "1"}
        with mock.patch.dict("os.environ", env0):
            a = ObjectIdentityRegistry(counter_file=self.dir / "a.json")
        with mock.patch.dict("os.environ", env1):
            b = ObjectIdentityRegistry(counter_file=self.dir / "b.json")
        codes_a = {a._next_code("person") for _ in range(5)}
        codes_b = {b._next_code("person") for _ in range(5)}
        self.assertFalse(codes_a & codes_b)
        self.assertTrue(all(int(c[2:]) % 2 == 0 for c in codes_a))
        self.assertTrue(all(int(c[2:]) % 2 == 1 for c in codes_b))

    def test_save_leaves_no_temp_files(self):
        path = self.dir / "c.json"
        reg = ObjectIdentityRegistry(counter_file=path)
        for _ in range(3):
            reg._next_code("vehicle")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["vehicle"], 1003)
        self.assertEqual([p.name for p in self.dir.iterdir()], ["c.json"])


class GalleryMatrixTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.codes = []
        p = mock.patch.object(object_identity, "extract_reid_embeddings",
                              side_effect=lambda crops: [self.codes.pop(0) for _ in crops])
        p.start()
        self.addCleanup(p.stop)
        self.reg = ObjectIdentityRegistry(counter_file=f"{tempfile.mkdtemp()}/c.json")

    def det(self, tid, cls="person"):
        return {"class_name": cls, "label": cls, "model": "coco", "confidence": 0.9, "track_id": tid,
                "bbox": [10, 10, 60, 90]}

    def test_reid_match_through_matrix(self):
        v = [1.0] + [0.0] * 7
        w = [0.0, 1.0] + [0.0] * 6
        self.codes = [v]
        first = self.reg.enrich_detections("cam-1", self.frame, [self.det(1)])[0]
        # Same appearance on another camera / new track → same global id
        self.codes = [list(v)]
        again = self.reg.enrich_detections("cam-2", self.frame, [self.det(9)])[0]
        self.assertEqual(again["global_object_id"], first["global_object_id"])
        # Different appearance → new id
        self.codes = [w]
        other = self.reg.enrich_detections("cam-2", self.frame, [self.det(10)])[0]
        self.assertNotEqual(other["global_object_id"], first["global_object_id"])

    def test_class_mismatch_not_matched(self):
        v = [1.0] + [0.0] * 7
        self.codes = [v]
        car = self.reg.enrich_detections("cam-1", self.frame, [self.det(1, "car")])[0]
        self.codes = [list(v)]
        truck = self.reg.enrich_detections("cam-1", self.frame, [self.det(2, "truck")])[0]
        self.assertNotEqual(car["global_object_id"], truck["global_object_id"])

    def test_pruned_entries_leave_matrix(self):
        v = [1.0] + [0.0] * 7
        self.codes = [v]
        self.reg.enrich_detections("cam-1", self.frame, [self.det(1)])
        gid = next(iter(self.reg._globals))
        self.reg._globals[gid].last_seen -= object_identity.GALLERY_TTL_SEC + 10
        self.reg._last_prune = 0.0
        self.reg._prune(object_identity.time.time())
        self.assertNotIn(gid, self.reg._globals)
        self.assertNotIn(gid, self.reg._gallery_of)
        self.assertEqual(sum(len(g.gids) for g in self.reg._galleries.values()), 0)

    def test_forget_camera_drops_binds(self):
        self.codes = [[1.0] + [0.0] * 7]
        self.reg.enrich_detections("cam-1", self.frame, [self.det(4)])
        self.assertIn(("cam-1", 4), self.reg._track_bind)
        self.reg.forget_camera("cam-1")
        self.assertNotIn(("cam-1", 4), self.reg._track_bind)


if __name__ == "__main__":
    unittest.main()
