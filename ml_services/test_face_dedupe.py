"""Run: python -m unittest test_face_dedupe  (from ml_services/).

Stage 2.1: the identity step reuses face pass 1 (parse_yolo_result) instead of running
face recognition again on the same box. Output must be identical; only the call count drops.
"""

import tempfile
import unittest
from unittest import mock

import numpy as np

import object_identity
from inference_engine import parse_yolo_result
from object_identity import ObjectIdentityRegistry, face_bbox_key, face_result_from_detection


class FakeFaceDB:
    """Deterministic stand-in for KnownFaceDB: identity depends only on the box."""

    def __init__(self):
        self.calls = 0

    def label_detection_detail(self, frame, cls_id, yolo_name, bbox):
        self.calls += 1
        x1 = int(bbox[0])
        if x1 < 100:
            return {"label": "Alice", "face_embedding": [0.1, 0.2], "face_match_score": 0.91, "face_match_method": "faiss"}
        if x1 < 300:
            return {"label": "unknown:U0007", "face_embedding": [0.3], "face_match_score": 0.4, "face_match_method": "unknown_cache"}
        return {"label": "unknown", "face_embedding": [], "face_match_score": None, "face_match_method": ""}


class _T:
    def __init__(self, v):
        self._v = v

    def __getitem__(self, i):
        return self

    def tolist(self):
        return list(self._v)

    def __float__(self):
        return float(self._v)

    def __int__(self):
        return int(self._v)


class _Box:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = _T(xyxy)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = {0: "person", 2: "car"}


def yolo_result():
    return _Result(
        [
            _Box(0, 0.9, [10, 10, 60, 120]),  # Alice
            _Box(0, 0.8, [150, 20, 200, 140]),  # unknown:U0007 (real UnknownFaceCache id format)
            _Box(0, 0.7, [400, 30, 450, 160]),  # no face
            _Box(2, 0.85, [500, 200, 620, 260]),  # car (no face work)
        ]
    )


class FaceDedupeTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((300, 700, 3), dtype=np.uint8)
        patcher = mock.patch.object(
            object_identity, "extract_reid_embeddings", side_effect=lambda crops: [[] for _ in crops]
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, dedupe: bool):
        db = FakeFaceDB()
        reg = ObjectIdentityRegistry(counter_file=f"{tempfile.mkdtemp(prefix='dedupe-')}/c.json")
        dets = parse_yolo_result(self.frame, yolo_result(), recognize_faces=True, model_tag="coco", face_db=db)
        pass1_calls = db.calls
        face_results = {}
        if dedupe:
            for d in dets:
                if d["class_id"] in (0, 80):
                    face_results[face_bbox_key(d["bbox"])] = face_result_from_detection(d)
        for i, d in enumerate(dets):
            d["track_id"] = i + 1  # what ByteTrack adds; it never changes bbox
        out = reg.enrich_detections("cam-1", self.frame, dets, face_db=db, face_results=face_results or None)
        return out, pass1_calls, db.calls

    @staticmethod
    def _strip(rows):
        keep = ("class_name", "label", "bbox", "face_identity_key", "face_match_score", "face_embedding", "object_type", "track_id")
        return [{k: r.get(k) for k in keep} for r in rows]

    def test_same_output_fewer_calls(self):
        old, p1_old, total_old = self._run(dedupe=False)
        new, p1_new, total_new = self._run(dedupe=True)
        self.assertEqual(self._strip(old), self._strip(new))
        # global ids are minted in the same order from fresh registries
        self.assertEqual([r.get("global_object_id") for r in old], [r.get("global_object_id") for r in new])
        self.assertEqual(p1_old, p1_new)
        self.assertEqual(total_old, p1_old + 3)  # old path: 3 persons recognised twice
        self.assertEqual(total_new, p1_new)  # new path: no second pass

    def test_identity_values(self):
        new, _, _ = self._run(dedupe=True)
        by_x = {r["bbox"][0]: r for r in new}
        self.assertEqual(by_x[10]["face_identity_key"], "face:alice")
        self.assertEqual(by_x[150]["face_identity_key"], "face:unknown:u0007")
        self.assertNotIn("face_identity_key", by_x[400])
        self.assertEqual(by_x[500]["object_type"], "vehicle")

    def test_unmatched_box_still_recognised(self):
        """A person box that pass 1 did not see (e.g. custom-model face) still gets pass 2."""
        db = FakeFaceDB()
        reg = ObjectIdentityRegistry(counter_file=f"{tempfile.mkdtemp(prefix='dedupe-')}/c.json")
        d = {"class_name": "person", "label": "person", "bbox": [20, 20, 70, 150], "track_id": 1, "model": "coco", "confidence": 0.9}
        other = {(1, 1, 2, 2): ("face:bob", 0.9, [0.5])}
        out = reg.enrich_detections("cam-1", self.frame, [d], face_db=db, face_results=other)
        self.assertEqual(db.calls, 1)
        self.assertEqual(out[0]["face_identity_key"], "face:alice")


if __name__ == "__main__":
    unittest.main()
