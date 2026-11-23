"""Run: python -m unittest test_face_track_gate  (from ml_services/)."""

import unittest

import numpy as np

from face_track_gate import FaceTrackGate


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class _FakeFaceDB:
    """label_detection_detail returns queued results; counts calls."""

    def __init__(self, labels):
        self.labels = list(labels)
        self.calls = 0

    def label_detection_detail(self, frame, cls_id, name, bbox):
        self.calls += 1
        label = self.labels.pop(0) if self.labels else "unknown"
        known = label not in ("unknown", "person") and not label.startswith("unknown:")
        return {
            "label": label,
            "face_embedding": [0.1, 0.2] if label != "unknown" else [],
            "face_match_score": 0.8 if known else None,
            "face_match_method": "linear" if label != "unknown" else "",
        }


def _person(track_id=None, bbox=(10, 10, 50, 120)):
    d = {"class_id": 0, "class_name": "person", "label": "person", "bbox": list(bbox), "model": "coco"}
    if track_id is not None:
        d["track_id"] = track_id
    return d


FRAME = np.zeros((200, 200, 3), dtype=np.uint8)


class FaceTrackGateTest(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.gate = FaceTrackGate(reverify_sec=30, unknown_retry_sec=1.0, clock=self.clock)

    def test_known_person_reused_until_reverify(self):
        db = _FakeFaceDB(["Ali", "Ali"])
        d1 = self.gate.label("cam", FRAME, [_person(7)], db)[0]
        self.assertEqual((d1["label"], db.calls), ("Ali", 1))
        self.clock.t += 10
        d2 = self.gate.label("cam", FRAME, [_person(7)], db)[0]
        self.assertEqual((d2["label"], d2["face_match_method"], db.calls), ("Ali", "track_cache", 1))
        self.assertEqual(d2["face_embedding"], [0.1, 0.2])
        self.clock.t += 21  # 31 s after first recognition
        self.gate.label("cam", FRAME, [_person(7)], db)
        self.assertEqual(db.calls, 2)

    def test_unknown_retried_after_retry_interval(self):
        db = _FakeFaceDB(["unknown", "Sara"])
        self.gate.label("cam", FRAME, [_person(3)], db)
        self.clock.t += 0.5
        d = self.gate.label("cam", FRAME, [_person(3)], db)[0]
        self.assertEqual((d["label"], db.calls), ("unknown", 1))
        self.clock.t += 0.6  # 1.1 s -> retry, person now faces the camera
        d = self.gate.label("cam", FRAME, [_person(3)], db)[0]
        self.assertEqual((d["label"], db.calls), ("Sara", 2))

    def test_no_track_id_always_recognised(self):
        db = _FakeFaceDB(["Ali", "Ali"])
        self.gate.label("cam", FRAME, [_person()], db)
        self.gate.label("cam", FRAME, [_person()], db)
        self.assertEqual(db.calls, 2)

    def test_tracks_are_per_camera(self):
        db = _FakeFaceDB(["Ali", "Bilal"])
        self.gate.label("cam-1", FRAME, [_person(1)], db)
        d = self.gate.label("cam-2", FRAME, [_person(1)], db)[0]
        self.assertEqual((d["label"], db.calls), ("Bilal", 2))

    def test_non_person_untouched(self):
        db = _FakeFaceDB([])
        car = {"class_id": 2, "class_name": "car", "label": "car", "bbox": [0, 0, 5, 5], "track_id": 1, "model": "coco"}
        weapon = {"class_id": 0, "class_name": "person", "label": "gun", "bbox": [0, 0, 5, 5], "model": "weapon"}
        out = self.gate.label("cam", FRAME, [car, weapon], db)
        self.assertEqual((out[0]["label"], out[1]["label"], db.calls), ("car", "gun", 0))

    def test_box_jump_forces_reverify(self):
        db = _FakeFaceDB(["Ali", "Bilal"])
        self.gate.label("cam", FRAME, [_person(7, bbox=(10, 10, 50, 120))], db)
        self.clock.t += 0.5
        # Same id, box moved a little (high IoU) -> cached
        d = self.gate.label("cam", FRAME, [_person(7, bbox=(14, 10, 54, 120))], db)[0]
        self.assertEqual((d["label"], db.calls), ("Ali", 1))
        self.clock.t += 0.5
        # Same id, box now somewhere else (id switched to another person) -> recognised again
        d = self.gate.label("cam", FRAME, [_person(7, bbox=(140, 10, 180, 120))], db)[0]
        self.assertEqual((d["label"], db.calls), ("Bilal", 2))

    def test_default_reverify_is_ten_seconds(self):
        gate = FaceTrackGate(unknown_retry_sec=1.0, clock=self.clock)
        self.assertEqual(gate.reverify_sec, 10.0)
        db = _FakeFaceDB(["Ali", "Ali"])
        gate.label("cam", FRAME, [_person(1)], db)
        self.clock.t += 9
        gate.label("cam", FRAME, [_person(1)], db)
        self.assertEqual(db.calls, 1)
        self.clock.t += 2
        gate.label("cam", FRAME, [_person(1)], db)
        self.assertEqual(db.calls, 2)

    def test_box_without_bbox_skipped(self):
        db = _FakeFaceDB([])
        d = {"class_id": 0, "class_name": "person", "label": "person", "model": "coco", "track_id": 1}
        self.gate.label("cam", FRAME, [d], db)
        self.assertEqual(db.calls, 0)

    def test_reset_and_forget(self):
        db = _FakeFaceDB(["Ali", "Ali", "Ali"])
        self.gate.label("cam", FRAME, [_person(5)], db)
        self.gate.reset("cam")
        self.gate.label("cam", FRAME, [_person(5)], db)
        self.assertEqual(db.calls, 2)
        self.clock.t += 120  # track gone for > forget_sec
        self.gate.label("cam", FRAME, [], db)
        self.gate._prune(self.clock.t)
        self.assertEqual(self.gate.size, 0)


if __name__ == "__main__":
    unittest.main()
