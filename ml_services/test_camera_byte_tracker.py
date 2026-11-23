"""Run: python -m unittest test_camera_byte_tracker  (from ml_services/)."""

import unittest

import numpy as np
from ultralytics.trackers.basetrack import BaseTrack

from camera_byte_tracker import CameraByteTrackerPool

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def box(x, cls="person", cls_id=0, conf=0.9):
    return {"class_name": cls, "class_id": cls_id, "confidence": conf, "bbox": [x, 100, x + 40, 200]}


def run(pool, cam, frames):
    out = []
    for dets in frames:
        out = pool.assign_track_ids(cam, dets, FRAME)
    return out


class CameraByteTrackerTest(unittest.TestCase):
    def test_ids_unique_per_camera_across_classes(self):
        pool = CameraByteTrackerPool()
        frames = [[box(10 + i), box(300 + i, "car", 2), box(500 + i, "backpack", 24)] for i in range(5)]
        out = run(pool, "cam-1", frames)
        ids = [d["track_id"] for d in out]
        self.assertEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 3, ids)
        self.assertTrue(all(i < 1000 for i in ids), ids)  # small counter ids, no hash mixing

    def test_cameras_do_not_reset_each_other(self):
        pool = CameraByteTrackerPool()
        a = run(pool, "cam-a", [[box(10 + i)] for i in range(4)])
        tid_a = a[0]["track_id"]
        # A new camera (new tracker) must not reset cam-a's ids or the global counter.
        run(pool, "cam-b", [[box(200 + i)] for i in range(4)])
        a2 = pool.assign_track_ids("cam-a", [box(15)], FRAME)
        self.assertEqual(a2[0]["track_id"], tid_a)

    def test_global_strack_counter_untouched(self):
        BaseTrack._count = 777
        pool = CameraByteTrackerPool()
        run(pool, "cam-x", [[box(10 + i), box(300 + i, "car", 2)] for i in range(3)])
        self.assertEqual(BaseTrack._count, 777)

    def test_absent_class_tracks_age_out(self):
        pool = CameraByteTrackerPool(track_buffer=30)
        run(pool, "cam-1", [[box(10 + i), box(300 + i, "car", 2)] for i in range(4)])
        car_tracker = pool._trackers[("cam-1", "2:car")]
        start = car_tracker.frame_id
        # Only people from now on: the car slot is still updated every frame.
        run(pool, "cam-1", [[box(14 + i)] for i in range(40)])
        self.assertEqual(car_tracker.frame_id, start + 40)
        self.assertEqual(len(car_tracker.tracked_stracks), 0)

    def test_reset_camera(self):
        pool = CameraByteTrackerPool()
        run(pool, "cam-1", [[box(10)]])
        pool.reset("cam-1")
        self.assertFalse(any(k[0] == "cam-1" for k in pool._trackers))
        out = pool.assign_track_ids("cam-1", [box(10)], FRAME)
        self.assertEqual(out[0]["track_id"], 1)

    def test_empty_detections(self):
        pool = CameraByteTrackerPool()
        self.assertEqual(pool.assign_track_ids("cam-1", [], FRAME), [])


if __name__ == "__main__":
    unittest.main()
