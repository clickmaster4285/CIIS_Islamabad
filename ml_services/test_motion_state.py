"""Run: python -m unittest test_motion_state  (from ml_services/)."""

import unittest

from motion_state import MOVING, STATIONARY, CameraMotionTracker, MotionConfig

CFG = MotionConfig(move_frac=0.15, scale_frac=0.30, start_sec=1.0, still_frac=0.05, stop_sec=3.0, track_ttl_sec=10.0)
FPS = 6.0  # live inference runs ~every 0.15 s


def det(x: float, y: float = 100.0, w: float = 100.0, h: float = 100.0, tid: int = 1) -> dict:
    return {"track_id": tid, "class_name": "car", "bbox": [x, y, x + w, y + h], "global_object_id": "GV1"}


class MotionStateTests(unittest.TestCase):
    def setUp(self):
        self.tracker = CameraMotionTracker("cam-1", CFG)
        self.t = 1000.0
        self.events: list[dict] = []

    def feed(self, positions, *, step: float = 1 / FPS, **kw) -> dict:
        d = {}
        for x in positions:
            d = det(x, **kw)
            self.events += self.tracker.update([d], self.t)
            self.t += step
        return d

    def kinds(self) -> list[str]:
        return [e["event"] for e in self.events]

    def test_parked_object_with_jitter_never_moves(self):
        d = self.feed([100 + (i % 3) * 4 for i in range(300)])  # ±4 px on a 100 px box, 50 s
        self.assertEqual(self.kinds(), [])
        self.assertEqual(d["motion_state"], STATIONARY)

    def test_short_bump_is_ignored(self):
        self.feed([100] * 10)
        self.feed([130] * 3)  # 0.5 s away, then back
        self.feed([100] * 20)
        self.assertEqual(self.kinds(), [])

    def test_move_then_park_gives_start_and_stop(self):
        self.feed([100] * 10)
        moving_from = self.t
        d = self.feed([100 + i * 10 for i in range(1, 31)])  # drives 300 px over 5 s
        self.assertEqual(d["motion_state"], MOVING)
        self.assertEqual(self.kinds(), ["start"])
        # Start time is when it first left the anchor, not when confirmed 1 s later
        self.assertLess(self.events[0]["at"], moving_from + 1.0)

        d = self.feed([400] * 30)  # parked 5 s
        self.assertEqual(self.kinds(), ["start", "stop"])
        self.assertEqual(d["motion_state"], STATIONARY)
        stop = self.events[1]
        self.assertEqual(stop["from_bbox"][0], 100)
        self.assertEqual(stop["to_bbox"][0], 400)
        self.assertEqual(stop["distance_px"], 300)
        self.assertEqual(stop["global_object_id"], "GV1")
        self.assertGreater(stop["duration_sec"], 4)

        # New resting place is the anchor: jitter there does not restart motion
        self.feed([400 + (i % 2) * 4 for i in range(60)])
        self.assertEqual(self.kinds(), ["start", "stop"])

    def test_approaching_camera_counts_as_motion(self):
        self.feed([100] * 10)
        # Same center, box grows 100 → 180 px over ~2.7 s
        for i in range(1, 17):
            size = 100 + i * 5
            off = 100 - (size - 100) / 2
            self.events += self.tracker.update([det(off, y=off, w=size, h=size)], self.t)
            self.t += 1 / FPS
        self.assertEqual(self.kinds(), ["start"])

    def test_object_leaving_while_moving_ends_as_lost(self):
        self.feed([100] * 10)
        self.feed([100 + i * 10 for i in range(1, 13)])
        self.assertEqual(self.kinds(), ["start"])
        self.t += 11.0
        self.events += self.tracker.update([], self.t)
        self.assertEqual(self.kinds(), ["start", "lost"])
        self.assertEqual(self.tracker.track_count(), 0)

    def test_faces_and_plates_are_ignored(self):
        for i in range(30):
            dets = [
                {"track_id": 7, "class_name": "face", "bbox": [100 + i * 10, 100, 150 + i * 10, 150]},
                {"track_id": 8, "class_name": "License_Plate", "bbox": [100 + i * 10, 300, 160 + i * 10, 320]},
            ]
            self.events += self.tracker.update(dets, self.t)
            self.t += 1 / FPS
        self.assertEqual(self.kinds(), [])
        self.assertEqual(self.tracker.track_count(), 0)
        self.assertNotIn("motion_state", dets[0])

    def test_detections_without_track_or_box_are_skipped(self):
        dets = [{"class_name": "car", "bbox": [0, 0, 10, 10]}, {"track_id": 5, "bbox": None}]
        self.assertEqual(self.tracker.update(dets, self.t), [])
        self.assertNotIn("motion_state", dets[0])
        self.assertEqual(self.tracker.track_count(), 0)


class GroundPointTests(unittest.TestCase):
    def test_parked_bikes_overlapping_do_not_move(self):
        tracker = CameraMotionTracker("cam-22", CFG)
        t, events = 1000.0, []
        # Real boxes of a parked motorcycle: top edge jumps 313 ↔ 386 (overlap), bottom stays 515
        for i in range(20):
            box = [975, 313, 1170, 515] if i % 2 else [976, 386, 1141, 516]
            events += tracker.update([{"track_id": 1, "global_object_id": "GV1019", "class_name": "motorcycle",
                                       "bbox": box}], t)
            t += 10.0
        self.assertEqual(events, [])

    def test_driven_bike_moves(self):
        tracker = CameraMotionTracker("cam-22", CFG)
        t, events = 1000.0, []
        for x in [975, 975, 975, 900, 800, 700, 600, 600, 600, 600, 600]:
            events += tracker.update([{"track_id": 1, "global_object_id": "GV1", "class_name": "motorcycle",
                                       "bbox": [x, 313, x + 195, 515]}], t)
            t += 1.0
        self.assertEqual([e["event"] for e in events], ["start", "stop"])


def person(hip_x: float, hip_y: float = 300.0, *, sitting: bool = True, wrist_dx: float = 0.0, tid: int = 1,
           gid: str = "GP1", box_dx: float = 0.0) -> dict:
    """Person with 17 keypoints: torso 100 px (shoulders 100 px above hips)."""
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    sh_y, knee_dy = hip_y - 100, 0.0 if sitting else 80.0
    for i, x in ((5, hip_x - 20), (6, hip_x + 20)):
        kps[i] = [x, sh_y, 0.9]
    kps[9] = [hip_x - 40 + wrist_dx, hip_y - 30, 0.9]
    kps[10] = [hip_x + 40 + wrist_dx, hip_y - 30, 0.9]
    for i, x in ((11, hip_x - 15), (12, hip_x + 15)):
        kps[i] = [x, hip_y, 0.9]
    # Sitting: knees forward (thigh horizontal); standing: knees straight below hips
    kps[13] = [hip_x - 15 + (0 if not sitting else 80), hip_y + knee_dy + (0 if not sitting else 5), 0.9]
    kps[14] = [hip_x + 15 + (0 if not sitting else 80), hip_y + knee_dy + (0 if not sitting else 5), 0.9]
    x1 = hip_x - 60 + box_dx
    return {"track_id": tid, "global_object_id": gid, "class_name": "person",
            "bbox": [x1, sh_y - 40, x1 + 120, hip_y + 120], "keypoints": kps}


class PoseMotionTests(unittest.TestCase):
    def setUp(self):
        self.tracker = CameraMotionTracker("cam-1", CFG)
        self.t = 1000.0
        self.events: list[dict] = []

    def step(self, d, dt=1.0):
        self.events += self.tracker.update([d], self.t)
        self.t += dt
        return d

    def kinds(self):
        return [e["event"] for e in self.events]

    def test_waving_and_leaning_in_chair_is_not_movement(self):
        for i in range(20):
            # Hands swing 60 px and the box shifts 40 px (leaning), hips stay put
            self.step(person(300, wrist_dx=60 * (i % 2), box_dx=40 * (i % 2)))
        self.assertEqual(self.kinds(), [])

    def test_walking_away_is_movement_with_body_shift_and_posture(self):
        for _ in range(3):
            d = self.step(person(300, sitting=True))
        self.assertEqual(d["posture"], "sitting")
        for x in range(380, 900, 80):  # stands up and walks 6 torso lengths
            d = self.step(person(x, sitting=False))
        self.assertEqual(d["posture"], "walking")
        for _ in range(5):
            d = self.step(person(900, sitting=False))
        self.assertEqual(self.kinds(), ["start", "stop"])
        stop = self.events[-1]
        self.assertGreater(stop["body_shift"], 5)
        self.assertEqual((self.events[0]["posture_from"], stop["posture_to"]), ("sitting", "standing"))
        self.assertEqual(d["posture"], "standing")

    def test_new_track_id_keeps_the_movement_by_lasting_id(self):
        for _ in range(3):
            self.step(person(300, tid=1))
        self.step(person(400, tid=1))
        self.step(person(500, tid=2))  # tracker lost him: new local ID, same GP1
        self.step(person(600, tid=3))
        for _ in range(5):
            self.step(person(600, tid=3))
        self.assertEqual(self.kinds(), ["start", "stop"])
        self.assertEqual(self.events[-1]["motion_key"], "g:GP1")
        self.assertEqual(self.events[-1]["track_id"], 3)
        self.assertEqual(self.tracker.track_count(), 1)

    def test_seated_thigh_pointing_at_high_camera_is_not_standing(self):
        from motion_state import posture

        d = person(300, sitting=False)
        self.assertEqual(posture(d["keypoints"]), "standing")
        d["keypoints"][13][1] = d["keypoints"][14][1] = 330  # thigh vertical but only 30 px (torso 100)
        self.assertEqual(posture(d["keypoints"]), "sitting")

    def test_slow_camera_does_not_forget_objects_between_frames(self):
        for _ in range(4):
            self.step(person(300), dt=20.0)  # one inference every 20 s
        self.assertGreaterEqual(self.tracker._ttl(), 60.0)
        self.assertEqual(self.tracker.track_count(), 1)
        self.assertEqual(self.kinds(), [])


if __name__ == "__main__":
    unittest.main()
