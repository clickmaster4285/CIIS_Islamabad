"""Run: python -m unittest test_motion_clips  (from ml_services/). Needs ffmpeg."""

import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from live_stream import _resolve_ffmpeg_path
from motion_clips import ClipConfig, MotionClipRecorder

FPS = 8.0  # sampled video frames per second
INFER_EVERY = 8  # inference (motion updates) about once per second


def frame(i: int) -> np.ndarray:
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, (i * 3) % 150 : (i * 3) % 150 + 10] = 255
    return img


def start(at: float, tid: int = 1, gid: str = "GV1") -> dict:
    return {"event": "start", "at": at, "track_id": tid, "global_object_id": gid, "class_name": "car",
            "from_bbox": [10, 10, 40, 40], "to_bbox": [20, 10, 50, 40]}


def stop(at: float, tid: int = 1, gid: str = "GV1") -> dict:
    return {"event": "stop", "at": at, "track_id": tid, "global_object_id": gid, "class_name": "car",
            "from_bbox": [10, 10, 40, 40], "to_bbox": [100, 10, 130, 40]}


@unittest.skipUnless(_resolve_ffmpeg_path(), "ffmpeg not installed")
class MotionClipRecorderTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="motion-clips-")
        cfg = ClipConfig(fps=FPS, preroll_sec=5.0, postroll_sec=2.0, min_sec=10.0, max_sec=30.0, out_dir=self.dir,
                        pixel_trigger=False)
        self.rec = MotionClipRecorder(cfg, ffmpeg_path=_resolve_ffmpeg_path)
        self.t = 1000.0
        self.i = 0
        self.moving: set[int] = set()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_for(self, seconds: float, events_at: dict | None = None):
        """Feed video frames at FPS; motion updates every INFER_EVERY frames; events_at: frame offset → events."""
        events_at = events_at or {}
        for n in range(int(seconds * FPS)):
            events = events_at.get(n, [])
            for ev in events:
                (self.moving.add if ev["event"] == "start" else self.moving.discard)(ev["track_id"])
            if events or n % INFER_EVERY == 0:
                dets = [{"track_id": tid, "class_name": "car", "motion_state": "moving", "bbox": [10 + n, 10, 40 + n, 40]}
                        for tid in self.moving]
                self.rec.on_motion("cam-5", dets, events, self.t)
            self.rec.add_frame("cam-5", frame(self.i), self.t)
            self.t += 1 / FPS
            self.i += 1

    def flush(self) -> list[dict]:
        while not self.rec._encode.empty():
            self.rec.write_clip(self.rec._encode.get())
        return self.rec.list_ready()

    def test_short_movement_gives_one_clip_of_at_least_10s_with_preroll(self):
        self.run_for(8)  # parked: fills pre-roll, nothing recorded
        self.assertEqual(self.flush(), [])
        t0 = self.t
        self.run_for(20, {0: [start(t0 - 1)], 16: [stop(t0 + 2)]})  # moves ~2 s
        clips = self.flush()
        self.assertEqual(len(clips), 1)
        c = clips[0]
        self.assertGreaterEqual(c["duration_sec"], 10.0)
        self.assertLess(c["duration_sec"], 14)
        self.assertLessEqual(c["started_at"], t0 - 4.5)  # 5 s pre-roll
        self.assertAlmostEqual(c["fps"], FPS, delta=0.5)
        self.assertEqual(c["camera_key"], "cam-5")
        mv = c["movements"][0]
        self.assertEqual((mv["global_object_id"], mv["still_moving"]), ("GV1", False))
        self.assertEqual(mv["to_bbox"][0], 100)
        cid = c["clip_id"]
        self.assertGreater(self.rec.file_path(cid, "video").stat().st_size, 0)
        self.assertTrue(self.rec.file_path(cid, "poster").is_file())
        self.assertTrue(self.rec.file_path(cid, "end").is_file())
        self.assertIsNone(self.rec.file_path(cid, "other"))
        self.assertGreater(self.rec.delete_clip(cid), 0)
        self.assertEqual(self.rec.list_ready(), [])

    def test_min_length_counts_from_first_recorded_frame(self):
        # Camera just connected: only 1 s in the pre-roll, motion stops at once
        self.run_for(1)
        t0 = self.t
        self.run_for(14, {0: [start(t0)], 1: [stop(t0 + 0.2)]})
        clips = self.flush()
        self.assertEqual(len(clips), 1)
        self.assertGreaterEqual(clips[0]["duration_sec"], 10.0)

    def test_long_movement_is_split_at_30s(self):
        self.run_for(1)
        self.run_for(70, {0: [start(self.t)]})
        clips = self.flush()
        self.assertEqual([c["part"] for c in clips], [1, 2])
        self.assertTrue(all(c["duration_sec"] <= 30.5 for c in clips))
        second = clips[1]["movements"][0]
        self.assertTrue(second["still_moving"])
        self.assertEqual(second["global_object_id"], "GV1")
        # Part 2 starts where part 1 left the object
        self.assertEqual(second["from_bbox"], clips[0]["movements"][0]["to_bbox"])

    def test_two_objects_moving_together_share_one_clip(self):
        self.run_for(6)
        t0 = self.t
        self.run_for(20, {0: [start(t0)], 3: [start(t0, tid=2, gid="GP4")], 16: [stop(t0 + 2), stop(t0 + 2, tid=2, gid="GP4")]})
        clips = self.flush()
        self.assertEqual(len(clips), 1)
        self.assertEqual(sorted(m["global_object_id"] for m in clips[0]["movements"]), ["GP4", "GV1"])

    def test_bad_clip_ids_are_rejected(self):
        self.assertIsNone(self.rec.file_path("../../etc/passwd", "video"))
        self.assertEqual(self.rec.delete_clip("..\\x"), 0)

    def test_metadata_is_valid_json_on_disk(self):
        self.run_for(1)
        self.run_for(15, {0: [start(self.t)], 3: [stop(self.t + 0.5)]})
        clips = self.flush()
        meta = json.loads((Path(self.dir) / f"{clips[0]['clip_id']}.json").read_text())
        self.assertEqual((meta["source_width"], meta["clip_width"]), (160, 160))


def small_stop(at: float, tid: int = 1, gid: str = "GV1", cls: str = "car") -> dict:
    # 4 px on a 160x120 frame (diagonal 200) = 2% → too small to keep
    return {"event": "stop", "at": at, "track_id": tid, "global_object_id": gid, "class_name": cls,
            "from_bbox": [10, 10, 40, 40], "to_bbox": [14, 10, 44, 40]}


@unittest.skipUnless(_resolve_ffmpeg_path(), "ffmpeg not installed")
class RealMovementFilterClipTests(unittest.TestCase):
    """Clips end-to-end with the real-movement filter (5% objects, 3% people of the diagonal)."""

    setUp = MotionClipRecorderTests.setUp
    tearDown = MotionClipRecorderTests.tearDown
    run_for = MotionClipRecorderTests.run_for
    flush = MotionClipRecorderTests.flush

    def test_small_movement_writes_no_clip(self):
        self.run_for(6)
        t0 = self.t
        self.run_for(20, {0: [start(t0)], 16: [small_stop(t0 + 2)]})
        self.assertEqual(self.flush(), [])
        self.assertEqual(self.rec.stats["clips_skipped_small"], 1)

    def test_clip_keeps_only_the_real_mover(self):
        self.run_for(6)
        t0 = self.t
        self.run_for(20, {0: [start(t0)], 3: [start(t0, tid=2, gid="GP4")],
                          16: [stop(t0 + 2), small_stop(t0 + 2, tid=2, gid="GP4", cls="person")]})
        clips = self.flush()
        self.assertEqual(len(clips), 1)
        self.assertEqual([m["global_object_id"] for m in clips[0]["movements"]], ["GV1"])


@unittest.skipUnless(_resolve_ffmpeg_path(), "ffmpeg not installed")
class MovementPathTests(unittest.TestCase):
    setUp = MotionClipRecorderTests.setUp
    tearDown = MotionClipRecorderTests.tearDown
    run_for = MotionClipRecorderTests.run_for
    flush = MotionClipRecorderTests.flush

    def test_path_is_drawn_on_clip_end_frame(self):
        self.run_for(6)
        t0 = self.t
        ev_stop = stop(t0 + 2)
        ev_stop["path"] = [[25, 25, t0], [70, 25, t0 + 1], [115, 25, t0 + 2]]
        self.run_for(20, {0: [start(t0)], 16: [ev_stop]})
        clips = self.flush()
        self.assertEqual(len(clips), 1)
        end = cv2.imread(str(self.rec.file_path(clips[0]["clip_id"], "end")))
        b, g, r = end[25, 90]  # on the line between the 2nd and 3rd point
        self.assertTrue(r > 150 and g > 150 and b < 120, (b, g, r))  # yellow


@unittest.skipUnless(_resolve_ffmpeg_path(), "ffmpeg not installed")
class FramePoseClipTests(unittest.TestCase):
    """Pose run on every clip frame: skeleton per frame + dense path."""

    setUp = MotionClipRecorderTests.setUp
    tearDown = MotionClipRecorderTests.tearDown
    run_for = MotionClipRecorderTests.run_for
    flush = MotionClipRecorderTests.flush

    def test_person_tracked_on_every_frame(self):
        seen = {"i": 0}

        def fake_pose(images):
            out = []
            for _ in images:
                i = seen["i"]
                seen["i"] += 1
                x = 25 + min(max(0, i - 48), 100)  # rests for the pre-roll, then walks right 1 px/frame
                kps = [[0.0, 0.0, 0.0] for _ in range(17)]
                kps[5], kps[6] = [x - 5, 40, 0.9], [x + 5, 40, 0.9]
                kps[11], kps[12] = [x - 4, 60, 0.9], [x + 4, 60, 0.9]
                out.append([{"bbox": [x - 10, 30, x + 10, 90], "score": 0.9, "keypoints": kps}])
            return out

        self.rec._pose_batch = fake_pose
        self.run_for(6)
        t0 = self.t
        ev_start = dict(start(t0), class_name="person", from_bbox=[15, 30, 35, 90], to_bbox=[20, 30, 40, 90])
        ev_stop = dict(stop(t0 + 9), class_name="person", from_bbox=[15, 30, 35, 90], to_bbox=[115, 30, 135, 90])
        self.run_for(20, {0: [ev_start], 72: [ev_stop]})
        clips = self.flush()
        self.assertEqual(len(clips), 1)
        mv = clips[0]["movements"][0]
        self.assertGreater(mv["pose_frames"], 100)       # found on (nearly) every frame
        self.assertGreater(len(mv["path"]), 50)          # dense path, not 2-3 live points
        self.assertGreater(mv["path"][-1][0], mv["path"][0][0] + 50)
        self.assertEqual(self.rec.stats["clips_pose_tracked"], 1)
        end = cv2.imread(str(self.rec.file_path(clips[0]["clip_id"], "end")))
        b, g, r = end[60, 70]  # on the dense yellow path
        self.assertTrue(r > 150 and g > 150 and b < 120, (b, g, r))


@unittest.skipUnless(_resolve_ffmpeg_path(), "ffmpeg not installed")
class DangerClipTests(unittest.TestCase):
    """A weapon / fire in view starts a clip at once, even when nothing moves."""

    setUp = MotionClipRecorderTests.setUp
    tearDown = MotionClipRecorderTests.tearDown
    flush = MotionClipRecorderTests.flush

    def feed(self, seconds: float, danger: list[dict] | None = None):
        for n in range(int(seconds * FPS)):
            if n % INFER_EVERY == 0:
                dets = list(danger or [])
                self.rec.on_motion("cam-5", [], [], self.t, [d for d in dets if self.rec.danger_class(d)])
            self.rec.add_frame("cam-5", frame(self.i), self.t)
            self.t += 1 / FPS
            self.i += 1

    def test_still_gun_gives_danger_clip(self):
        self.feed(6)
        gun = {"class_name": "gun", "model": "weapon", "confidence": 0.8, "bbox": [50, 50, 60, 60]}
        self.feed(4, [gun])   # gun in view 4 s, never moves
        self.feed(12)         # gone
        clips = self.flush()
        self.assertEqual(len(clips), 1)
        c = clips[0]
        self.assertEqual(c["trigger"], "danger")
        mv = c["movements"][0]
        self.assertEqual((mv["class_name"], mv["danger"], mv["track_id"], mv["still_moving"]), ("gun", True, None, False))

    def test_gun_left_in_place_gives_only_one_clip(self):
        self.feed(6)
        gun = {"class_name": "handgun", "model": "weapon", "bbox": [50, 50, 60, 60]}
        self.feed(70, [gun])          # on the desk for 70 s
        self.feed(3)                  # flicker: missed for 3 s
        self.feed(20, [gun])          # back at the same spot
        self.feed(12)
        self.assertEqual(len(self.flush()), 1)

    def test_gun_that_moves_gets_a_new_clip(self):
        self.feed(6)
        self.feed(4, [{"class_name": "handgun", "model": "weapon", "bbox": [10, 10, 30, 30]}])
        self.feed(12)
        self.feed(4, [{"class_name": "handgun", "model": "weapon", "bbox": [110, 60, 130, 80]}])
        self.feed(12)
        self.assertEqual(len(self.flush()), 2)

    def test_fire_is_danger_and_person_is_not(self):
        self.assertEqual(self.rec.danger_class({"class_name": "fire", "model": "smoke"}), "fire")
        self.assertEqual(self.rec.danger_class({"class_name": "heavy-weapon"}), "heavy-weapon")
        self.assertEqual(self.rec.danger_class({"class_name": "person", "model": "coco"}), "")

    def test_danger_ignores_cooldown(self):
        rec = MotionClipRecorder(ClipConfig(cooldown_sec=999, out_dir=self.dir, pixel_trigger=False), ffmpeg_path=_resolve_ffmpeg_path)
        rec._cameras["cam-5"] = __import__("motion_clips")._Camera(cooldown_until=self.t + 999)
        rec.on_motion("cam-5", [], [], self.t, [{"class_name": "fire", "bbox": [1, 1, 5, 5]}])
        self.assertIsNotNone(rec._cameras["cam-5"].recording)


class RealMovementRuleTests(unittest.TestCase):
    """is_real_move on a 600x800 frame: diagonal 1000 px, so 1% = 10 px."""

    def setUp(self):
        self.rec = MotionClipRecorder(ClipConfig(out_dir=tempfile.gettempdir()))

    def mv(self, cls: str, dx: float = 0.0, grow: float = 1.0):
        from motion_clips import _Movement

        w = 100 * grow
        return _Movement(track_id=1, global_object_id="G1", class_name=cls, started_at=0.0,
                         from_bbox=[100, 100, 200, 200], to_bbox=[100 + dx, 100, 100 + dx + w, 100 + w])

    def real(self, mv) -> bool:
        return self.rec.is_real_move(mv, 600, 800)

    def test_object_needs_5_percent(self):
        self.assertFalse(self.real(self.mv("car", dx=40)))   # 4%
        self.assertTrue(self.real(self.mv("car", dx=55)))    # 5.5%

    def test_person_needs_5_percent(self):
        self.assertFalse(self.real(self.mv("person", dx=40)))  # 4%
        self.assertTrue(self.real(self.mv("person", dx=55)))   # 5.5%

    def test_vehicle_types_need_5_percent(self):
        for cls in ("truck", "motorcycle", "microbus", "pickup-van"):
            self.assertFalse(self.real(self.mv(cls, dx=40)), cls)
            self.assertTrue(self.real(self.mv(cls, dx=55)), cls)

    def test_hand_wave_does_not_count(self):
        # Arm raised above the head: box grows upward, feet stay where they are
        mv = self.mv("person")
        mv.to_bbox = [100, 20, 200, 200]
        self.assertFalse(self.real(mv))
        # Arm stretched out sideways: box widens, feet move a little
        mv.to_bbox = [100, 100, 260, 200]
        self.assertFalse(self.real(mv))

    def test_hips_must_move_when_pose_seen(self):
        mv = self.mv("person", dx=80)  # box moved 8%
        mv.body_shift = 0.4  # but the hips barely moved: wrong / stretched box
        self.assertFalse(self.real(mv))
        mv.body_shift = 2.0
        self.assertTrue(self.real(mv))

    def test_standing_up_in_place_does_not_count(self):
        mv = self.mv("person", dx=5)
        mv.posture_from, mv.posture_to, mv.body_shift = "sitting", "standing", 0.6
        mv.to_bbox = [105, 40, 205, 205]  # taller box, same place
        self.assertFalse(self.real(mv))

    def test_other_objects_make_no_clip(self):
        self.assertFalse(self.real(self.mv("cell phone", dx=200)))
        self.assertFalse(self.real(self.mv("chair", dx=200)))

    def test_walking_toward_or_away_from_camera_counts(self):
        self.assertTrue(self.real(self.mv("person", grow=1.6)))
        self.assertTrue(self.real(self.mv("person", grow=0.6)))
        self.assertFalse(self.real(self.mv("person", grow=1.2)))  # standing up / leaning

    def test_weapons_always_kept(self):
        self.assertTrue(self.real(self.mv("heavy-weapon", dx=1)))
        self.assertTrue(self.real(self.mv("gun")))

    def test_other_object_start_does_not_open_a_recording(self):
        ev = {"event": "start", "at": 1.0, "track_id": 9, "class_name": "cell phone",
              "from_bbox": [10, 10, 20, 20], "to_bbox": [30, 10, 40, 20]}
        self.rec.on_motion("cam-x", [], [ev], 1.0)
        self.assertIsNone(self.rec._cameras["cam-x"].recording)
        self.rec.on_motion("cam-x", [], [dict(ev, class_name="person")], 1.0)
        self.assertIsNotNone(self.rec._cameras["cam-x"].recording)

    def test_unknown_size_is_kept(self):
        self.assertTrue(self.rec.is_real_move(self.mv("car", dx=1), 0, 0))

    def test_zero_threshold_keeps_everything(self):
        rec = MotionClipRecorder(ClipConfig(keep_min_pct=0, keep_person_pct=0, out_dir=tempfile.gettempdir()))
        self.assertTrue(rec.is_real_move(self.mv("person", dx=1), 600, 800))


def scene(person_x: float | None, arm_up: bool = False, clock: int = 0) -> np.ndarray:
    """Grey room; a white 'person' (12x40 px) standing / walking at person_x."""
    img = np.full((120, 160, 3), 100, dtype=np.uint8)
    if person_x is not None:
        x = int(person_x)
        img[60:100, x : x + 12] = 240
        if arm_up:
            img[52:60, x + 12 : x + 20] = 240  # raised hand beside the head
    if clock:
        img[2:6, 2:6] = 100 + 60 * (clock % 2)  # on-screen clock digit flickering
    return img


def white_box_pose(images):
    """Fake YOLO pose: the white 'person' in each image (big blob only, hand ignored for hips)."""
    out = []
    for img in images:
        ys, xs = np.where(img[:, :, 0] > 200)
        if len(xs) < 50:
            out.append([])
            continue
        x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        body_x = float(np.median(xs))
        kps = [[0.0, 0.0, 0.0] for _ in range(17)]
        kps[5], kps[6] = [body_x - 4, 66, 0.9], [body_x + 4, 66, 0.9]
        kps[11], kps[12] = [body_x - 3, 82, 0.9], [body_x + 3, 82, 0.9]
        out.append([{"bbox": [x1, y1, x2, y2], "score": 0.9, "keypoints": kps}])
    return out


class PixelMotionCheckTests(unittest.TestCase):
    def setUp(self):
        from motion_clips import _Camera

        self.rec = MotionClipRecorder(ClipConfig(out_dir=tempfile.mkdtemp()))
        self.cam = _Camera()

    def feed(self, imgs) -> list[bool]:
        return [self.rec.pixel_motion(self.cam, img) for img in imgs]

    def test_still_scene_is_quiet(self):
        self.assertFalse(any(self.feed([scene(60)] * 20)))

    def test_walking_person_is_motion(self):
        self.feed([scene(None)] * 5)
        self.assertTrue(any(self.feed([scene(10 + 3 * i) for i in range(6)])))

    def test_onscreen_clock_is_not_motion(self):
        self.assertFalse(any(self.feed([scene(None, clock=i) for i in range(20)])))

    def test_light_switch_is_not_motion(self):
        self.feed([scene(None)] * 5)
        bright = np.full((120, 160, 3), 220, dtype=np.uint8)
        self.assertFalse(any(self.feed([bright] * 3)))


@unittest.skipUnless(_resolve_ffmpeg_path(), "ffmpeg not installed")
class PixelTriggerClipTests(unittest.TestCase):
    """Live AI never saw the movement; the pixel check starts the clip, YOLO on the clip decides."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="motion-clips-")
        cfg = ClipConfig(fps=FPS, preroll_sec=5.0, postroll_sec=2.0, min_sec=10.0, max_sec=30.0, out_dir=self.dir)
        self.rec = MotionClipRecorder(cfg, ffmpeg_path=_resolve_ffmpeg_path, pose_batch=white_box_pose)
        self.t = 1000.0

    tearDown = MotionClipRecorderTests.tearDown
    flush = MotionClipRecorderTests.flush

    def play(self, imgs):
        for img in imgs:
            self.rec.add_frame("cam-p", img, self.t)
            self.t += 1 / FPS

    def test_quick_walk_past_is_recorded_without_live_ai(self):
        self.play([scene(None)] * int(6 * FPS))                   # empty doorway
        self.play([scene(5 + 4 * i) for i in range(30)])          # walks across in ~4 s
        self.play([scene(None)] * int(14 * FPS))                  # gone
        clips = self.flush()
        self.assertEqual(len(clips), 1)
        movers = clips[0]["movements"]
        self.assertEqual([m["class_name"] for m in movers], ["person"])
        self.assertLess(movers[0]["from_bbox"][0], 30)
        self.assertGreater(movers[0]["to_bbox"][0], 100)
        self.assertEqual(self.rec.stats["clips_discovered_movers"], 1)

    def test_hand_wave_in_place_is_not_kept(self):
        self.play([scene(60)] * int(6 * FPS))                     # person standing
        self.play([scene(60, arm_up=(i // 4) % 2 == 0) for i in range(40)])  # waves a hand
        self.play([scene(60)] * int(14 * FPS))
        self.assertEqual(self.flush(), [])
        self.assertGreaterEqual(self.rec.stats["clips_skipped_small"], 1)


class ManyMoversTests(unittest.TestCase):
    """Several people / vehicles moving on one camera at once all go into the clip."""

    def setUp(self):
        self.rec = MotionClipRecorder(ClipConfig(out_dir=tempfile.mkdtemp()))
        self.cam = "cam-m"

    @staticmethod
    def ev(kind, at, tid, key, x0, x1, cls="person", gid=""):
        return {"event": kind, "at": at, "track_id": tid, "motion_key": key, "global_object_id": gid,
                "class_name": cls, "from_bbox": [x0, 300, x0 + 80, 500], "to_bbox": [x1, 300, x1 + 80, 500]}

    @staticmethod
    def det(tid, key, x):
        return {"track_id": tid, "motion_key": key, "motion_state": "moving", "bbox": [x, 300, x + 80, 500]}

    def recording(self):
        return self.rec._cameras[self.cam].recording

    def test_three_movers_at_once_are_all_kept(self):
        r = self.rec
        r.on_motion(self.cam, [], [self.ev("start", 1, 1, "g:GP1", 100, 140),
                                   self.ev("start", 1, 2, "g:GP2", 1100, 1060),
                                   self.ev("start", 2, 3, "g:GV3", 600, 640, cls="car")], 2)
        r.on_motion(self.cam, [self.det(1, "g:GP1", 500), self.det(2, "g:GP2", 600), self.det(3, "g:GV3", 1000)], [], 6)
        r.on_motion(self.cam, [], [self.ev("stop", 7, 1, "g:GP1", 100, 520), self.ev("stop", 7, 2, "g:GP2", 1100, 580),
                                   self.ev("stop", 7, 3, "g:GV3", 600, 1010, cls="car")], 7)
        rec = self.recording()
        rec.source_size = (1280, 720)
        self.assertEqual(sorted(mv.key for mv in r.real_movements(rec)), ["g:GP1", "g:GP2", "g:GV3"])

    def test_lasting_id_given_while_walking_keeps_the_movement(self):
        r = self.rec
        r.on_motion(self.cam, [], [self.ev("start", 1, 2, "t:2", 1100, 1060)], 1)
        # The tracker gives the person its lasting ID while it is still walking
        r.on_motion(self.cam, [self.det(2, "g:GP2", 700)], [], 5)
        r.on_motion(self.cam, [], [self.ev("stop", 9, 2, "g:GP2", 1100, 500, gid="GP2")], 9)
        rec = self.recording()
        self.assertEqual(list(rec.movements), ["g:GP2"])
        mv = rec.movements["g:GP2"]
        self.assertFalse(mv.still_moving)
        self.assertEqual(mv.from_bbox[0], 1100)  # point A kept
        self.assertEqual(mv.to_bbox[0], 500)     # point B reached
        self.assertEqual(mv.global_object_id, "GP2")

    def test_walk_pause_walk_is_one_move_from_first_point(self):
        r = self.rec
        r.on_motion(self.cam, [], [self.ev("start", 1, 1, "g:GP1", 100, 140)], 1)
        r.on_motion(self.cam, [], [self.ev("stop", 4, 1, "g:GP1", 100, 160)], 4)
        r.on_motion(self.cam, [], [self.ev("start", 7, 1, "g:GP1", 160, 200)], 7)
        r.on_motion(self.cam, [], [self.ev("stop", 10, 1, "g:GP1", 160, 230)], 10)
        mv = self.recording().movements["g:GP1"]
        self.assertEqual((mv.from_bbox[0], mv.to_bbox[0]), (100, 230))

    def test_movement_without_stop_event_ends_after_stale_time(self):
        r = self.rec
        r.on_motion(self.cam, [], [self.ev("start", 1, 4, "t:4", 100, 300)], 1)
        rec = self.recording()
        rec.frames.append((1.0, b""))
        with r._lock:
            r._check_camera(self.cam, r._cameras[self.cam], 1 + r.config.moving_stale_sec + 1)
        self.assertFalse(rec.movements["t:4"].still_moving)


class FakeStream:
    def __init__(self):
        self.seq = 0
        self.lock = threading.Lock()

    def get_latest(self):
        with self.lock:
            return frame(self.seq), self.seq


class SamplerThreadTests(unittest.TestCase):
    def test_sampler_reads_new_frames_only(self):
        tmp = tempfile.mkdtemp(prefix="motion-clips-")
        self.addCleanup(shutil.rmtree, tmp, True)
        stream = FakeStream()
        rec = MotionClipRecorder(
            ClipConfig(fps=20.0, out_dir=tmp, pixel_trigger=False), ffmpeg_path=_resolve_ffmpeg_path, frame_sources=lambda: {"cam-1": stream}
        )
        rec.start()
        try:
            time.sleep(0.4)
            self.assertEqual(rec.status()["preroll_frames"], {"cam-1": 1})  # same seq → kept once
            for _ in range(3):
                with stream.lock:
                    stream.seq += 1
                time.sleep(0.15)
            self.assertEqual(rec.status()["preroll_frames"]["cam-1"], 4)
        finally:
            rec.stop()


if __name__ == "__main__":
    unittest.main()
