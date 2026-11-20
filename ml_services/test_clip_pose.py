"""Run: python -m unittest test_clip_pose  (from ml_services/)."""

import unittest

import numpy as np

from clip_pose import body_point, dense_path, run_pose, smooth_path, track_person


def pose_at(x: float, y: float = 100.0, w: float = 40.0, h: float = 100.0) -> dict:
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[11] = [x - 8, y + 10, 0.9]  # hips at box middle-ish
    kps[12] = [x + 8, y + 10, 0.9]
    return {"bbox": [x - w / 2, y - h / 2, x + w / 2, y + h / 2], "score": 0.9, "keypoints": kps}


class TrackPersonTests(unittest.TestCase):
    def test_follows_walker_frame_by_frame(self):
        # Walker moves 6 px per frame; a seated person sits still at x=300
        frames = [[pose_at(50 + 6 * i), pose_at(300)] for i in range(40)]
        times = [i / 8 for i in range(40)]
        track = track_person(frames, times, pose_at(50)["bbox"])
        self.assertTrue(all(t is not None for t in track))
        self.assertAlmostEqual(track[-1]["point"][0], 50 + 6 * 39)
        path = dense_path(track)
        self.assertEqual(len(path), 40)  # one point per frame
        self.assertLess(path[0][0], path[-1][0])

    def test_short_miss_is_bridged_long_miss_is_lost(self):
        frames = [[pose_at(50 + 4 * i)] for i in range(10)] + [[]] * 3 + [[pose_at(50 + 4 * i)] for i in range(13, 20)]
        track = track_person(frames, list(range(20)), pose_at(50)["bbox"])
        self.assertIsNone(track[11])
        self.assertIsNotNone(track[15])  # found again after 3 missed frames
        lost = track_person([[pose_at(50)]] + [[]] * 15 + [[pose_at(50)]], list(range(17)), pose_at(50)["bbox"])
        self.assertIsNone(lost[-1])  # > max_gap: stays lost instead of grabbing a new person

    def test_two_walkers_do_not_swap(self):
        frames = [[pose_at(100 + 5 * i), pose_at(400 - 5 * i)] for i in range(20)]
        taken = [set() for _ in frames]
        a = track_person(frames, list(range(20)), pose_at(100)["bbox"], taken=taken)
        b = track_person(frames, list(range(20)), pose_at(400)["bbox"], taken=taken)
        self.assertGreater(a[-1]["point"][0], 100)
        self.assertLess(b[-1]["point"][0], 400)

    def test_path_starts_at_movement_start(self):
        frames = [[pose_at(50)] for _ in range(10)] + [[pose_at(50 + 10 * i)] for i in range(1, 11)]
        times = [i / 8 for i in range(20)]
        path = dense_path(track_person(frames, times, pose_at(50)["bbox"]), start_time=10 / 8)
        self.assertGreaterEqual(path[0][2], 10 / 8 - 0.5)

    def test_body_point_falls_back_to_box_centre(self):
        p = pose_at(100)
        p["keypoints"] = [[0, 0, 0.0]] * 17
        self.assertEqual(body_point(p), (100.0, 100.0))

    def test_smooth_keeps_ends(self):
        path = [[0, 0, 0], [10, 9, 1], [20, 21, 2], [30, 30, 3]]
        sm = smooth_path(path)
        self.assertEqual(sm[0], path[0])
        self.assertEqual(sm[-1], path[-1])

    def test_run_pose_batches(self):
        calls = []

        def fake(batch):
            calls.append(len(batch))
            return [[] for _ in batch]

        out = run_pose([np.zeros((4, 4, 3), np.uint8)] * 20, fake, batch_size=8)
        self.assertEqual(calls, [8, 8, 4])
        self.assertEqual(len(out), 20)


if __name__ == "__main__":
    unittest.main()
