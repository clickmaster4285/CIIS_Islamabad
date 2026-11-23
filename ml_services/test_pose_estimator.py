"""Run: python -m unittest test_pose_estimator  (from ml_services/)."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pose_estimator import attach_poses, is_person, parse_pose_result, resolve_pose_weights


def kp(x: float, y: float) -> list[list[float]]:
    return [[x + i, y + i, 0.9] for i in range(17)]


def pose(box, score=0.8, at=(0, 0)):
    return {"bbox": box, "score": score, "keypoints": kp(*at)}


class AttachPoseTests(unittest.TestCase):
    def test_keypoints_go_to_the_matching_person(self):
        a = {"class_name": "person", "bbox": [0, 0, 100, 200]}
        b = {"class_name": "person", "bbox": [300, 0, 400, 200]}
        n = attach_poses([a, b], [pose([302, 4, 398, 198], at=(350, 50)), pose([2, 2, 98, 202], at=(50, 50))])
        self.assertEqual(n, 2)
        self.assertEqual(a["keypoints"][0][:2], [50, 50])
        self.assertEqual(b["keypoints"][0][:2], [350, 50])
        self.assertEqual(len(a["keypoints"]), 17)

    def test_objects_never_get_pose(self):
        laptop = {"class_name": "laptop", "bbox": [0, 0, 100, 200]}
        self.assertEqual(attach_poses([laptop], [pose([0, 0, 100, 200])]), 0)
        self.assertNotIn("keypoints", laptop)
        self.assertFalse(is_person(laptop))

    def test_far_pose_is_not_attached(self):
        a = {"class_name": "person", "bbox": [0, 0, 100, 200]}
        self.assertEqual(attach_poses([a], [pose([500, 500, 600, 700])]), 0)
        self.assertNotIn("keypoints", a)

    def test_one_pose_per_person(self):
        a = {"class_name": "person", "bbox": [0, 0, 100, 200]}
        b = {"class_name": "person", "bbox": [5, 0, 105, 200]}  # overlapping people
        attach_poses([a, b], [pose([0, 0, 100, 200], at=(1, 1))])
        self.assertEqual(sum("keypoints" in d for d in (a, b)), 1)


class ParsePoseTests(unittest.TestCase):
    def test_scales_back_to_original_frame(self):
        data = np.zeros((1, 17, 3), dtype=np.float32)
        data[0, :, 0], data[0, :, 1], data[0, :, 2] = 10, 20, 0.5
        result = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=np.array([[1, 2, 3, 4]], dtype=np.float32), conf=np.array([0.9])),
            keypoints=SimpleNamespace(data=data),
        )
        out = parse_pose_result(result, sx=2.0, sy=3.0)
        self.assertEqual(out[0]["bbox"], [2.0, 6.0, 6.0, 12.0])
        self.assertEqual(out[0]["keypoints"][0], [20.0, 60.0, 0.5])

    def test_empty_result(self):
        self.assertEqual(parse_pose_result(SimpleNamespace(boxes=None, keypoints=None)), [])


class WeightsTests(unittest.TestCase):
    def test_default_weights_present(self):
        weights = resolve_pose_weights()
        if weights is None:
            self.skipTest("pose weights not installed on this machine")
        self.assertTrue(Path(weights).is_file())


if __name__ == "__main__":
    unittest.main()
