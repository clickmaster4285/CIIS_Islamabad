"""Run: python -m unittest test_osd_filter  (from ml_services/)."""

import unittest

from live_stream import filter_osd_detections


class OsdFilterTests(unittest.TestCase):
    def test_text_removed_vehicles_beyond_gate_kept(self):
        dets = [
            {"bbox": [30, 35, 400, 70], "class_name": "clock"},       # date / time overlay
            {"bbox": [250, 150, 360, 250], "class_name": "car"},      # road beyond the gate (centre in old band)
            {"bbox": [100, 120, 300, 205], "class_name": "car"},      # far car reaching below the strip
            {"bbox": [560, 300, 700, 420], "class_name": "person"},
        ]
        kept = [d["bbox"][1] for d in filter_osd_detections(dets, 720, 1280)]
        self.assertEqual(kept, [150, 120, 300])

    def test_small_corner_text_below_strip_removed(self):
        dets = [{"bbox": [20, 165, 200, 200], "class_name": "tv"}]   # 35 px tall label just under the strip
        self.assertEqual(filter_osd_detections(dets, 720, 1280), [])

    def test_right_corner_text_removed(self):
        dets = [{"bbox": [1100, 165, 1260, 200], "class_name": "tv"}]
        self.assertEqual(filter_osd_detections(dets, 720, 1280), [])

    def test_small_box_mid_frame_below_strip_kept(self):
        # Same size / height as corner text but in the middle of the frame: a real object.
        dets = [{"bbox": [560, 165, 720, 200], "class_name": "bird"}]
        self.assertEqual(len(filter_osd_detections(dets, 720, 1280)), 1)

    def test_corner_fractions_cannot_cover_whole_width(self):
        # Legacy 0.55 / 0.55 settings used to make the "corner" guard cover the whole frame.
        dets = [{"bbox": [600, 165, 680, 200], "class_name": "bird"}]
        kept = filter_osd_detections(dets, 720, 1280, left_frac=0.55, right_frac=0.55)
        self.assertEqual(len(kept), 1)


if __name__ == "__main__":
    unittest.main()
