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


if __name__ == "__main__":
    unittest.main()
