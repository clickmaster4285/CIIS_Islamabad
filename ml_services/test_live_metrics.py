"""Run: python -m unittest test_live_metrics  (from ml_services/)."""

import unittest

from live_metrics import LiveMetrics, percentile


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class PercentileTests(unittest.TestCase):
    def test_empty_and_basic(self):
        self.assertEqual(percentile([], 95), 0.0)
        self.assertEqual(percentile([5.0], 95), 5.0)
        vals = [float(i) for i in range(1, 101)]
        self.assertEqual(percentile(vals, 50), 51.0)
        self.assertEqual(percentile(vals, 95), 95.0)
        self.assertEqual(percentile(vals, 100), 100.0)


class LiveMetricsTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.m = LiveMetrics(window_sec=10.0, stale_sec=1.0, clock=self.clock, sample_system=False)

    def _infer(self, cam: str, *, every: float, n: int, age: float = 0.1, seq_step: int = 1, ok: bool = True):
        seq = 0
        for _ in range(n):
            start = self.clock.t
            prev = seq
            seq += seq_step
            self.m.record_camera_inference(
                cam,
                started_at=start,
                finished_at=start + 0.05,
                frame_ts=start - age,
                seq=seq,
                prev_seq=prev,
                ok=ok,
                error="" if ok else "boom",
            )
            self.clock.t += every

    def test_fps_gap_and_receiving(self):
        self._infer("cam-1", every=0.5, n=20)  # 10 s at 2 fps
        snap = self.m.snapshot(cameras={"cam-1": {"connected": True}}, sla_sec=1.0)
        cam = snap["cameras"][0]
        self.assertAlmostEqual(cam["infer_fps"], 2.0, delta=0.15)
        self.assertAlmostEqual(cam["gap_p95_sec"], 0.5, places=3)
        self.assertTrue(cam["receiving"])
        self.assertEqual(snap["summary"]["cameras_receiving"], 1)
        self.assertTrue(snap["summary"]["sla_met"])

    def test_starved_and_never_inferred(self):
        self._infer("cam-1", every=0.5, n=4)
        self.clock.t += 5.0  # cam-1 goes quiet
        snap = self.m.snapshot(
            cameras={"cam-1": {"connected": True}, "cam-2": {"connected": True}, "cam-3": {"connected": False}},
            sla_sec=1.0,
        )
        s = snap["summary"]
        self.assertEqual(s["cameras_connected"], 2)
        self.assertEqual(s["cameras_receiving"], 0)
        self.assertEqual(sorted(s["starved_keys"]), ["cam-1", "cam-2"])  # disconnected cam-3 not counted
        self.assertFalse(s["sla_met"])
        cam2 = next(c for c in snap["cameras"] if c["key"] == "cam-2")
        self.assertIsNone(cam2["last_ok_at"])
        self.assertEqual(cam2["inferences"], 0)

    def test_slow_gap_fails_sla_even_if_recent(self):
        self._infer("cam-1", every=3.0, n=3)
        self.clock.t -= 3.0  # snapshot right after the last inference
        snap = self.m.snapshot(cameras={"cam-1": {"connected": True}}, sla_sec=1.0)
        self.assertTrue(snap["cameras"][0]["receiving"])
        self.assertFalse(snap["summary"]["sla_met"])  # p95 gap 3 s > 1 s

    def test_stale_skipped_and_errors(self):
        self._infer("cam-1", every=0.5, n=4, age=2.5, seq_step=5)
        self._infer("cam-2", every=0.5, n=2, ok=False)
        snap = self.m.snapshot(sla_sec=1.0)
        c1 = next(c for c in snap["cameras"] if c["key"] == "cam-1")
        c2 = next(c for c in snap["cameras"] if c["key"] == "cam-2")
        self.assertEqual(c1["stale_frames"], 4)
        self.assertEqual(c1["frames_skipped"], 3 * 4)  # 4 skipped per step after the first
        self.assertAlmostEqual(c1["queue_delay_p50_sec"], 2.5, places=3)
        self.assertEqual(c2["errors"], 2)
        self.assertEqual(c2["inferences"], 0)
        self.assertEqual(c2["last_error"], "boom")

    def test_skips(self):
        self.m.record_camera_skip("cam-1", "no_new_frame")
        self.m.record_camera_skip("cam-1", "no_new_frame")
        self.m.record_camera_skip("cam-1", "no_frame")
        c = self.m.snapshot(sla_sec=1.0)["cameras"][0]
        self.assertEqual((c["no_new_frame"], c["no_frame"]), (2, 1))

    def test_model_stats(self):
        for i in range(10):
            self.m.record_model("coco", images=4 if i % 2 else 2, latency=0.020, lock_wait=0.010)
            self.clock.t += 1.0
        self.m.record_model("face_recognize", latency=0.005, kind="cpu")
        snap = self.m.snapshot(sla_sec=1.0)
        coco = snap["models"]["coco"]
        self.assertEqual(coco["avg_batch"], 3.0)
        self.assertEqual(coco["max_batch"], 4)
        self.assertAlmostEqual(coco["images_per_sec"], 3.0, delta=0.01)
        self.assertAlmostEqual(coco["latency_p95_ms"], 20.0, places=1)
        self.assertAlmostEqual(coco["lock_wait_avg_ms"], 10.0, places=1)
        self.assertAlmostEqual(coco["busy_pct"], 2.0, places=1)  # 10 × 20 ms over 10 s
        # cpu work is not counted as GPU busy
        self.assertAlmostEqual(snap["summary"]["gpu_model_busy_pct"], 2.0, places=1)

    def test_window_drops_old_events_and_reset(self):
        self._infer("cam-1", every=0.5, n=4)
        self.clock.t += 30.0  # beyond window
        snap = self.m.snapshot(cameras={"cam-1": {"connected": True}}, sla_sec=1.0)
        self.assertEqual(snap["cameras"][0]["infer_fps"], 0.0)
        self.assertEqual(snap["cameras"][0]["inferences"], 4)  # lifetime counter kept
        self.m.reset()
        self.assertEqual(self.m.snapshot(sla_sec=1.0)["cameras"], [])

    def test_stage_stats(self):
        for i in range(10):
            self.m.record_stage("acquire.mjpeg_imdecode", 0.004 if i < 9 else 0.040)
            self.clock.t += 1.0
        st = self.m.snapshot(sla_sec=1.0)["stages"]["acquire.mjpeg_imdecode"]
        self.assertAlmostEqual(st["calls_per_sec"], 1.0, places=2)
        self.assertAlmostEqual(st["p50_ms"], 4.0, places=2)
        self.assertAlmostEqual(st["max_ms"], 40.0, places=2)
        self.assertAlmostEqual(st["busy_pct"], 0.8, places=1)  # 76 ms over 10 s, 1-decimal rounding
        self.m.reset()
        self.assertEqual(self.m.snapshot(sla_sec=1.0)["stages"], {})

    def test_forget_camera(self):
        self._infer("cam-1", every=0.5, n=2)
        self.m.forget_camera("cam-1")
        self.assertEqual(self.m.snapshot(sla_sec=1.0)["cameras"], [])


if __name__ == "__main__":
    unittest.main()
