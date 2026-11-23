"""Run: python -m unittest test_render_on_view  (from ml_services/)."""

import threading
import time
import unittest

import numpy as np

import live_stream


class _FakeStream:
    connected = True

    def __init__(self):
        self.frame = np.zeros((72, 128, 3), dtype=np.uint8)

    def get_frame(self):
        return self.frame.copy()

    def stop(self):
        pass


def _manager(only_watched: bool) -> live_stream.LiveStreamManager:
    mgr = live_stream.LiveStreamManager()
    mgr._render_only_watched = only_watched
    mgr._render_grace = 0.5
    mgr._frame_interval = 0.02
    for key in ("cam-a", "cam-b"):
        mgr._sessions[key] = live_stream._CameraSession(key, _FakeStream(), "rtsp://x")
    return mgr


def _run_render_loop(mgr, seconds: float) -> None:
    mgr._running = True
    t = threading.Thread(target=mgr._render_loop, daemon=True)
    t.start()
    time.sleep(seconds)
    mgr._running = False
    t.join(2)


def _run_render_loop_until(mgr, cond, timeout: float) -> bool:
    """Run the render loop until ``cond()`` is true (or timeout). Returns cond()."""
    mgr._running = True
    t = threading.Thread(target=mgr._render_loop, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not cond():
        time.sleep(0.01)
    mgr._running = False
    t.join(2)
    return bool(cond())


class RenderOnViewTest(unittest.TestCase):
    def test_legacy_renders_every_camera(self):
        mgr = _manager(only_watched=False)
        _run_render_loop(mgr, 0.5)
        self.assertIsNotNone(mgr._sessions["cam-a"].latest_jpeg)
        self.assertIsNotNone(mgr._sessions["cam-b"].latest_jpeg)

    def test_unwatched_camera_is_not_rendered(self):
        mgr = _manager(only_watched=True)
        mgr.get_latest_jpeg("cam-a")  # a viewer opened cam-a's MJPEG
        _run_render_loop(mgr, 0.5)
        self.assertIsNotNone(mgr._sessions["cam-a"].latest_jpeg)
        self.assertIsNone(mgr._sessions["cam-b"].latest_jpeg)

    def test_viewer_grace_expires(self):
        mgr = _manager(only_watched=True)
        mgr.get_latest_jpeg("cam-a")
        # Wait (up to 3 s) until a render really happened — otherwise "no render after the
        # grace period" below would pass vacuously.
        self.assertTrue(
            _run_render_loop_until(mgr, lambda: mgr._sessions["cam-a"].rendered_at > 0, timeout=3.0),
            "the watched camera was never rendered",
        )
        first = mgr._sessions["cam-a"].rendered_at
        self.assertGreater(first, 0)
        time.sleep(mgr._render_grace + 0.3)  # past grace, no new views
        _run_render_loop(mgr, 0.3)
        self.assertEqual(mgr._sessions["cam-a"].rendered_at, first)

    def test_preview_snapshot_renders_on_demand(self):
        mgr = _manager(only_watched=True)
        jpeg = mgr.get_preview_jpeg("cam-b")  # no render loop running at all
        self.assertTrue(jpeg and jpeg[:2] == b"\xff\xd8")

    def test_status_has_frame_without_preview(self):
        mgr = _manager(only_watched=True)
        mgr._registry["cam-b"] = "rtsp://x"
        mgr._sessions["cam-b"].infer_seq_done = 3  # inference saw frames, nobody watched
        cams = {c["key"]: c for c in mgr.status()["cameras"]}
        self.assertTrue(cams["cam-b"]["has_frame"])


if __name__ == "__main__":
    unittest.main()
