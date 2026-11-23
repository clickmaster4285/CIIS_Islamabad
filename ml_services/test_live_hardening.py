"""Run: python -m unittest test_live_hardening  (from ml_services/).

Stream URL checks, credential redaction, ffmpeg command flags, closed-session handling,
lazy evidence JPEG, stale specialist boxes and MJPEG generator termination.
"""

import asyncio
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import live_stream
from motion_clips import ClipConfig, MotionClipRecorder


class _FakeStream:
    connected = True
    keep_native = False

    def __init__(self):
        self.frame = np.zeros((72, 128, 3), dtype=np.uint8)
        self.seq = 1

    def get_frame(self):
        return self.frame.copy()

    def get_latest(self):
        return self.frame.copy(), self.seq

    def stop(self):
        pass


class UrlTests(unittest.TestCase):
    def test_redact(self):
        self.assertEqual(
            live_stream.redact_url("rtsp://admin:S3cr%40t@10.0.0.5:554/Streaming/Channels/101"),
            "rtsp://admin:***@10.0.0.5:554/Streaming/Channels/101",
        )
        self.assertEqual(live_stream.redact_url("rtsp://10.0.0.5/x"), "rtsp://10.0.0.5/x")
        self.assertIn("rtsp://u:***@h", live_stream.redact_url("Error opening rtsp://u:p@h/1 timeout"))

    def test_scheme_allowlist(self):
        with mock.patch.dict(os.environ, {"ML_ALLOWED_STREAM_SCHEMES": ""}):
            self.assertTrue(live_stream.stream_url_allowed("rtsp://a:b@h/1"))
            self.assertTrue(live_stream.stream_url_allowed("RTSPS://h/1"))
            for bad in ("file:///etc/passwd", "/etc/passwd", "http://h/x.mjpg", "concat:a|b", ""):
                self.assertFalse(live_stream.stream_url_allowed(bad), bad)
            self.assertNotIn("http", live_stream._protocol_whitelist())
        with mock.patch.dict(os.environ, {"ML_ALLOWED_STREAM_SCHEMES": "rtsp,http"}):
            self.assertTrue(live_stream.stream_url_allowed("http://h/x.mjpg"))
            self.assertIn("http", live_stream._protocol_whitelist().split(","))

    def test_register_rejects_bad_scheme(self):
        mgr = live_stream.LiveStreamManager()
        self.assertFalse(mgr.register_camera("cam-1", "file:///etc/passwd"))
        self.assertFalse(mgr.is_registered("cam-1"))

    def test_status_redacts(self):
        mgr = live_stream.LiveStreamManager()
        mgr._registry["cam-1"] = "rtsp://admin:pw@1.2.3.4/1"
        row = mgr.status()["cameras"][0]
        self.assertEqual(row["rtsp_url"], "rtsp://admin:***@1.2.3.4/1")


class FfmpegCmdTests(unittest.TestCase):
    def test_cmd_has_whitelist_timeout_and_fps_cap(self):
        live_stream._ffmpeg_version_cache["/fake/ffmpeg"] = (8, 1)
        with mock.patch.object(live_stream, "_use_nvdec", return_value=False), \
                mock.patch.dict(os.environ, {"ML_INGEST_FPS": "10", "ML_FFMPEG_TIMEOUT_FLAG": ""}):
            s = live_stream.FfmpegCameraStream("rtsp://u:p@h/1", "t", "/fake/ffmpeg")
            cmd = s._ffmpeg_cmd()
        self.assertLess(cmd.index("-protocol_whitelist"), cmd.index("-i"))
        self.assertIn("-timeout", cmd)
        self.assertEqual(cmd[cmd.index("-fpsmax") + 1], "10")

    def test_old_ffmpeg_uses_stimeout(self):
        live_stream._ffmpeg_version_cache["/old/ffmpeg"] = (4, 2)
        with mock.patch.dict(os.environ, {"ML_FFMPEG_TIMEOUT_FLAG": ""}):
            self.assertEqual(live_stream._ffmpeg_timeout_cli_flag("/old/ffmpeg"), "stimeout")
            args, vf = live_stream._ingest_fps_args("/old/ffmpeg")
        self.assertIn("select=", vf)
        self.assertEqual(args, ["-vsync", "vfr"])

    def test_default_ingest_fps_covers_consumers(self):
        with mock.patch.dict(os.environ, {"ML_INGEST_FPS": "", "ML_MOTION_CLIP_FPS": "8", "ML_LIVE_STREAM_FPS": "12"}):
            self.assertEqual(live_stream._ingest_fps(), 12.0)

    def test_stall_kills_process(self):
        live_stream._ffmpeg_version_cache["/fake/ffmpeg"] = (8, 1)
        with mock.patch.object(live_stream, "_use_nvdec", return_value=False):
            s = live_stream.FfmpegCameraStream("rtsp://h/1", "t", "/fake/ffmpeg")
        proc = mock.Mock()
        proc.poll.return_value = None
        s._proc = proc
        s._proc_started_at = time.time() - 30
        s.frame_ts = 0.0
        s._check_stall(time.time(), 15.0)
        proc.kill.assert_called_once()
        s.running = False


def _manager():
    mgr = live_stream.LiveStreamManager()
    mgr._sessions["cam-a"] = live_stream._CameraSession("cam-a", _FakeStream(), "rtsp://x")
    mgr._registry["cam-a"] = "rtsp://x"
    return mgr


class ManagerTests(unittest.TestCase):
    def test_lazy_evidence(self):
        mgr = _manager()
        frame = np.full((72, 128, 3), 120, dtype=np.uint8)
        with mock.patch.object(live_stream, "encode_jpeg", wraps=live_stream.encode_jpeg) as enc:
            mgr._publish_results("cam-a", [], frame=frame)
            enc.assert_not_called()  # nothing encoded until someone asks
            self.assertTrue(mgr.get_detection_snapshot("cam-a")["has_evidence"])
            a = mgr.get_evidence_jpeg("cam-a")
            b = mgr.get_evidence_jpeg("cam-a")
            self.assertEqual(enc.call_count, 1)
        self.assertTrue(a and a[:2] == b"\xff\xd8" and a is b)

    def test_closed_session_results_dropped(self):
        mgr = _manager()
        session = mgr._sessions["cam-a"]
        with mock.patch.object(mgr, "_infer_frame", return_value=[{"class_name": "car", "bbox": [0, 0, 5, 5]}]), \
                mock.patch.object(live_stream, "get_camera_session_manager", create=True):
            with mgr._lock:
                mgr._detach_session_locked("cam-a")
            mgr._infer_and_publish(session, session.stream.get_frame(), 2, 0.0, 1, time.time())
        self.assertNotIn("cam-a", mgr._detections)
        self.assertFalse(session.infer_busy)

    def test_unregister_resets_trackers_and_binds(self):
        mgr = _manager()
        with mock.patch.object(mgr._byte_trackers, "reset") as reset, \
                mock.patch.object(mgr._object_identity, "forget_camera") as forget:
            mgr.unregister_camera("cam-a")
        reset.assert_called_with("cam-a")
        forget.assert_called_with("cam-a")
        self.assertNotIn("cam-a", mgr._sessions)

    def test_render_only_watched_default_on(self):
        with mock.patch.dict(os.environ, {"ML_LIVE_RENDER_ONLY_WHEN_WATCHED": ""}):
            os.environ.pop("ML_LIVE_RENDER_ONLY_WHEN_WATCHED", None)
            self.assertTrue(live_stream.LiveStreamManager()._render_only_watched)

    def test_mjpeg_generator_ends_when_session_gone(self):
        mgr = _manager()
        mgr._frame_interval = 0.01

        async def consume():
            parts = 0
            async for _part in mgr.aiter_mjpeg("cam-a", mode="raw"):
                parts += 1
                if parts == 1:
                    with mgr._lock:
                        mgr._detach_session_locked("cam-a")
            return parts

        self.assertEqual(asyncio.run(asyncio.wait_for(consume(), 5)), 1)

    def test_mjpeg_generator_ends_on_disconnect_and_idle(self):
        mgr = _manager()

        async def disconnected():
            return True

        async def consume(**kw):
            return [p async for p in mgr.aiter_mjpeg("cam-a", **kw)]

        self.assertEqual(asyncio.run(asyncio.wait_for(consume(mode="raw", is_disconnected=disconnected), 5)), [])
        with mock.patch.object(live_stream, "_mjpeg_idle_timeout", return_value=0.2):
            # live mode with no rendered JPEG ever → idle timeout ends the stream
            mgr._render_only_watched = False
            out = asyncio.run(asyncio.wait_for(consume(mode="live"), 5))
        self.assertEqual(out, [])


class StaleDangerTests(unittest.TestCase):
    def test_stale_box_does_not_start_danger_clip(self):
        tmp = tempfile.mkdtemp(prefix="motion-clips-")
        self.addCleanup(shutil.rmtree, tmp, True)
        rec = MotionClipRecorder(ClipConfig(out_dir=tmp))
        gun = {"class_name": "gun", "model": "weapon", "bbox": [50, 50, 60, 60], "stale": True}
        rec.on_motion("cam-1", [], [], 1000.0, [gun])
        self.assertIsNone(rec._cameras["cam-1"].recording)
        fresh = dict(gun, stale=False)
        rec.on_motion("cam-1", [], [], 1001.0, [fresh])
        self.assertIsNotNone(rec._cameras["cam-1"].recording)
        # stale copy keeps the running danger clip alive
        rec.on_motion("cam-1", [], [], 1002.0, [gun])
        mv = next(iter(rec._cameras["cam-1"].recording.movements.values()))
        self.assertEqual(mv.last_seen, 1002.0)


class OrphanSweepTests(unittest.TestCase):
    def test_sweep_removes_old_orphans_only(self):
        tmp = Path(tempfile.mkdtemp(prefix="motion-clips-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        rec = MotionClipRecorder(ClipConfig(out_dir=str(tmp)))
        for name in ("a.part.mp4", "b.mp4", "b.jpg", "c.mp4", "c.json", "d.json.tmp", "new.mp4"):
            (tmp / name).write_bytes(b"x")
        old = time.time() - 3600
        for name in ("a.part.mp4", "b.mp4", "b.jpg", "c.mp4", "c.json", "d.json.tmp"):
            os.utime(tmp / name, (old, old))
        removed = rec.sweep_orphans(max_age_sec=600)
        self.assertEqual(removed, 4)
        self.assertEqual(sorted(p.name for p in tmp.iterdir()), ["c.json", "c.mp4", "new.mp4"])


if __name__ == "__main__":
    unittest.main()
