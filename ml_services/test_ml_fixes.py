"""Run: python -m unittest test_ml_fixes  (from ml_services/).

Focused, GPU-free / weight-free tests for the audit fixes: unknown-label handling, plate key
normalisation + date filter, captures.csv lock/atomic write, freeze detection, face gallery
swap under the lock, offline job TTL pruning, ReID fallback policy, calibration, journey
ingest queue and Find-in-Video face coordinate scaling.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


# --------------------------------------------------------------------------- unknown labels
class UnknownLabelTests(unittest.TestCase):
    def test_unknown_variants(self):
        from face_recognizer import is_unknown_label

        for label in ("", "  ", "unknown", "Unknown", "person", "face", "unknown:U0007", "UNKNOWN:U0001"):
            self.assertTrue(is_unknown_label(label), label)
        for label in ("Ali Khan", "visitor:12", "U0007"):
            self.assertFalse(is_unknown_label(label), label)


# --------------------------------------------------------------------------- plates
class PlateKeyTests(unittest.TestCase):
    def setUp(self):
        import plate_recognizer as pr

        self.pr = pr

    def test_isb_and_ict_series_stay_distinct(self):
        pr = self.pr
        self.assertEqual(pr.canonicalize_plate("ISB123"), "ISB123")
        self.assertEqual(pr.canonicalize_plate("ICT123"), "ICT123")
        self.assertFalse(pr.plates_are_same_vehicle("ISB123", "ICT123", 0.9, 0.9))
        self.assertNotEqual(pr.canonicalize_plate("ISB 123"), pr.canonicalize_plate("ICT 123"))

    def test_region_words_stripped_only_at_edges(self):
        pr = self.pr
        self.assertEqual(pr.canonicalize_plate("ICT ABC 123"), "ABC123")
        self.assertEqual(pr.canonicalize_plate("LEA 1234 PUNJAB"), "LEA1234")
        self.assertEqual(pr.canonicalize_plate("PUNJABLEA1234"), "LEA1234")  # long word glued
        self.assertEqual(pr.canonicalize_plate("ABC PAK 123"), "ABCPAK123")  # mid-plate kept
        self.assertEqual(pr.canonicalize_plate("APICT12"), "APICT12")  # substring never removed

    def test_canonicalize_is_idempotent(self):
        pr = self.pr
        for text in ("ICT ABC 123", "LEA 1234", "ISB123", "MAR 123"):
            key = pr.canonicalize_plate(text)
            self.assertEqual(pr.canonicalize_plate(pr.format_plate_display(key)), key, text)

    def test_month_and_weekday_series_are_valid_plates(self):
        pr = self.pr
        for plate in ("MAR 123", "SUN 4567", "MAR123", "LEA 2019"):
            self.assertFalse(pr.looks_like_datetime_ocr(plate), plate)
            self.assertTrue(pr.is_valid_plate(plate, min_len=4), plate)

    def test_whole_date_text_is_rejected(self):
        pr = self.pr
        for text in ("TUE 23 NOV 2026", "05082026", "2026 11 23", "MON 2026", "12:30:45"):
            self.assertTrue(pr.looks_like_datetime_ocr(text), text)
            self.assertFalse(pr.is_valid_plate(text, min_len=4), text)

    def test_osd_band_text_uses_aggressive_filter(self):
        pr = self.pr
        self.assertFalse(pr.looks_like_datetime_ocr("TUE 014"))
        self.assertTrue(pr.looks_like_datetime_ocr("TUE 014", from_osd_band=True))
        self.assertFalse(pr.is_valid_plate("MAR123", min_len=4, from_osd_band=True))


class CaptureCsvLockTests(unittest.TestCase):
    def setUp(self):
        import plate_recognizer as pr

        self.pr = pr
        self.dir = Path(tempfile.mkdtemp(prefix="plate-csv-"))
        self.csv = self.dir / "captures.csv"

    def test_lock_file_protocol(self):
        with self.pr.captures_csv_lock(self.csv) as lock_path:
            self.assertEqual(lock_path.name, "captures.csv.lock")
            pid, ts = lock_path.read_text(encoding="ascii").split()
            self.assertEqual(int(pid), os.getpid())
            self.assertAlmostEqual(int(ts), time.time(), delta=5)
            with self.assertRaises(self.pr.CsvLockTimeout):
                with self.pr.captures_csv_lock(self.csv, timeout=0.2):
                    pass
        self.assertFalse((self.dir / "captures.csv.lock").exists())

    def test_stale_lock_is_broken(self):
        lock = self.dir / "captures.csv.lock"
        lock.write_text("999999 0\n", encoding="ascii")
        old = time.time() - 120
        os.utime(lock, (old, old))
        with self.pr.captures_csv_lock(self.csv, timeout=1.0, stale_after=30):
            self.assertTrue(lock.exists())
        self.assertFalse(lock.exists())

    def test_atomic_write_and_upsert(self):
        pr = self.pr
        rows = [{"timestamp": "2026-11-23T10:00:00", "camera_key": "cam-1", "plate_number": "ABC 123",
                 "det_conf": 0.9, "ocr_conf": 0.8, "plate_image": "licence plates/plates/a.jpg",
                 "frame_image": "licence plates/frames/a.jpg"}]
        pr.write_csv_atomic(self.csv, rows)
        self.assertEqual(list(self.dir.glob(".captures.csv.*.tmp")), [])
        engine = pr.PlateEngine.__new__(pr.PlateEngine)
        engine._captures_csv = lambda: self.csv  # type: ignore[method-assign]
        engine._upsert_csv_row({**rows[0], "ocr_conf": 0.95}, match_plate_rel="licence plates/plates/a.jpg")
        engine._upsert_csv_row({**rows[0], "plate_number": "XYZ 9", "plate_image": "b.jpg"})
        import csv

        with self.csv.open(encoding="utf-8", newline="") as f:
            got = list(csv.DictReader(f))
        self.assertEqual([r["plate_number"] for r in got], ["ABC 123", "XYZ 9"])
        self.assertEqual(got[0]["ocr_conf"], "0.95")
        self.assertFalse((self.dir / "captures.csv.lock").exists())

    def test_clear_camera_tracks_takes_save_lock(self):
        pr = self.pr
        engine = pr.PlateEngine.__new__(pr.PlateEngine)
        engine._save_lock = threading.Lock()
        engine._visits = {"a": {"camera_key": "cam-1"}, "b": {"camera_key": "cam-2"}}
        engine._save_lock.acquire()
        t = threading.Thread(target=engine.clear_camera_tracks, args=("cam-1",))
        t.start()
        time.sleep(0.1)
        self.assertIn("a", engine._visits)  # blocked on the lock
        engine._save_lock.release()
        t.join(2)
        self.assertEqual(list(engine._visits), ["b"])


# --------------------------------------------------------------------------- freeze
class FreezeTests(unittest.TestCase):
    def test_low_fps_alone_is_not_a_freeze(self):
        from cam_health.freeze import freeze_score

        self.assertLess(freeze_score(0.6, fps=0.3), 0.85)
        self.assertLess(freeze_score(0.02, fps=0.5), 0.85)  # static scene, not identical

    def test_identical_fingerprints_need_a_streak(self):
        from cam_health.freeze import freeze_score, identical_streak

        s1 = identical_streak(0.0, 0)
        s2 = identical_streak(0.0, s1)
        s3 = identical_streak(0.0, s2)
        self.assertEqual((s1, s2, s3), (1, 2, 3))
        self.assertLess(freeze_score(0.0, streak=s1), 0.85)
        self.assertLess(freeze_score(0.0, streak=s2), 0.85)
        self.assertGreaterEqual(freeze_score(0.0, streak=s3), 0.85)
        self.assertEqual(identical_streak(0.3, 5), 0)

    def test_frame_counter_is_authoritative(self):
        from cam_health.freeze import freeze_score

        self.assertEqual(freeze_score(0.5, frame_seq=10, previous_frame_seq=10), 1.0)
        self.assertLess(freeze_score(0.0, frame_seq=11, previous_frame_seq=10), 0.85)
        self.assertEqual(freeze_score(0.5, frame_ts=100.0, previous_frame_ts=100.0), 1.0)
        self.assertLess(freeze_score(0.0, frame_ts=101.0, previous_frame_ts=100.0), 0.85)

    def test_analyzer_streak_and_alerts(self):
        from cam_health import analyze_camera_health

        rng = np.random.default_rng(0)
        frame = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
        first = analyze_camera_health(frame, fps=0.5)
        fp = first["image"]["fingerprint"]
        types_ = lambda r: {a["alert_type"] for a in r["alerts"]}  # noqa: E731
        self.assertNotIn("frozen_frame", types_(first))
        streak = 0
        res = first
        for _ in range(3):
            res = analyze_camera_health(frame, previous_fingerprint=fp, fps=0.5,
                                        previous_identical_streak=streak)
            streak = res["stream"]["identical_streak"]
        self.assertEqual(streak, 3)
        self.assertIn("frozen_frame", types_(res))
        one = analyze_camera_health(frame, previous_fingerprint=fp, fps=0.5)
        self.assertNotIn("frozen_frame", types_(one))
        self.assertIn("possible_frozen_frame", types_(one))


# --------------------------------------------------------------------------- face gallery swap
class _FakeRecognizer:
    def match(self, a, b, _mode):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _fake_db():
    from face_recognizer import KnownFaceDB, TrackRecognitionCache, UnknownFaceCache, _FaissGallery

    db = KnownFaceDB.__new__(KnownFaceDB)
    db.threshold = 0.5
    db.known = []
    db.enrollment_min_quality = 0.0
    db.faiss_min_embeddings = 0
    db.faiss_top_k = 5
    db._lock = threading.Lock()
    db._build_lock = threading.Lock()
    db._replace_pending = False
    db._faiss = _FaissGallery()
    db.unknown_cache = UnknownFaceCache()
    db.track_cache = TrackRecognitionCache()
    db.recognizer = _FakeRecognizer()
    return db


class GallerySwapTests(unittest.TestCase):
    def test_reload_never_exposes_an_empty_gallery(self):
        db = _fake_db()
        db.apply_db_embeddings([{"identity": "Ali", "embedding": [1.0, 0.0]}])
        db.track_cache.put("cam", 1, identity="unknown:U0001", score=0.0, embedding=[])

        building = threading.Event()
        release = threading.Event()
        real_build = db._build_index

        def slow_build(known):
            building.set()
            release.wait(2)
            return real_build(known)

        db._build_index = slow_build  # type: ignore[method-assign]
        db.reload()
        t = threading.Thread(
            target=db.apply_db_embeddings,
            args=([{"identity": "Bilal", "embedding": [0.0, 1.0]}],),
        )
        t.start()
        self.assertTrue(building.wait(2))
        # While the new gallery is being built, matching still sees the old one.
        with db._lock:
            self.assertEqual([r.identity for r in db.known], ["Ali"])
        result = db.recognize_feature(np.array([[1.0, 0.0]], dtype=np.float32), use_unknown_cache=False)
        self.assertEqual(result.identity, "Ali")
        release.set()
        t.join(2)
        self.assertEqual([r.identity for r in db.known], ["Bilal"])
        self.assertIsNone(db.track_cache.get("cam", 1))  # stale track identities cleared
        self.assertEqual(
            db.recognize_feature(np.array([[0.0, 1.0]], dtype=np.float32), use_unknown_cache=False).identity,
            "Bilal",
        )

    def test_apply_without_reload_appends(self):
        db = _fake_db()
        db.apply_db_embeddings([{"identity": "Ali", "embedding": [1.0, 0.0]}])
        db.apply_db_embeddings([{"identity": "Bilal", "embedding": [0.0, 1.0]}])
        self.assertEqual(sorted(r.identity for r in db.known), ["Ali", "Bilal"])
        self.assertEqual(db.replace_embeddings([{"identity": "Zara", "embedding": [1.0, 1.0]}]), 1)
        self.assertEqual([r.identity for r in db.known], ["Zara"])

    def test_offline_lookup_does_not_mint_unknown_ids(self):
        db = _fake_db()
        db.apply_db_embeddings([{"identity": "Ali", "embedding": [1.0, 0.0]}])
        res = db.recognize_feature(np.array([[0.0, 1.0]], dtype=np.float32), use_unknown_cache=False)
        self.assertEqual(res.identity, "unknown")
        self.assertEqual(db.unknown_cache.size, 0)


# --------------------------------------------------------------------------- job TTL pruning
class JobPruneTests(unittest.TestCase):
    def _check(self, mod, make_row):
        root = Path(tempfile.mkdtemp(prefix=f"{mod.__name__}-"))
        with mock.patch.object(mod, "_TMP", root), mock.patch.dict(mod._JOBS, clear=True):
            old_dir = root / "old"
            old_dir.mkdir()
            new_dir = root / "new"
            new_dir.mkdir()
            run_dir = root / "running"
            run_dir.mkdir()
            orphan = root / "orphan"
            orphan.mkdir()
            now = time.time()
            ancient = now - 10 * 3600
            for p in (old_dir, run_dir, orphan):
                os.utime(p, (ancient, ancient))
            mod._JOBS["old"] = make_row("old", "done", ancient, old_dir)
            mod._JOBS["new"] = make_row("new", "error", now, new_dir)
            mod._JOBS["run"] = make_row("run", "running", ancient, run_dir)
            removed = mod.prune_jobs(now=now, ttl_sec=6 * 3600)
            self.assertEqual(removed, 1)
            self.assertEqual(sorted(mod._JOBS), ["new", "run"])
            self.assertFalse(old_dir.exists())
            self.assertFalse(orphan.exists())  # unreferenced + older than TTL
            self.assertTrue(new_dir.exists())
            self.assertTrue(run_dir.exists())  # running jobs are never touched
        return root

    def test_search_jobs(self):
        import video_search_jobs as mod

        self._check(mod, lambda jid, st, ts, d: {
            "job_id": jid, "status": st, "updated_at": ts, "work_dir": str(d)})

    def test_analyze_jobs_delete_outputs(self):
        import video_analyze_jobs as mod

        outputs = {}

        def row(jid, st, ts, d):
            out = d / f"{jid}_tagged.mp4"
            out.write_bytes(b"x")
            os.utime(out, (ts, ts))
            outputs[jid] = out
            return {"job_id": jid, "status": st, "updated_at": ts, "work_dir": str(d),
                    "output_path": str(out)}

        self._check(mod, row)
        self.assertFalse(outputs["old"].exists())
        self.assertTrue(outputs["run"].exists())

    def test_prune_never_deletes_outside_tmp(self):
        import video_analyze_jobs as mod

        root = Path(tempfile.mkdtemp(prefix="vaj-root-"))
        outside = Path(tempfile.mkdtemp(prefix="vaj-outside-")) / "keep.mp4"
        outside.write_bytes(b"x")
        with mock.patch.object(mod, "_TMP", root), mock.patch.dict(mod._JOBS, clear=True):
            mod._JOBS["x"] = {"job_id": "x", "status": "done", "updated_at": 0,
                              "output_path": str(outside), "work_dir": str(outside.parent)}
            mod.prune_jobs(now=time.time(), ttl_sec=60)
        self.assertTrue(outside.exists())

    def test_analyze_output_goes_under_tmp(self):
        import video_analyze_jobs as mod

        root = Path(tempfile.mkdtemp(prefix="vaj-out-"))
        src = Path(tempfile.mkdtemp(prefix="vaj-src-")) / "clip.mp4"
        with mock.patch.object(mod, "_TMP", root), mock.patch.dict(mod._JOBS, clear=True), \
                mock.patch.object(mod, "_EXECUTOR") as executor:
            job_id = mod.start_analyze_job(str(src), person=True, vehicle=False, weapon=False,
                                           fire=False, match_staff=False, sample_fps=1.0,
                                           cleanup_video=False)
            out = Path(mod._JOBS[job_id]["output_path"])
            self.assertEqual(out.parent, root / "outputs")
            self.assertNotEqual(out.parent, src.parent)
            self.assertTrue(executor.submit.called)


# --------------------------------------------------------------------------- ReID policy
class ReidPolicyTests(unittest.TestCase):
    def setUp(self):
        import reid_extractor as rx

        self.rx = rx
        self._saved = (rx._osnet_model, rx._osnet_failed, rx._osnet_device)
        rx._osnet_model, rx._osnet_failed = None, False
        self.crop = np.random.default_rng(1).integers(0, 255, (160, 64, 3), dtype=np.uint8)

    def tearDown(self):
        self.rx._osnet_model, self.rx._osnet_failed, self.rx._osnet_device = self._saved

    def test_strict_osnet_failure_returns_empty(self):
        rx = self.rx
        with mock.patch.dict(os.environ, {"ML_REID_BACKEND": "osnet", "ML_REID_ALLOW_HANDCRAFTED_FALLBACK": "0"}), \
                mock.patch.object(rx, "_resolve_osnet_weights", return_value=None):
            self.assertEqual(rx.extract_reid_embedding(self.crop), [])
            self.assertEqual(rx.extract_reid_embeddings([self.crop, self.crop]), [[], []])
            self.assertEqual(rx.reid_model_name(), "")

    def test_fallback_flag_allows_handcrafted(self):
        rx = self.rx
        with mock.patch.dict(os.environ, {"ML_REID_BACKEND": "osnet", "ML_REID_ALLOW_HANDCRAFTED_FALLBACK": "1"}), \
                mock.patch.object(rx, "_resolve_osnet_weights", return_value=None):
            vec = rx.extract_reid_embedding(self.crop)
            self.assertEqual(len(vec), rx.REID_DIM)
            batch = rx.extract_reid_embeddings([self.crop, None, self.crop])
            # First crop is no longer swallowed by the model-loading attempt.
            self.assertEqual([len(v) for v in batch], [rx.REID_DIM, 0, rx.REID_DIM])
            self.assertEqual(rx.reid_model_name(), "handcrafted")

    def test_parts_backend_is_handcrafted(self):
        rx = self.rx
        with mock.patch.dict(os.environ, {"ML_REID_BACKEND": "parts"}):
            vec, model = rx.extract_reid_embedding_tagged(self.crop)
            self.assertEqual((len(vec), model), (rx.REID_DIM, "handcrafted"))

    def test_lazy_load_runs_once_across_threads(self):
        rx = self.rx
        calls = []

        def fake_weights():
            calls.append(1)
            time.sleep(0.05)
            return None

        with mock.patch.dict(os.environ, {"ML_REID_BACKEND": "osnet"}), \
                mock.patch.object(rx, "_resolve_osnet_weights", side_effect=fake_weights):
            threads = [threading.Thread(target=rx._load_osnet) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(2)
        self.assertEqual(len(calls), 1)


# --------------------------------------------------------------------------- calibration
class CalibrationTests(unittest.TestCase):
    def test_requires_both_lists(self):
        from face_calibration import suggest_threshold

        self.assertIsNone(suggest_threshold([0.8, 0.9], [])["threshold"])
        self.assertIsNone(suggest_threshold([], [0.1])["threshold"])

    def test_min_half_total_error_and_alias(self):
        from face_calibration import suggest_threshold

        out = suggest_threshold([0.7, 0.8, 0.9], [0.1, 0.2, 0.3], target_far=0.0)
        self.assertEqual(out["min_half_total_error"], 0.0)
        self.assertEqual(out["eer"], out["min_half_total_error"])
        self.assertEqual(out["eer_threshold"], out["min_half_total_error_threshold"])
        self.assertEqual(out["min_half_total_error_threshold"], 0.7)


# --------------------------------------------------------------------------- journey sender
def _import_journey_manager():
    """journey_manager without its heavy deps (YOLO / live stream are stubbed)."""
    fake_pipeline = types.ModuleType("journey_pipeline")
    fake_pipeline.JourneyCameraPipeline = object
    fake_live = types.ModuleType("live_stream")
    fake_live.get_live_manager = lambda: None
    with mock.patch.dict(sys.modules, {"journey_pipeline": fake_pipeline, "live_stream": fake_live}):
        sys.modules.pop("journey_manager", None)
        mod = importlib.import_module("journey_manager")
    sys.modules.pop("journey_manager", None)
    return mod


class JourneySenderTests(unittest.TestCase):
    def test_bounded_queue_drops_oldest(self):
        jm = _import_journey_manager()
        gate = threading.Event()
        sent = []

        def post(payload):
            gate.wait(2)
            sent.append(payload["i"])

        sender = jm._ObservationSender(post, max_size=3)
        with self.assertLogs(jm.logger, level="WARNING") as logs:
            for i in range(10):
                sender.submit({"i": i})
        stats = sender.stats()
        self.assertLessEqual(stats["queued"], 3)
        self.assertGreaterEqual(stats["dropped_total"], 6)
        self.assertTrue(any("dropped" in line for line in logs.output))
        gate.set()
        deadline = time.monotonic() + 2
        while sender.stats()["queued"] and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
        self.assertEqual(sent[-3:], [7, 8, 9])  # newest kept

    def test_manager_singleton_is_locked(self):
        jm = _import_journey_manager()
        with mock.patch.object(jm, "JourneyManager", side_effect=lambda: (time.sleep(0.05), object())[1]) as ctor:
            got = []
            threads = [threading.Thread(target=lambda: got.append(jm.get_journey_manager())) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(2)
        self.assertEqual(ctor.call_count, 1)
        self.assertEqual(len({id(g) for g in got}), 1)


# --------------------------------------------------------------------------- find-in-video face
class SearchFaceScaleTests(unittest.TestCase):
    def test_face_scaled_back_to_original(self):
        from video_search_face import _scale_face_to_original

        face = np.array([20, 40, 60, 80] + [10] * 10 + [0.9], dtype=np.float32)
        out = _scale_face_to_original(face, 2.0)
        self.assertEqual(out[:4].tolist(), [10, 20, 30, 40])
        self.assertEqual(out[4:14].tolist(), [5] * 10)
        self.assertAlmostEqual(float(out[14]), 0.9, places=5)  # score untouched
        self.assertEqual(face[0], 20)  # input not mutated


if __name__ == "__main__":
    unittest.main()
