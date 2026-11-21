"""
Live multi-camera inference load test (Stage 0 baseline).

Two modes:

  synthetic  Loads the real LiveStreamManager + models in THIS process and feeds N fake
             cameras that replay frames from one video file (already decoded, 1280x720,
             same as the scaled RTSP reader). Steps through camera counts, e.g. 10,25,50,86,100.
             Measures the inference side only (no RTSP / decode cost).
             Stop the ML service first: both would share the GPU.

      python live_load_test.py synthetic --video sample.mp4 --cams 10,25,50,86,100 \
          --duration 60 --mix "general_objects:0.6,face_recognition:0.2,smoke_fire+weapon:0.2"

  observe    Polls /live/metrics of the RUNNING service with real cameras for --duration
             seconds and reports the same table (includes real decode / RTSP effects).

      python live_load_test.py observe --url http://127.0.0.1:8100 --duration 120

Synthetic mode has no side effects: motion clips are not recorded, plate snapshots are
not saved, and global object IDs use a temporary counter file.

Each step reports cameras receiving inference within the SLA (ML_LIVE_SLA_SEC, default
max(1 s, 2 x ML_LIVE_INFER_INTERVAL)), per-camera gap p50/p95, queue delay, per-model
throughput / batch / latency / lock wait, GPU / CPU / RAM. JSON goes to --out.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


# ── report helpers ───────────────────────────────────────────────────────────
def _median(vals: list[float]) -> float:
    return float(statistics.median(vals)) if vals else 0.0


def summarize(snap: dict[str, Any], label: str) -> dict[str, Any]:
    s = snap.get("summary", {})
    cams = [c for c in snap.get("cameras", []) if c.get("connected")]
    models = snap.get("models", {})
    system = snap.get("system", {})
    host = system.get("host", {})
    gpu = system.get("gpu", {})
    return {
        "step": label,
        "cameras": s.get("cameras_connected", 0),
        "receiving": s.get("cameras_receiving", 0),
        "starved": s.get("cameras_starved", 0),
        "sla_sec": snap.get("sla_sec"),
        "sla_met": bool(s.get("sla_met")),
        "total_infer_fps": s.get("total_infer_fps", 0.0),
        "cam_fps_median": round(_median([c["infer_fps"] for c in cams]), 3),
        "cam_fps_min": round(min((c["infer_fps"] for c in cams), default=0.0), 3),
        "gap_p50_median": round(_median([c["gap_p50_sec"] for c in cams]), 3),
        "gap_p95_median": round(_median([c["gap_p95_sec"] for c in cams]), 3),
        "gap_p95_worst": s.get("worst_gap_p95_sec", 0.0),
        "queue_delay_p95_median": round(_median([c["queue_delay_p95_sec"] for c in cams]), 3),
        "latency_p95_median": round(_median([c["latency_p95_sec"] for c in cams]), 3),
        "stale_frames": sum(c.get("stale_frames", 0) for c in cams),
        "errors": sum(c.get("errors", 0) for c in cams),
        "gpu_model_busy_pct": s.get("gpu_model_busy_pct", 0.0),
        "gpu_util_pct": gpu.get("util_pct"),
        "gpu_mem_used_mb": gpu.get("mem_used_mb"),
        "gpu_decoder_pct": gpu.get("decoder_util_pct"),
        "cpu_pct": host.get("cpu_pct"),
        "process_cpu_pct": host.get("process_cpu_pct"),
        "process_rss_mb": host.get("process_rss_mb"),
        "models": {
            k: {
                "img_s": v["images_per_sec"],
                "batch": v["avg_batch"],
                "p50_ms": v["latency_p50_ms"],
                "p95_ms": v["latency_p95_ms"],
                "wait_ms": v["lock_wait_avg_ms"],
                "busy_pct": v["busy_pct"],
            }
            for k, v in models.items()
        },
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        ("step", 10),
        ("cameras", 7),
        ("receiving", 9),
        ("starved", 7),
        ("total_infer_fps", 9),
        ("cam_fps_median", 8),
        ("gap_p95_median", 8),
        ("gap_p95_worst", 8),
        ("queue_delay_p95_median", 8),
        ("gpu_model_busy_pct", 8),
        ("gpu_util_pct", 6),
        ("cpu_pct", 6),
        ("sla_met", 6),
    ]
    heads = ["step", "cams", "recv", "starved", "fps_all", "fps/cam", "gap95", "gap95max", "qdelay95", "gpuBusy%", "gpu%", "cpu%", "SLA"]
    print()
    print("  ".join(h.ljust(w) for h, (_, w) in zip(heads, cols)))
    for r in rows:
        cells = []
        for key, w in cols:
            v = r.get(key)
            if isinstance(v, bool):
                v = "PASS" if v else "FAIL"
            cells.append(str("-" if v is None else v).ljust(w))
        print("  ".join(cells))
    print()
    for r in rows:
        print(f"[{r['step']}] per model (images/s, avg batch, p95 ms, avg lock wait ms, busy %):")
        for name, m in sorted(r["models"].items(), key=lambda kv: -kv[1]["busy_pct"]):
            print(
                f"    {name:18s} {m['img_s']:8.2f} img/s  batch {m['batch']:5.2f}  "
                f"p95 {m['p95_ms']:7.1f} ms  wait {m['wait_ms']:7.1f} ms  busy {m['busy_pct']:6.1f}%"
            )


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Report written: {path}")


# ── synthetic mode ───────────────────────────────────────────────────────────
class ReplayStream:
    """Fake camera reader with the same interface as FfmpegCameraStream (latest frame only)."""

    def __init__(self, label: str, frames: list, fps: float, offset: int) -> None:
        self.label = label
        self.rtsp_url = f"synthetic://{label}"
        self.keep_native = False
        self._frames = frames
        self._interval = 1.0 / max(1.0, fps)
        self._idx = offset % len(frames)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_seq = 0
        self.frame_ts = 0.0
        self.running = True
        self.connected = False
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{label}")

    def _run(self) -> None:
        nxt = time.time()
        while self.running:
            frame = self._frames[self._idx]
            self._idx = (self._idx + 1) % len(self._frames)
            with self.lock:
                self.frame = frame  # shared read-only array; get_latest copies like the real reader
                self.frame_seq += 1
                self.frame_ts = time.time()
                self.connected = True
            nxt += self._interval
            time.sleep(max(0.0, nxt - time.time()))

    def get_latest(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_seq
            return self.frame.copy(), self.frame_seq

    def get_latest_meta(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_seq, 0.0
            return self.frame.copy(), self.frame_seq, self.frame_ts

    def get_frame(self):
        return self.get_latest()[0]

    def stop(self) -> None:
        self.running = False


def load_frames(video: str, max_frames: int, width: int, height: int) -> list:
    import cv2

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max_frames) if total > 0 else 1
    frames = []
    i = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            scale = min(width / float(w), height / float(h), 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        i += 1
    cap.release()
    if not frames:
        raise SystemExit(f"No frames decoded from {video}")
    return frames


def parse_mix(spec: str) -> list[tuple[list[str], float]]:
    """'general_objects:0.6,face_recognition:0.2,smoke_fire+weapon:0.2' → [(purposes, weight)]."""
    out: list[tuple[list[str], float]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, _, weight = part.partition(":")
        purposes = [p.strip() for p in name.split("+") if p.strip()]
        out.append((purposes, float(weight) if weight else 1.0))
    if not out:
        out = [(["general_objects"], 1.0)]
    return out


def assign_purposes(n: int, mix: list[tuple[list[str], float]]) -> list[list[str]]:
    """Deterministic split of n cameras by weight (largest remainder)."""
    total = sum(w for _, w in mix) or 1.0
    exact = [n * w / total for _, w in mix]
    counts = [int(e) for e in exact]
    for i in sorted(range(len(mix)), key=lambda i: -(exact[i] - counts[i]))[: n - sum(counts)]:
        counts[i] += 1
    out: list[list[str]] = []
    for (purposes, _), c in zip(mix, counts):
        out.extend([list(purposes)] * c)
    return out


def service_running(url: str) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2):
            return True
    except Exception:
        return False


def run_synthetic(args: argparse.Namespace) -> None:
    if not args.force and service_running(args.url):
        raise SystemExit(
            f"ML service is answering at {args.url}. Stop it first (it shares the GPU), "
            "or pass --force to run anyway."
        )
    from load_env import load_ml_env

    load_ml_env()
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        os.environ[k.strip()] = v.strip()

    import live_stream
    from object_identity import ObjectIdentityRegistry

    frames = load_frames(args.video, args.frames, 1280, 720)
    print(f"Loaded {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]}) from {args.video}")

    mgr = live_stream.LiveStreamManager()
    # ── no side effects: clips, global-ID counters, plate snapshots ──
    mgr._clips.start = lambda: None  # type: ignore[method-assign]
    mgr._clips.submit = lambda *a, **k: None  # type: ignore[method-assign]
    tmpdir = Path(tempfile.mkdtemp(prefix="live_load_"))
    mgr._object_identity = ObjectIdentityRegistry(counter_file=tmpdir / "identity_counters.json")

    mgr.start()
    if not mgr._running:
        raise SystemExit("LiveStreamManager failed to start (weights / GPU?) — see log above.")
    if mgr._plate_engine is not None:
        real = mgr._plate_engine.detect_and_read

        def _no_save(*a, **k):
            k["save"] = False
            return real(*a, **k)

        mgr._plate_engine.detect_and_read = _no_save  # type: ignore[method-assign]

    mix = parse_mix(args.mix)
    steps = sorted({int(x) for x in args.cams.split(",") if x.strip()})
    plan = assign_purposes(max(steps), mix)
    rnd = random.Random(7)
    rnd.shuffle(plan)

    rows: list[dict[str, Any]] = []
    snaps: list[dict[str, Any]] = []
    streams: list[ReplayStream] = []
    try:
        for n in steps:
            while len(streams) < n:
                i = len(streams)
                key = f"load-{i + 1:03d}"
                stream = ReplayStream(key, frames, args.fps, offset=i * 7)
                stream.thread.start()
                streams.append(stream)
                with mgr._lock:
                    mgr._registry[key] = stream.rtsp_url
                    mgr._purposes[key] = list(plan[i])
                    mgr._sessions[key] = live_stream._CameraSession(key, stream, stream.rtsp_url)
            print(f"\n=== {n} cameras: warmup {args.warmup}s, measure {args.duration}s ===")
            time.sleep(args.warmup)
            mgr.metrics(reset=True)
            t_end = time.time() + args.duration
            while time.time() < t_end:
                time.sleep(min(10.0, max(0.0, t_end - time.time())))
                s = mgr.metrics(include_cameras=False)["summary"]
                print(
                    f"  {n} cams: receiving {s['cameras_receiving']}/{s['cameras_connected']}  "
                    f"fps_all {s['total_infer_fps']}  gap95max {s['worst_gap_p95_sec']}s  "
                    f"gpuBusy {s['gpu_model_busy_pct']}%"
                )
            snap = mgr.metrics()
            snaps.append({"cameras": n, "snapshot": snap})
            rows.append(summarize(snap, f"{n}cams"))
            print_table(rows[-1:])
    finally:
        mgr.stop()
        for s in streams:
            s.stop()

    print("\n=== Summary ===")
    print_table(rows)
    out = Path(args.out) if args.out else HERE / "tmp" / f"live_load_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_report(
        out,
        {
            "mode": "synthetic",
            "video": args.video,
            "frames": len(frames),
            "reader_fps": args.fps,
            "mix": args.mix,
            "env_overrides": args.env or [],
            "rows": rows,
            "snapshots": snaps,
        },
    )


# ── observe mode ─────────────────────────────────────────────────────────────
def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_observe(args: argparse.Namespace) -> None:
    base = args.url.rstrip("/")
    print(f"Observing {base}/live/metrics for {args.duration}s (counters reset at start)")
    _get_json(f"{base}/live/metrics?reset=true&cameras=false")
    series: list[dict[str, Any]] = []
    t_end = time.time() + args.duration
    while time.time() < t_end:
        time.sleep(min(args.interval, max(0.0, t_end - time.time())))
        snap = _get_json(f"{base}/live/metrics?cameras=false")
        s = snap["summary"]
        series.append({"at": snap["at"], **{k: v for k, v in s.items() if k != "starved_keys"}})
        print(
            f"  receiving {s['cameras_receiving']}/{s['cameras_connected']}  "
            f"fps_all {s['total_infer_fps']}  gap95max {s['worst_gap_p95_sec']}s  "
            f"gpuBusy {s['gpu_model_busy_pct']}%"
        )
    snap = _get_json(f"{base}/live/metrics")
    row = summarize(snap, f"live{snap['summary']['cameras_connected']}")
    print_table([row])
    starved = snap["summary"].get("starved_keys") or []
    if starved:
        print(f"Starved cameras ({len(starved)}): {', '.join(starved)}")
    out = Path(args.out) if args.out else HERE / "tmp" / f"live_observe_{time.strftime('%Y%m%d_%H%M%S')}.json"
    write_report(out, {"mode": "observe", "url": base, "rows": [row], "series": series, "snapshot": snap})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    sy = sub.add_parser("synthetic", help="in-process load test with replayed video frames")
    sy.add_argument("--video", required=True, help="video file to replay (people/vehicles in view)")
    sy.add_argument("--cams", default="10,25,50,86,100", help="camera counts to step through")
    sy.add_argument("--duration", type=float, default=60.0, help="measure seconds per step")
    sy.add_argument("--warmup", type=float, default=15.0, help="settle seconds before each step")
    sy.add_argument("--fps", type=float, default=12.0, help="frames/s each fake camera produces")
    sy.add_argument("--frames", type=int, default=120, help="frames to preload from the video")
    sy.add_argument(
        "--mix",
        default="general_objects:1",
        help="purpose mix, e.g. 'general_objects:0.6,face_recognition:0.2,smoke_fire+weapon:0.2'",
    )
    sy.add_argument("--env", action="append", help="override env for this run, e.g. --env ML_LIVE_INFER_WORKERS=8")
    sy.add_argument("--url", default="http://127.0.0.1:8100", help="service URL (to refuse if it is running)")
    sy.add_argument("--force", action="store_true", help="run even if the ML service is up")
    sy.add_argument("--out", help="JSON report path (default ml_services/tmp/...)")

    ob = sub.add_parser("observe", help="poll /live/metrics of the running service")
    ob.add_argument("--url", default="http://127.0.0.1:8100")
    ob.add_argument("--duration", type=float, default=120.0)
    ob.add_argument("--interval", type=float, default=10.0)
    ob.add_argument("--out", help="JSON report path (default ml_services/tmp/...)")

    args = ap.parse_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_observe(args)


if __name__ == "__main__":
    main()
