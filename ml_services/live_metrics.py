"""
Live inference metrics (Stage 0 of the multi-camera scheduler work).

Observation only: nothing here changes what gets inferred or when. The live manager,
plate engine, face DB and journey pipeline call the record_* hooks; /live/metrics
returns snapshot().

Per camera  : inference FPS, gap between inferences (p50/p95/max), frame age when
              picked (queue delay), end-to-end latency, frames skipped, stale frames,
              errors, last successful inference time, receiving / starved vs the SLA.
Per model   : calls/s, images/s, batch size, latency p50/p95, lock wait.
System      : CPU, RAM, process RSS, GPU utilisation + memory, torch CUDA memory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any, Callable


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _gpu_index() -> int:
    """Torch-visible GPU index from ML_DEVICE ('0', 'cuda:1'); 0 when it is 'cpu' or unset."""
    raw = (os.getenv("ML_DEVICE", "0") or "0").strip().lower().replace("cuda:", "").replace("cuda", "")
    try:
        return int(raw)
    except ValueError:
        return 0


def gpu_physical_id() -> str:
    """Host GPU id (nvidia-smi / NVML index, or a GPU UUID) of the device this process uses.

    ML_DEVICE is an index into the GPUs that CUDA_VISIBLE_DEVICES exposes, so with
    CUDA_VISIBLE_DEVICES=1 and ML_DEVICE=0 the process really runs on host GPU 1.
    """
    idx = _gpu_index()
    visible = [v.strip() for v in (os.getenv("CUDA_VISIBLE_DEVICES") or "").split(",") if v.strip()]
    if visible and 0 <= idx < len(visible):
        return visible[idx]
    return str(idx)


def ml_node_id() -> str:
    """Short id of this ML process (ML_NODE_ID, else 'gpu<N>' from CUDA_VISIBLE_DEVICES, else '').

    Used to keep per-process files (identity counters, motion clips) apart when two
    processes (gpu0 / gpu1) run from the same directory.
    """
    raw = (os.getenv("ML_NODE_ID") or "").strip()
    if not raw:
        visible = [v.strip() for v in (os.getenv("CUDA_VISIBLE_DEVICES") or "").split(",") if v.strip()]
        if len(visible) == 1:
            raw = f"gpu{visible[0]}"
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in raw)[:40]


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[k])


def _r(value: float, nd: int = 3) -> float:
    return round(float(value), nd)


class _CameraStats:
    __slots__ = (
        "starts",
        "finishes",
        "gaps",
        "ages",
        "latencies",
        "last_start",
        "last_ok_at",
        "last_error",
        "last_error_at",
        "inferences",
        "errors",
        "frames_skipped",
        "stale",
        "no_new_frame",
        "no_frame",
    )

    def __init__(self, maxlen: int) -> None:
        self.starts: deque[float] = deque(maxlen=maxlen)
        self.finishes: deque[float] = deque(maxlen=maxlen)
        self.gaps: deque[tuple[float, float]] = deque(maxlen=maxlen)  # (at, gap)
        self.ages: deque[tuple[float, float]] = deque(maxlen=maxlen)  # (at, frame age at pick)
        self.latencies: deque[tuple[float, float]] = deque(maxlen=maxlen)  # (at, capture→done)
        self.last_start = 0.0
        self.last_ok_at = 0.0
        self.last_error = ""
        self.last_error_at = 0.0
        self.inferences = 0
        self.errors = 0
        self.frames_skipped = 0
        self.stale = 0
        self.no_new_frame = 0
        self.no_frame = 0


class _ModelStats:
    __slots__ = ("events", "calls", "images", "kind")

    def __init__(self, maxlen: int, kind: str) -> None:
        # (at, images, latency, lock_wait)
        self.events: deque[tuple[float, int, float, float]] = deque(maxlen=maxlen)
        self.calls = 0
        self.images = 0
        self.kind = kind


def _thread_group(name: str) -> str:
    """Collapse per-camera / per-worker thread names into groups."""
    if not name:
        return "native (torch/opencv/cuda pools)"
    n = name.lower()
    for prefix, group in (
        ("cam-", "camera readers (pipe + mjpeg decode)"),
        ("fferr-", "ffmpeg stderr drain"),
        ("live-infer", "live inference workers"),
        ("live-render", "live render (draw + jpeg)"),
        ("motion-clip", "motion clip recorder"),
        ("clip", "motion clip recorder"),
        ("journey", "journey pipeline"),
        ("live-boot", "camera boot"),
        ("anyio", "http server"),
        ("asyncio", "http server"),
        ("threadpoolexecutor", "http server"),
        ("metrics-gil", "metrics probe"),
        ("mainthread", "main"),
    ):
        if n.startswith(prefix):
            return group
    return name


class _ThreadCpuSampler:
    """CPU cores used per thread group (+ ffmpeg child processes) between two samples."""

    def __init__(self) -> None:
        self._prev: dict[int, float] = {}
        self._prev_at = 0.0
        self._children: dict[int, Any] = {}
        self._proc = None

    def sample(self) -> dict[str, Any]:
        try:
            import psutil  # type: ignore
        except Exception:
            return {}
        try:
            if self._proc is None:
                self._proc = psutil.Process()
            now = time.time()
            names = {t.native_id: t.name for t in threading.enumerate() if getattr(t, "native_id", None)}
            cur: dict[int, float] = {}
            groups: dict[str, dict[str, float]] = {}
            first = not self._prev
            dt = max(1e-6, now - self._prev_at)
            for th in self._proc.threads():
                total = float(th.user_time + th.system_time)
                cur[th.id] = total
                if first:
                    continue
                delta = max(0.0, total - self._prev.get(th.id, total))
                g = groups.setdefault(_thread_group(names.get(th.id, "")), {"cores": 0.0, "threads": 0})
                g["cores"] += delta / dt
                g["threads"] += 1
            ff_cores = 0.0
            ff_count = 0
            alive: dict[int, Any] = {}
            for child in self._proc.children(recursive=True):
                try:
                    c = self._children.get(child.pid) or child
                    pct = c.cpu_percent(None)  # first call per child primes it (0.0)
                    alive[child.pid] = c
                    if "ffmpeg" in (c.name() or "").lower():
                        ff_cores += pct / 100.0
                        ff_count += 1
                except Exception:
                    continue
            self._children = alive
            self._prev = cur
            self._prev_at = now
            if first:
                return {}
            rows = {k: {"cores": round(v["cores"], 2), "threads": int(v["threads"])} for k, v in groups.items()}
            rows = dict(sorted(rows.items(), key=lambda kv: -kv[1]["cores"]))
            return {
                "interval_sec": round(dt, 1),
                "python_process_cores": round(sum(v["cores"] for v in rows.values()), 2),
                "by_thread_group": rows,
                "ffmpeg_processes": ff_count,
                "ffmpeg_cores": round(ff_cores, 2),
            }
        except Exception as exc:
            return {"error": str(exc)[:120]}


class _SystemSampler:
    """CPU / RAM / GPU readings, cached so /live/metrics stays cheap."""

    def __init__(self, ttl: float = 2.0) -> None:
        self._ttl = ttl
        self._at = 0.0
        self._cached: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._nvml = None
        self._nvml_handle = None
        self._nvml_tried = False
        self._proc = None
        self._threads = _ThreadCpuSampler()

    def _gpu_nvml(self) -> dict[str, Any] | None:
        if not self._nvml_tried:
            self._nvml_tried = True
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                gid = gpu_physical_id()
                self._nvml = pynvml
                if gid.isdigit():
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(int(gid))
                else:
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByUUID(gid)
            except Exception:
                self._nvml = None
        if self._nvml is None:
            return None
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            out: dict[str, Any] = {
                "source": "nvml",
                "util_pct": int(util.gpu),
                "mem_used_mb": int(mem.used // (1024 * 1024)),
                "mem_total_mb": int(mem.total // (1024 * 1024)),
            }
            try:
                dec = self._nvml.nvmlDeviceGetDecoderUtilization(self._nvml_handle)
                out["decoder_util_pct"] = int(dec[0])
            except Exception:
                pass
            return out
        except Exception:
            return None

    @staticmethod
    def _gpu_smi() -> dict[str, Any] | None:
        exe = shutil.which("nvidia-smi")
        if not exe:
            return None
        try:
            raw = subprocess.run(
                [
                    exe,
                    "-i",
                    gpu_physical_id(),
                    "--query-gpu=utilization.gpu,memory.used,memory.total,utilization.decoder",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
            line = raw.splitlines()[0] if raw else ""
            parts = [p.strip() for p in line.split(",")]
            out: dict[str, Any] = {
                "source": "nvidia-smi",
                "util_pct": int(float(parts[0])),
                "mem_used_mb": int(float(parts[1])),
                "mem_total_mb": int(float(parts[2])),
            }
            if len(parts) > 3 and parts[3] not in ("", "[N/A]"):
                out["decoder_util_pct"] = int(float(parts[3]))
            return out
        except Exception:
            return None

    @staticmethod
    def _torch_mem() -> dict[str, Any]:
        try:
            import sys

            torch = sys.modules.get("torch")  # never import torch just for metrics
            if torch is None or not torch.cuda.is_available():
                return {}
            return {
                "torch_allocated_mb": int(torch.cuda.memory_allocated() // (1024 * 1024)),
                "torch_reserved_mb": int(torch.cuda.memory_reserved() // (1024 * 1024)),
            }
        except Exception:
            return {}

    def _host(self) -> dict[str, Any]:
        try:
            import psutil  # type: ignore

            if self._proc is None:
                self._proc = psutil.Process()
                self._proc.cpu_percent(None)
            vm = psutil.virtual_memory()
            return {
                "cpu_pct": psutil.cpu_percent(None),
                "cpu_count": psutil.cpu_count(),
                "ram_used_mb": int(vm.used // (1024 * 1024)),
                "ram_total_mb": int(vm.total // (1024 * 1024)),
                "ram_pct": vm.percent,
                "process_cpu_pct": self._proc.cpu_percent(None),
                "process_rss_mb": int(self._proc.memory_info().rss // (1024 * 1024)),
                "process_threads": self._proc.num_threads(),
            }
        except Exception:
            return {"cpu_pct": None, "note": "psutil not installed"}

    def sample(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            if self._cached and now - self._at < self._ttl:
                return self._cached
            gpu = self._gpu_nvml() or self._gpu_smi() or {}
            gpu.update(self._torch_mem())
            self._cached = {"host": self._host(), "gpu": gpu, "cpu_threads": self._threads.sample(), "sampled_at": now}
            self._at = now
            return self._cached


class LiveMetrics:
    """Thread-safe counters + rolling windows for live inference."""

    def __init__(
        self,
        *,
        window_sec: float | None = None,
        stale_sec: float | None = None,
        clock: Callable[[], float] = time.time,
        sample_system: bool = True,
    ) -> None:
        self.window_sec = window_sec if window_sec is not None else max(5.0, _env_float("ML_METRICS_WINDOW_SEC", 60.0))
        self.stale_sec = stale_sec if stale_sec is not None else max(0.1, _env_float("ML_LIVE_STALE_SEC", 2.0))
        self._clock = clock
        self._lock = threading.Lock()
        self._cams: dict[str, _CameraStats] = {}
        self._models: dict[str, _ModelStats] = {}
        self._stages: dict[str, deque[tuple[float, float]]] = {}
        self._maxlen = 4096
        self._stage_maxlen = 30000
        self._gil_thread: threading.Thread | None = None
        self._since = clock()
        self._system = _SystemSampler() if sample_system else None

    # ── recording hooks ────────────────────────────────────────────────────
    def _cam(self, key: str) -> _CameraStats:
        st = self._cams.get(key)
        if st is None:
            st = _CameraStats(self._maxlen)
            self._cams[key] = st
        return st

    def record_camera_inference(
        self,
        camera_key: str,
        *,
        started_at: float,
        finished_at: float,
        frame_ts: float = 0.0,
        seq: int = 0,
        prev_seq: int = 0,
        ok: bool = True,
        error: str = "",
    ) -> None:
        with self._lock:
            st = self._cam(camera_key)
            if st.last_start > 0:
                st.gaps.append((started_at, started_at - st.last_start))
            st.last_start = started_at
            st.starts.append(started_at)
            if frame_ts > 0:
                age = max(0.0, started_at - frame_ts)
                st.ages.append((started_at, age))
                if age > self.stale_sec:
                    st.stale += 1
            if prev_seq > 0 and seq > prev_seq + 1:
                st.frames_skipped += seq - prev_seq - 1
            if ok:
                st.inferences += 1
                st.finishes.append(finished_at)
                st.last_ok_at = finished_at
                if frame_ts > 0:
                    st.latencies.append((finished_at, max(0.0, finished_at - frame_ts)))
            else:
                st.errors += 1
                st.last_error = (error or "")[:200]
                st.last_error_at = finished_at

    def record_camera_skip(self, camera_key: str, reason: str) -> None:
        """reason: 'no_frame' (reader has nothing) or 'no_new_frame' (same seq as last time)."""
        with self._lock:
            st = self._cam(camera_key)
            if reason == "no_frame":
                st.no_frame += 1
            elif reason == "no_new_frame":
                st.no_new_frame += 1

    def record_model(
        self,
        model: str,
        *,
        images: int = 1,
        latency: float = 0.0,
        lock_wait: float = 0.0,
        kind: str = "gpu",
    ) -> None:
        now = self._clock()
        with self._lock:
            st = self._models.get(model)
            if st is None:
                st = _ModelStats(self._maxlen, kind)
                self._models[model] = st
            st.calls += 1
            st.images += int(images)
            st.events.append((now, int(images), float(latency), float(lock_wait)))

    def record_stage(self, name: str, seconds: float) -> None:
        """One timed step of the pipeline (profiling). Cheap: one lock + deque append."""
        now = self._clock()
        with self._lock:
            dq = self._stages.get(name)
            if dq is None:
                dq = deque(maxlen=self._stage_maxlen)
                self._stages[name] = dq
            dq.append((now, float(seconds)))

    def start_gil_probe(self, period: float = 0.005) -> None:
        """Thread that sleeps `period` and records how late it wakes (GIL / scheduler contention)."""
        if self._gil_thread is not None:
            return

        def _run() -> None:
            while True:
                t0 = time.perf_counter()
                time.sleep(period)
                self.record_stage("sync.gil_wakeup_lag", max(0.0, time.perf_counter() - t0 - period))

        self._gil_thread = threading.Thread(target=_run, daemon=True, name="metrics-gil-probe")
        self._gil_thread.start()

    def forget_camera(self, camera_key: str) -> None:
        with self._lock:
            self._cams.pop(camera_key, None)

    def reset(self) -> None:
        with self._lock:
            self._cams.clear()
            self._models.clear()
            for dq in self._stages.values():
                dq.clear()
            self._since = self._clock()

    # ── snapshot ───────────────────────────────────────────────────────────
    def snapshot(
        self,
        *,
        cameras: dict[str, dict[str, Any]] | None = None,
        sla_sec: float = 1.0,
        target_interval_sec: float = 0.5,
        include_system: bool = True,
    ) -> dict[str, Any]:
        """
        cameras: {key: {"connected": bool, "purposes": [...]}} from the live manager, so cameras
        that never got an inference still show up (as starved) and disconnected ones are
        not counted against the SLA.
        """
        now = self._clock()
        cut = now - self.window_sec
        meta = cameras or {}
        # Copy the raw windows under the lock; percentiles / filtering run outside it so
        # recording hooks on the inference threads are never blocked by a snapshot.
        with self._lock:
            span = max(1e-6, min(self.window_sec, now - self._since))
            keys = sorted(set(meta.keys()) | set(self._cams.keys()))
            cam_copy: dict[str, Any] = {}
            for key, st in self._cams.items():
                cam_copy[key] = {
                    "finishes": list(st.finishes),
                    "gaps": list(st.gaps),
                    "ages": list(st.ages),
                    "latencies": list(st.latencies),
                    "last_ok_at": st.last_ok_at,
                    "inferences": st.inferences,
                    "errors": st.errors,
                    "last_error": st.last_error,
                    "frames_skipped": st.frames_skipped,
                    "stale": st.stale,
                    "no_new_frame": st.no_new_frame,
                    "no_frame": st.no_frame,
                }
            model_copy = {
                name: (ms.kind, list(ms.events), ms.calls, ms.images) for name, ms in self._models.items()
            }
            stage_copy = {
                name: (list(dq), len(dq) == dq.maxlen) for name, dq in self._stages.items() if dq
            }

        cam_rows: list[dict[str, Any]] = []
        connected = receiving = 0
        starved: list[str] = []
        worst_p95_gap = 0.0
        total_fps = 0.0
        for key in keys:
            st = cam_copy.get(key)
            m = meta.get(key, {})
            # With manager metadata, a camera it no longer knows is not "connected".
            is_conn = bool(m.get("connected")) if meta else st is not None
            fin = [t for t in (st["finishes"] if st else ()) if t >= cut]
            gaps = [g for t, g in (st["gaps"] if st else ()) if t >= cut]
            ages = [a for t, a in (st["ages"] if st else ()) if t >= cut]
            lats = [v for t, v in (st["latencies"] if st else ()) if t >= cut]
            fps = len(fin) / span
            last_ok = st["last_ok_at"] if st else 0.0
            since_ok = (now - last_ok) if last_ok else None
            is_recv = bool(is_conn and since_ok is not None and since_ok <= sla_sec)
            if is_conn:
                connected += 1
                if is_recv:
                    receiving += 1
                else:
                    starved.append(key)
            p95_gap = percentile(gaps, 95)
            if is_conn:
                worst_p95_gap = max(worst_p95_gap, p95_gap)
            total_fps += fps
            cam_rows.append(
                {
                    "key": key,
                    "connected": is_conn,
                    "purposes": list(m.get("purposes") or []),
                    "receiving": is_recv,
                    "infer_fps": _r(fps),
                    "gap_p50_sec": _r(percentile(gaps, 50)),
                    "gap_p95_sec": _r(p95_gap),
                    "gap_max_sec": _r(max(gaps) if gaps else 0.0),
                    "queue_delay_p50_sec": _r(percentile(ages, 50)),
                    "queue_delay_p95_sec": _r(percentile(ages, 95)),
                    "latency_p50_sec": _r(percentile(lats, 50)),
                    "latency_p95_sec": _r(percentile(lats, 95)),
                    "last_ok_at": _r(last_ok, 3) if last_ok else None,
                    "sec_since_last_ok": _r(since_ok) if since_ok is not None else None,
                    "inferences": st["inferences"] if st else 0,
                    "errors": st["errors"] if st else 0,
                    "last_error": (st["last_error"] if st else "") or None,
                    "frames_skipped": st["frames_skipped"] if st else 0,
                    "stale_frames": st["stale"] if st else 0,
                    "no_new_frame": st["no_new_frame"] if st else 0,
                    "no_frame": st["no_frame"] if st else 0,
                }
            )

        model_rows: dict[str, Any] = {}
        for name, (kind, events, calls, images) in sorted(model_copy.items()):
            ev = [e for e in events if e[0] >= cut]
            lat = [e[2] for e in ev]
            waits = [e[3] for e in ev]
            imgs = sum(e[1] for e in ev)
            model_rows[name] = {
                "kind": kind,
                "calls_per_sec": _r(len(ev) / span),
                "images_per_sec": _r(imgs / span),
                "avg_batch": _r(imgs / len(ev), 2) if ev else 0.0,
                "max_batch": max((e[1] for e in ev), default=0),
                "latency_p50_ms": _r(percentile(lat, 50) * 1000, 1),
                "latency_p95_ms": _r(percentile(lat, 95) * 1000, 1),
                "lock_wait_avg_ms": _r((sum(waits) / len(waits)) * 1000 if waits else 0.0, 1),
                "lock_wait_p95_ms": _r(percentile(waits, 95) * 1000, 1),
                "busy_pct": _r(100.0 * sum(lat) / span, 1),
                "total_calls": calls,
                "total_images": images,
            }

        stage_rows: dict[str, Any] = {}
        for name, (dq, full) in sorted(stage_copy.items()):
            ev = [e for e in dq if e[0] >= cut]
            if not ev:
                continue
            # A full deque may hold less than the window: measure over what it holds.
            st_span = span
            if full:
                st_span = max(1e-6, min(span, now - dq[0][0]))
            secs = [e[1] for e in ev]
            stage_rows[name] = {
                "calls_per_sec": _r(len(ev) / st_span, 2),
                "avg_ms": _r(1000 * sum(secs) / len(secs), 2),
                "p50_ms": _r(1000 * percentile(secs, 50), 2),
                "p95_ms": _r(1000 * percentile(secs, 95), 2),
                "max_ms": _r(1000 * max(secs), 2),
                # 100% = one thread fully busy with this step
                "busy_pct": _r(100.0 * sum(secs) / st_span, 1),
            }

        gpu_busy = sum(v["busy_pct"] for v in model_rows.values() if v["kind"] == "gpu")
        out: dict[str, Any] = {
            "at": _r(now),
            "window_sec": self.window_sec,
            "measured_sec": _r(span, 1),
            "sla_sec": sla_sec,
            "target_interval_sec": target_interval_sec,
            "stale_sec": self.stale_sec,
            "summary": {
                "cameras_total": len(keys),
                "cameras_connected": connected,
                "cameras_receiving": receiving,
                "cameras_starved": len(starved),
                "starved_keys": starved,
                "total_infer_fps": _r(total_fps),
                "worst_gap_p95_sec": _r(worst_p95_gap),
                "sla_met": connected > 0 and receiving == connected and worst_p95_gap <= sla_sec,
                # Sum of time inside GPU model calls / wall time. Near (or above) 100% means GPU
                # work is saturated; most of it is serialised by the one global predict lock.
                "gpu_model_busy_pct": _r(gpu_busy, 1),
            },
            "models": model_rows,
            "stages": stage_rows,
            "cameras": cam_rows,
        }
        if include_system and self._system is not None:
            out["system"] = self._system.sample()
        return out


_metrics: LiveMetrics | None = None
_metrics_lock = threading.Lock()


def get_live_metrics() -> LiveMetrics:
    global _metrics
    with _metrics_lock:
        if _metrics is None:
            _metrics = LiveMetrics()
        return _metrics
