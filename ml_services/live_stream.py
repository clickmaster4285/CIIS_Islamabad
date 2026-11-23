"""
Live RTSP streams with purpose-gated multi-model YOLO inference + optional plate OCR.
Only models relevant to each camera's purpose run on that feed.

Shared Camera Session ingest (one decode per camera):
  NVR main stream ──► ONE FFmpeg NVDEC ──► SHARED FRAME BUFFER
         ├─► Partitioned Inference Workers → Result Buffer (AI FPS)
         ├─► Render Thread → Browser MJPEG (live FPS)
         ├─► Journey / attendance / evidence consumers
         └─► Do not open a second FFmpeg for the same camera
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import asyncio
import threading
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Iterator
from urllib.parse import quote

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

_DEFAULT_FFMPEG_CAPTURE_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer+discardcorrupt|"
    "flags;low_delay|"
    "err_detect;ignore_err|"
    "probesize;500000|"
    "analyzeduration;500000|"
    "max_delay;0|"
    "reorder_queue_size;0|"
    "stimeout;5000000"
)
if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = os.getenv(
        "ML_RTSP_FFMPEG_OPTIONS",
        _DEFAULT_FFMPEG_CAPTURE_OPTIONS,
    )

import cv2
import numpy as np

from camera_byte_tracker import CameraByteTrackerPool
from face_recognizer import KnownFaceDB
from face_track_gate import FaceTrackGate, is_face_target
from inference_engine import (
    ALLOWED_COCO_CLASS_IDS,
    SMOKE_FIRE_MIN_CONF,
    WEAPON_MIN_CONF,
    custom_only_class_ids,
    get_cuda_status,
    get_face_db,
    get_yolo_coco_model,
    get_yolo_custom_model,
    get_yolo_smoke_model,
    get_yolo_weapon_model,
    gpu_predict_lock,
    keep_custom_classes_only,
    merge_triple_detections,
    parse_yolo_result,
    resolve_coco_weights_path,
    resolve_custom_weights_path,
    resolve_ml_device,
    resolve_smoke_weights_path,
    resolve_weapon_weights_path,
)
from live_metrics import get_live_metrics
from motion_clips import MotionClipRecorder
from motion_state import MotionRegistry
from pose_estimator import POSE_CONF, POSE_IMGSZ, attach_poses, get_pose_model, is_person, parse_pose_result
from object_identity import face_bbox_key, face_result_from_detection, get_object_identity_registry
from plate_recognizer import PlateEngine, get_plate_engine


def build_rtsp_url(
    ip: str,
    user: str = "admin",
    password: str = "",
    port: str | int = "554",
    path: str = "/Streaming/Channels/101",
) -> str:
    encoded_password = quote(password, safe="")
    if password:
        return f"rtsp://{user}:{encoded_password}@{ip}:{port}{path}"
    return f"rtsp://{user}@{ip}:{port}{path}"


def _rtsp_config() -> dict[str, str]:
    return {
        "user": os.getenv("CAMERA_RTSP_USER", "admin"),
        "password": os.getenv("CAMERA_RTSP_PASSWORD", ""),
        "port": os.getenv("CAMERA_RTSP_PORT", "554"),
        "path": os.getenv("CAMERA_RTSP_PATH", "/Streaming/Channels/101"),
    }


def _boot_camera_ips() -> list[str]:
    raw = os.getenv("ML_LIVE_BOOT_IPS", "").strip()
    if not raw:
        return []
    return [ip.strip() for ip in raw.split(",") if ip.strip()]


_CAM_KEY_RE = re.compile(r"^cam-\d+$", re.IGNORECASE)
_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _key_may_be_rtsp_host(key: str) -> bool:
    """True only when the stream key looks like an IP / hostname, not cam-{id}."""
    text = (key or "").strip()
    if not text or _CAM_KEY_RE.match(text):
        return False
    if _IPV4_RE.match(text):
        return True
    if "." in text and not text.startswith("cam-"):
        return True
    return text in _boot_camera_ips()



def _require_registered() -> bool:
    """When true (default), refuse to open streams for keys not in the Django sync registry."""
    return os.getenv("ML_REQUIRE_REGISTERED", "true").strip().lower() in ("true", "1", "yes")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


_URL_CREDS_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<user>[^:@/\s]*)(?::[^@/\s]*)?@")


def redact_url(text: str | None) -> str:
    """rtsp://user:pass@host/... → rtsp://user:***@host/... (also inside longer messages)."""
    if not text:
        return text or ""
    return _URL_CREDS_RE.sub(lambda m: f"{m.group('scheme')}{m.group('user')}:***@", str(text))


def allowed_stream_schemes() -> frozenset[str]:
    """URL schemes the ML node will open (ML_ALLOWED_STREAM_SCHEMES, default rtsp,rtsps).

    The Django backend only registers rtsp:// URLs. Add http/https here only if some
    camera really is an HTTP MJPEG source — otherwise ffmpeg could be pointed at any
    URL or local file through the register API.
    """
    raw = os.getenv("ML_ALLOWED_STREAM_SCHEMES", "rtsp,rtsps")
    out = {s.strip().lower().rstrip(":/") for s in raw.split(",") if s.strip()}
    return frozenset(out or {"rtsp", "rtsps"})


def stream_url_allowed(url: str | None) -> bool:
    text = (url or "").strip()
    if "://" not in text:
        return False
    scheme = text.split("://", 1)[0].lower()
    return scheme in allowed_stream_schemes()


def _protocol_whitelist() -> str:
    """ffmpeg -protocol_whitelist: RTSP transports only, plus http(s) if that scheme is allowed."""
    protos = ["rtsp", "rtsps", "rtp", "srtp", "tcp", "udp", "tls"]
    schemes = allowed_stream_schemes()
    if "http" in schemes:
        protos += ["http", "httpproxy"]
    if "https" in schemes:
        protos += ["https"]
    return ",".join(protos)


def _resolve_ffmpeg_path() -> str | None:
    custom = os.getenv("FFMPEG_PATH", "").strip()
    if custom and os.path.isfile(custom):
        return custom
    found = shutil.which("ffmpeg")
    if found:
        return found
    base = Path(__file__).resolve().parent.parent / "tools" / "ffmpeg" / "bin"
    for name in ("ffmpeg.exe", "ffmpeg"):
        candidate = base / name
        if candidate.is_file():
            return str(candidate)
    return None


def _rtsp_decode_backend() -> str:
    return os.getenv("ML_RTSP_DECODE", "ffmpeg").strip().lower()


def _cuda_device_index() -> str:
    """GPU index for NVDEC (from ML_DEVICE)."""
    raw = (os.getenv("ML_DEVICE", "0") or "0").strip()
    low = raw.lower()
    if low == "cpu":
        return "0"
    if low.startswith("cuda:"):
        return low.split(":", 1)[1] or "0"
    return raw if raw.isdigit() else "0"


def _nvdec_requested() -> bool:
    """User wants GPU decode always unless explicitly disabled."""
    val = os.getenv("ML_RTSP_NVDEC", "true").strip().lower()
    return val not in ("0", "false", "no", "off")


_ffmpeg_cuda_support: bool | None = None
_ffmpeg_cuda_logged = False


def _ffmpeg_supports_cuda(ffmpeg_path: str) -> bool:
    global _ffmpeg_cuda_support, _ffmpeg_cuda_logged
    if _ffmpeg_cuda_support is not None:
        return _ffmpeg_cuda_support
    try:
        proc = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        blob = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
        _ffmpeg_cuda_support = "cuda" in blob
    except Exception as exc:
        print(f"[live] ffmpeg CUDA probe failed: {exc}")
        _ffmpeg_cuda_support = False
    if not _ffmpeg_cuda_logged:
        _ffmpeg_cuda_logged = True
        if _ffmpeg_cuda_support:
            print(f"[live] NVDEC available via ffmpeg (device={_cuda_device_index()})")
        else:
            print(
                "[live] WARNING: ffmpeg has no CUDA hwaccel — "
                "install CUDA-enabled ffmpeg for GPU RTSP decode"
            )
    return _ffmpeg_cuda_support


def _use_nvdec(ffmpeg_path: str) -> bool:
    return _nvdec_requested() and _ffmpeg_supports_cuda(ffmpeg_path)


def _rtsp_scale_size() -> tuple[int, int]:
    """
    Live AI/view capture size after FFmpeg scale (NVR keeps original 4K recording).

    Defaults to 1280x720 — missing env must NOT open native 4K MJPEG for multi-cam.
    Set both width/height to 0 AND ML_RTSP_ALLOW_NATIVE=true to keep native.
    """
    w = max(0, _env_int("ML_RTSP_SCALE_WIDTH", 1280))
    h = max(0, _env_int("ML_RTSP_SCALE_HEIGHT", 720))
    return w, h


def _force_rtsp_scale() -> bool:
    """Default ON: always scale unless explicitly disabled."""
    return str(os.getenv("ML_RTSP_FORCE_SCALE", "true") or "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _allow_native_rtsp() -> bool:
    """Native 4K live path is opt-in only (dangerous at 24+ cameras)."""
    return str(os.getenv("ML_RTSP_ALLOW_NATIVE", "false") or "false").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _ffmpeg_threads() -> str:
    return os.getenv("ML_FFMPEG_THREADS", "1").strip() or "1"


def _ffmpeg_socket_timeout_us() -> str:
    return os.getenv("ML_FFMPEG_STIMEOUT_US", "10000000").strip() or "10000000"


_ffmpeg_version_cache: dict[str, tuple[int, int]] = {}


def _ffmpeg_version(ffmpeg_path: str | None = None) -> tuple[int, int]:
    """(major, minor) of the ffmpeg binary; (0, 0) when unknown (git builds, probe failure)."""
    path = ffmpeg_path or _resolve_ffmpeg_path() or "ffmpeg"
    if path in _ffmpeg_version_cache:
        return _ffmpeg_version_cache[path]
    ver = (0, 0)
    try:
        proc = subprocess.run([path, "-hide_banner", "-version"], capture_output=True, text=True, timeout=8, check=False)
        m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", proc.stdout or "")
        if m:
            ver = (int(m.group(1)), int(m.group(2)))
    except Exception as exc:
        print(f"[live] ffmpeg version probe failed: {exc}")
    _ffmpeg_version_cache[path] = ver
    return ver


def _ffmpeg_timeout_cli_flag(ffmpeg_path: str | None = None) -> str:
    """CLI flag for the RTSP socket I/O timeout (microseconds).

    ffmpeg >= 5: '-timeout' is the socket I/O timeout ('-stimeout' was removed).
    ffmpeg 4.x : '-timeout' means *listen* timeout and turns on listen mode (the stream never
                 opens) — the socket I/O timeout is '-stimeout' there.
    Unknown version (git build): 'timeout'. ML_FFMPEG_TIMEOUT_FLAG overrides.
    """
    raw = os.getenv("ML_FFMPEG_TIMEOUT_FLAG", "").strip().lstrip("-")
    if raw:
        return raw
    major, _minor = _ffmpeg_version(ffmpeg_path)
    return "stimeout" if 0 < major < 5 else "timeout"


def _ingest_fps() -> float:
    """Max frames/s ffmpeg hands to Python per camera (ML_INGEST_FPS; 0 = camera rate).

    Default: the fastest consumer — motion clip sampling (ML_MOTION_CLIP_FPS, 8) or the
    browser preview (ML_LIVE_STREAM_FPS, 12) — and never below 8. Inference only takes the
    newest frame, so frames above this rate were decoded to JPEG and thrown away.
    """
    raw = os.getenv("ML_INGEST_FPS", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    clip_fps = min(25.0, max(1.0, _env_float("ML_MOTION_CLIP_FPS", 8.0)))
    view_fps = max(5, min(_env_int("ML_LIVE_STREAM_FPS", 12), 30))
    return float(max(8.0, clip_fps, view_fps))


def _ingest_fps_args(ffmpeg_path: str) -> tuple[list[str], str]:
    """(output args, extra -vf filter) that cap the frame rate without duplicating frames."""
    fps = _ingest_fps()
    if fps <= 0:
        return [], ""
    major, minor = _ffmpeg_version(ffmpeg_path)
    if (major, minor) >= (4, 4) or major == 0:
        return ["-fpsmax", f"{fps:g}"], ""
    # Old ffmpeg: drop frames closer than 1/fps apart (software frames; append after hwdownload)
    gap = max(0.0, 0.97 / fps)
    return ["-vsync", "vfr"], f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{gap:.4f})'"


def _mjpeg_quality(*, keep_native: bool) -> str:
    # Higher -q:v = cheaper JPEG. Avoid q:v 2 on live (was crushing 24-cam servers).
    if keep_native:
        return os.getenv("ML_MJPEG_QUALITY_NATIVE", "5").strip() or "5"
    return os.getenv("ML_MJPEG_QUALITY", "7").strip() or "7"


def _rtsp_stream_input_flags(ffmpeg_path: str | None = None) -> list[str]:
    flags = [
        "-protocol_whitelist",
        _protocol_whitelist(),
        "-rtsp_transport",
        "tcp",
        f"-{_ffmpeg_timeout_cli_flag(ffmpeg_path)}",
        _ffmpeg_socket_timeout_us(),
    ]
    reorder = os.getenv("ML_FFMPEG_REORDER_QUEUE_SIZE", "").strip()
    if reorder:
        flags += ["-reorder_queue_size", reorder]
    return flags


def _rtsp_scale_filter(use_cuda_scale: bool) -> str | None:
    """FFmpeg -vf string to downscale before MJPEG pipe / YOLO."""
    w, h = _rtsp_scale_size()
    if w <= 0 and h <= 0:
        # Safety net: never return None when force-scale is on.
        if _force_rtsp_scale() or not _allow_native_rtsp():
            w, h = 1280, 720
        else:
            return None
    if w <= 0:
        w = -2
    if h <= 0:
        h = -2
    # Keep aspect ratio; never upscale small streams.
    if use_cuda_scale:
        # NVDEC frames stay on GPU → scale_cuda → download for MJPEG encode.
        return (
            f"scale_cuda=w={w}:h={h}:force_original_aspect_ratio=decrease,"
            "hwdownload,format=nv12"
        )
    return f"scale={w}:{h}:force_original_aspect_ratio=decrease:force_divisible_by=2"


class CameraStream:
    """Background reader that keeps the newest frame; reconnects on decode failures."""

    def __init__(self, rtsp_url: str, label: str, drain_reads: int = 2):
        self.rtsp_url = rtsp_url
        self.label = label
        self.keep_native = False
        self.drain_reads = max(1, drain_reads)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_seq = 0
        self.frame_ts = 0.0  # wall time the newest frame was stored (metrics: frame age)
        self.running = True
        self.connected = False
        self.cap: cv2.VideoCapture | None = None
        self._fail_streak = 0
        self._max_fail_before_reconnect = max(5, _env_int("ML_RTSP_MAX_FAILS", 30))
        self._reconnect_delay = max(0.5, _env_float("ML_RTSP_RECONNECT_SEC", 2.0))
        self._open_retry_delay = max(0.5, _env_float("ML_RTSP_OPEN_RETRY_SEC", 3.0))
        self._logged_res = False
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{label}")

    @staticmethod
    def _open(source: str):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open RTSP stream: {redact_url(source)}")
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    @staticmethod
    def _valid_frame(frame: np.ndarray | None) -> bool:
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            return False
        if len(frame.shape) < 2:
            return False
        h, w = frame.shape[:2]
        return h >= 32 and w >= 32

    def _release_cap(self) -> None:
        cap = self.cap
        self.cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _open_cap(self) -> bool:
        self._release_cap()
        try:
            self.cap = self._open(self.rtsp_url)
            self._fail_streak = 0
            return True
        except Exception as exc:
            print(f"[live] Failed to open {self.label}: {redact_url(str(exc))}")
            self.connected = False
            return False

    def _read_frame(self) -> np.ndarray | None:
        cap = self.cap
        if cap is None:
            return None
        try:
            for _ in range(self.drain_reads):
                if not cap.grab():
                    return None
            ret, frame = cap.retrieve()
            if ret and self._valid_frame(frame):
                return frame
            ret, frame = cap.read()
            if ret and self._valid_frame(frame):
                return frame
        except cv2.error:
            return None
        except Exception:
            return None
        return None

    def _run(self):
        while self.running:
            if self.cap is None:
                if not self._open_cap():
                    time.sleep(self._open_retry_delay)
                    continue

            frame = self._read_frame()
            if frame is not None:
                if not self._logged_res:
                    h, w = frame.shape[:2]
                    self._logged_res = True
                    print(f"[live] {self.label} native stream {w}x{h} (opencv)")
                with self.lock:
                    self.frame = frame
                    self.frame_seq += 1
                    self.frame_ts = time.time()
                    self.connected = True
                self._fail_streak = 0
                continue

            self._fail_streak += 1
            self.connected = False
            if self._fail_streak >= self._max_fail_before_reconnect:
                print(
                    f"[live] Reconnecting {self.label} after "
                    f"{self._fail_streak} failed frame read(s)"
                )
                self._release_cap()
                self._fail_streak = 0
                time.sleep(self._reconnect_delay)
            else:
                time.sleep(0.02)

    def get_frame(self) -> np.ndarray | None:
        frame, _ = self.get_latest()
        return frame

    def get_latest(self) -> tuple[np.ndarray | None, int]:
        """Copy of newest frame + monotonic seq (Latest Frame Buffer read)."""
        with self.lock:
            if self.frame is None:
                return None, self.frame_seq
            return self.frame.copy(), self.frame_seq

    def get_latest_meta(self) -> tuple[np.ndarray | None, int, float]:
        """get_latest() plus the wall time that frame was stored, read under one lock."""
        with self.lock:
            if self.frame is None:
                return None, self.frame_seq, 0.0
            return self.frame.copy(), self.frame_seq, self.frame_ts

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self._release_cap()


def _stall_timeout_sec() -> float:
    """No new frame for this long → kill ffmpeg so the reconnect path runs (0 = off)."""
    return max(0.0, _env_float("ML_RTSP_STALL_SEC", 15.0))


_stall_streams: "weakref.WeakSet[FfmpegCameraStream]" = weakref.WeakSet()
_stall_lock = threading.Lock()
_stall_thread: threading.Thread | None = None


def _stall_watchdog_loop() -> None:
    while True:
        time.sleep(1.0)
        limit = _stall_timeout_sec()
        if limit <= 0:
            continue
        with _stall_lock:
            streams = list(_stall_streams)
        now = time.time()
        for stream in streams:
            try:
                stream._check_stall(now, limit)
            except Exception as exc:
                print(f"[live] stall watchdog error: {exc}")


def _watch_for_stall(stream: "FfmpegCameraStream") -> None:
    global _stall_thread
    with _stall_lock:
        _stall_streams.add(stream)
        if _stall_thread is None or not _stall_thread.is_alive():
            _stall_thread = threading.Thread(target=_stall_watchdog_loop, daemon=True, name="cam-stall-watchdog")
            _stall_thread.start()


class FfmpegCameraStream:
    """
    RTSP reader via ffmpeg:
      4K camera (NVR still records original)
        → NVDEC decode
        → scale to 1080p inside FFmpeg
        → MJPEG pipe → Latest Frame Buffer → YOLO / Render / Browser
    """

    def __init__(self, rtsp_url: str, label: str, ffmpeg_path: str, keep_native: bool = False):
        self.rtsp_url = rtsp_url
        self.label = label
        self.ffmpeg_path = ffmpeg_path
        self.keep_native = bool(keep_native)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.frame_seq = 0
        self.frame_ts = 0.0  # wall time the newest frame was stored (metrics: frame age)
        self.running = True
        self.connected = False
        self._proc: subprocess.Popen | None = None
        self._fail_streak = 0
        self._max_fail_before_reconnect = max(5, _env_int("ML_RTSP_MAX_FAILS", 30))
        self._reconnect_delay = max(0.5, _env_float("ML_RTSP_RECONNECT_SEC", 2.0))
        self._open_retry_delay = max(0.5, _env_float("ML_RTSP_OPEN_RETRY_SEC", 3.0))
        self._logged_res = False
        self._use_nvdec = _use_nvdec(ffmpeg_path)
        # Prefer GPU scale when NVDEC is on; fall back to CPU scale on ffmpeg errors.
        # Native/4K ANPR path skips FFmpeg scale entirely.
        self._use_cuda_scale = self._use_nvdec and not self.keep_native
        self._last_stderr = ""
        # Stall watchdog: wall time the current ffmpeg started / last produced a frame.
        self._proc_started_at = 0.0
        self.stalls = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{label}")
        _watch_for_stall(self)

    def _check_stall(self, now: float, limit: float) -> None:
        """Kill a running ffmpeg that has produced no frame for `limit` seconds."""
        proc = self._proc
        if not self.running or proc is None or proc.poll() is not None:
            return
        last = max(self._proc_started_at, self.frame_ts)
        if last <= 0 or now - last < limit:
            return
        self.stalls += 1
        print(f"[live] {self.label} no frame for {now - last:.0f}s — restarting ffmpeg (stall #{self.stalls})")
        self.connected = False
        try:
            proc.kill()  # stdout read returns EOF → normal reconnect path
        except Exception:
            pass
        self._proc_started_at = now  # one kill per stall window

    def _ffmpeg_cmd(self) -> list[str]:
        """
        Decode with NVDEC when available.
        Default: scale to 1080p in FFmpeg, then MJPEG pipe.
        ANPR (keep_native): leave the original 4K frame in the buffer.
        """
        cmd: list[str] = [
            self.ffmpeg_path,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            _ffmpeg_threads(),
        ]
        if self._use_nvdec:
            cmd += [
                "-hwaccel",
                "cuda",
                "-hwaccel_device",
                _cuda_device_index(),
            ]
            if (
                not self.keep_native
                and self._use_cuda_scale
                and _rtsp_scale_filter(True)
            ):
                # Keep frames on GPU until after scale_cuda.
                cmd += ["-hwaccel_output_format", "cuda"]

        cmd += [
            "-fflags",
            "nobuffer+discardcorrupt+genpts",
            "-flags",
            "low_delay",
            *_rtsp_stream_input_flags(self.ffmpeg_path),
            "-i",
            self.rtsp_url,
            "-an",
        ]

        vf = None
        if not self.keep_native:
            vf = _rtsp_scale_filter(use_cuda_scale=bool(self._use_nvdec and self._use_cuda_scale))
            if not vf:
                # Last-resort CPU scale — never pipe native 4K MJPEG by accident.
                vf = "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2"
        fps_args, fps_filter = _ingest_fps_args(self.ffmpeg_path)
        if fps_filter:
            vf = f"{vf},{fps_filter}" if vf else fps_filter
        if vf:
            cmd += ["-vf", vf]

        mux_qsize = os.getenv("ML_FFMPEG_MAX_MUXING_QUEUE_SIZE", "1024").strip() or "1024"
        cmd += [
            *fps_args,
            "-max_muxing_queue_size",
            mux_qsize,
            "-f",
            "mjpeg",
            "-q:v",
            _mjpeg_quality(keep_native=self.keep_native),
            "-",
        ]
        return cmd

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Prevent stderr pipe fill; keep last error line for reconnect logs."""
        try:
            if proc.stderr is None:
                return
            chunks: list[str] = []
            while True:
                line = proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    chunks.append(text)
                    if len(chunks) > 8:
                        chunks = chunks[-8:]
            if chunks:
                self._last_stderr = redact_url(" | ".join(chunks))[-240:]
        except Exception:
            pass

    def _maybe_fallback_cpu_scale(self) -> None:
        """If scale_cuda fails, switch to CPU scale on next reconnect."""
        err = (self._last_stderr or "").lower()
        if not self._use_cuda_scale:
            return
        markers = (
            "scale_cuda",
            "impossible to convert",
            "function not found",
            "no device",
            "invalid argument",
            "hwdownload",
        )
        if any(m in err for m in markers):
            self._use_cuda_scale = False
            print(
                f"[live] {self.label} scale_cuda unavailable — "
                "falling back to CPU scale=1080p in FFmpeg"
            )

    def _stop_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _decode_mjpeg_buffer(self, buffer: bytearray) -> np.ndarray | None:
        metrics = get_live_metrics()
        while True:
            t0 = time.perf_counter()
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9")
            if start == -1 or end == -1 or end < start:
                return None
            jpg = bytes(buffer[start : end + 2])
            del buffer[: end + 2]
            arr = np.frombuffer(jpg, dtype=np.uint8)
            t1 = time.perf_counter()
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            metrics.record_stage("acquire.mjpeg_scan", t1 - t0)
            metrics.record_stage("acquire.mjpeg_imdecode", time.perf_counter() - t1)
            if frame is not None and CameraStream._valid_frame(frame):
                return frame
            if not buffer:
                return None

    def _run(self) -> None:
        sw, sh = _rtsp_scale_size()
        if self.keep_native:
            scale_note = "native-4K"
        else:
            if not (sw or sh):
                sw, sh = 1280, 720
            scale_note = f"scale={sw}x{sh}"
        if self._use_nvdec:
            print(
                f"[live] {self.label} opening GPU NVDEC + FFmpeg {scale_note} "
                f"(cuda:{_cuda_device_index()})"
            )
        elif _nvdec_requested():
            print(f"[live] {self.label} NVDEC unavailable — software decode + FFmpeg {scale_note}")
        else:
            print(f"[live] {self.label} opening FFmpeg {scale_note}")

        while self.running:
            self._stop_proc()
            # Re-check CUDA availability; keep CPU-scale fallback once chosen.
            self._use_nvdec = _use_nvdec(self.ffmpeg_path)
            if not self._use_nvdec:
                self._use_cuda_scale = False
            decode_tag = "ffmpeg+nvdec" if self._use_nvdec else "ffmpeg"
            if self.keep_native:
                decode_tag += "+native"
            elif self._use_nvdec and self._use_cuda_scale:
                decode_tag += "+scale_cuda"
            elif sw or sh:
                decode_tag += "+scale"
            try:
                self._last_stderr = ""
                self._proc_started_at = time.time()
                self._proc = subprocess.Popen(
                    self._ffmpeg_cmd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                threading.Thread(
                    target=self._drain_stderr,
                    args=(self._proc,),
                    daemon=True,
                    name=f"fferr-{self.label}",
                ).start()
            except OSError as exc:
                print(f"[live] Failed to open {self.label} ({decode_tag}): {redact_url(str(exc))}")
                self.connected = False
                time.sleep(self._open_retry_delay)
                continue

            if not self._proc.stdout:
                self._stop_proc()
                time.sleep(self._open_retry_delay)
                continue

            buffer = bytearray()
            self._fail_streak = 0
            while self.running and self._proc.poll() is None:
                try:
                    chunk = self._proc.stdout.read(8192)
                except Exception:
                    chunk = b""
                if not chunk:
                    self._fail_streak += 1
                    if self._fail_streak >= self._max_fail_before_reconnect:
                        break
                    time.sleep(0.02)
                    continue

                buffer.extend(chunk)
                while True:
                    frame = self._decode_mjpeg_buffer(buffer)
                    if frame is None:
                        break
                    if not self._logged_res:
                        h, w = frame.shape[:2]
                        self._logged_res = True
                        print(
                            f"[live] {self.label} capture {w}x{h} "
                            f"({decode_tag}; NVR may still record 4K)"
                        )
                    with self.lock:
                        self.frame = frame
                        self.frame_seq += 1
                        self.frame_ts = time.time()
                        self.connected = True
                    self._fail_streak = 0

            self.connected = False
            if self.running:
                err_hint = (self._last_stderr or "").strip()
                self._maybe_fallback_cpu_scale()
                if err_hint:
                    print(f"[live] Reconnecting {self.label} ({decode_tag}): {err_hint}")
                else:
                    print(f"[live] Reconnecting {self.label} ({decode_tag})")
                self._stop_proc()
                time.sleep(self._reconnect_delay)

    def get_frame(self) -> np.ndarray | None:
        frame, _ = self.get_latest()
        return frame

    def get_latest(self) -> tuple[np.ndarray | None, int]:
        """Copy of newest frame + monotonic seq (Latest Frame Buffer read)."""
        with self.lock:
            if self.frame is None:
                return None, self.frame_seq
            return self.frame.copy(), self.frame_seq

    def get_latest_meta(self) -> tuple[np.ndarray | None, int, float]:
        """get_latest() plus the wall time that frame was stored, read under one lock."""
        with self.lock:
            if self.frame is None:
                return None, self.frame_seq, 0.0
            return self.frame.copy(), self.frame_seq, self.frame_ts

    def stop(self) -> None:
        self.running = False
        self._stop_proc()
        self.thread.join(timeout=2.0)


def create_camera_stream(
    rtsp_url: str,
    label: str,
    keep_native: bool = False,
) -> CameraStream | FfmpegCameraStream:
    # GPU NVDEC requires the ffmpeg path — skip OpenCV when NVDEC is requested.
    backend = _rtsp_decode_backend()
    ffmpeg = _resolve_ffmpeg_path()
    if backend == "opencv" and not _nvdec_requested():
        stream = CameraStream(rtsp_url, label)
        stream.keep_native = bool(keep_native)
        return stream
    if ffmpeg:
        return FfmpegCameraStream(rtsp_url, label, ffmpeg, keep_native=keep_native)
    print(f"[live] ffmpeg not found — falling back to OpenCV for {label}")
    stream = CameraStream(rtsp_url, label)
    stream.keep_native = bool(keep_native)
    return stream


def _overlay_name(det: dict[str, Any]) -> str:
    """Human-readable name/class — never replace this with the global ID alone."""
    display_id = str(det.get("display_id") or det.get("global_object_id") or "").strip()
    name = str(det.get("label") or "").strip()
    cls = str(det.get("class_name") or "").strip()
    if not name or (display_id and name.lower() == display_id.lower()) or _OBJECT_ID_LABEL.match(name):
        name = cls or "object"
    return name


def _overlay_text(det: dict[str, Any]) -> str:
    """Format live box text as '{global_id} {actual_label}' when both exist."""
    display_id = str(det.get("display_id") or det.get("global_object_id") or "").strip()
    name = _overlay_name(det)
    if display_id and name and display_id.lower() not in name.lower():
        return f"{display_id} {name}".strip()
    return (display_id or name).strip()


_POSE_DRAW = os.getenv("ML_POSE_DRAW", "true").strip().lower() in ("true", "1", "yes")


def _draw_skeleton(output: np.ndarray, kps: list, thickness: int, min_conf: float = 0.3) -> None:
    draw_skeleton(output, kps, thickness, min_conf)


from motion_path_draw import draw_motion_path, draw_skeleton  # noqa: E402


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]], label_scale: float = 0.55) -> np.ndarray:
    if not detections:
        return frame
    output = frame.copy()
    h, _ = output.shape[:2]
    font_scale = max(0.32, label_scale * (h / 720))
    thickness = max(1, int(font_scale * 1.5))
    box_thickness = max(1, int(font_scale * 1.2))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        name = _overlay_name(det)
        conf = float(det.get("confidence", 0))
        x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
        is_unknown = (
            bool(det.get("is_unknown"))
            or name.lower() == "unknown"
            or name.lower().startswith("unknown")
            or bool(_OBJECT_ID_LABEL.match(name))
        )
        is_alert = bool(det.get("alert"))
        if is_alert:
            color = (0, 0, 255)
        elif is_unknown:
            color = (0, 140, 255)
        else:
            color = (0, 220, 0)
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thickness)
        if _POSE_DRAW and det.get("keypoints"):
            _draw_skeleton(output, det["keypoints"], box_thickness)
        if det.get("motion_path"):
            draw_motion_path(output, det["motion_path"], max(2, box_thickness + 1))
        posture_word = str(det.get("posture") or "").strip()
        label = f"{_overlay_text(det)} {conf:.2f}{' ' + posture_word if posture_word else ''}".strip()
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = int(x1)
        text_y = max(text_h + 4, int(y1) - 4)
        cv2.rectangle(
            output,
            (text_x, text_y - text_h - 3),
            (text_x + text_w + 4, text_y + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(output, label, (text_x + 2, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return output


_GENERIC_FACE_LABELS = frozenset({"", "unknown", "person", "face"})
_OBJECT_ID_LABEL = re.compile(r"^(?:gp|go|gv|t)\d+$", re.IGNORECASE)


def _is_generic_face_label(label: str) -> bool:
    value = (label or "").strip().lower()
    if not value or value in _GENERIC_FACE_LABELS:
        return True
    if value.startswith("unknown") or value.startswith("face:unknown"):
        return True
    return bool(_OBJECT_ID_LABEL.match(value))


def assign_overlay_ids(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put a stable ID on every live box: GP/GO/GV from identity, else ByteTrack T#.

    Keeps the real label/class on the detection — never overwrite it with ID-only text.
    """
    for det in detections or []:
        gid = str(det.get("global_object_id") or "").strip()
        tid = det.get("track_id")
        try:
            track_no = int(tid) if tid is not None else None
        except (TypeError, ValueError):
            track_no = None
        display_id = gid or (f"T{track_no}" if track_no is not None else "")
        if display_id:
            det["display_id"] = display_id

        cls = str(det.get("class_name") or "").strip()
        label = str(det.get("label") or "").strip()
        if cls.lower() in ("person", "face") and _is_generic_face_label(label):
            det["is_unknown"] = True
            # Keep a readable class label; global ID stays in display_id only
            det["label"] = cls or "person"
    return detections


def filter_enrolled_staff_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only show boxes/labels for recognized enrolled staff (attendance-eligible identities)."""
    kept: list[dict[str, Any]] = []
    for det in detections or []:
        if det.get("alert"):
            kept.append(det)
            continue
        # Always keep license-plate OCR overlays
        if str(det.get("model") or "").strip().lower() == "plate":
            kept.append(det)
            continue
        cls = str(det.get("class_name") or det.get("label") or "").strip().lower()
        if cls in ("license_plate", "license plate", "number_plate", "number plate"):
            kept.append(det)
            continue
        if cls not in ("person", "face"):
            continue
        label = str(det.get("label") or "").strip()
        if not label or _is_generic_face_label(label):
            continue
        kept.append(det)
    return kept


# COCO class ids for ANPR vehicle association (car / motorcycle / bus / truck)
_ANPR_VEHICLE_CLASS_IDS: frozenset[int] = frozenset({2, 3, 5, 7})
_ANPR_VEHICLE_NAMES: frozenset[str] = frozenset({"car", "motorcycle", "bus", "truck", "vehicle"})
_FACE_PURPOSE_CLASS_IDS: frozenset[int] = frozenset({0})  # person
# General objects = COCO allowlist WITHOUT vehicles (vehicles only when ANPR is selected)
_GENERAL_OBJECT_CLASS_IDS: frozenset[int] = frozenset(
    cid for cid in ALLOWED_COCO_CLASS_IDS if cid not in _ANPR_VEHICLE_CLASS_IDS
)

# Legacy purpose codes → model-centric codes (must match Django PURPOSE_ALIASES)
_PURPOSE_ALIASES: dict[str, str] = {
    "object_detection": "general_objects",
    "surveillance": "general_objects",
    "zone_monitoring": "general_objects",
    "thermal": "smoke_fire",
}

_SMOKE_FIRE_NAMES: frozenset[str] = frozenset({"smoke", "fire", "flame", "burning"})
_WEAPON_NAMES: frozenset[str] = frozenset(
    {
        "weapon",
        "gun",
        "pistol",
        "rifle",
        "firearm",
        "knife",
        "knife_weapon",
        "sword",
        "machete",
        "heavy-weapon",
    }
)
_PLATE_NAMES: frozenset[str] = frozenset(
    {
        "license_plate",
        "license plate",
        "number_plate",
        "number plate",
        "plate",
    }
)


def filter_detections_for_purposes(
    detections: list[dict[str, Any]],
    purposes: list[str] | set[str],
) -> list[dict[str, Any]]:
    """Hard gate: keep only detections that belong to the selected model purposes."""
    purpose_set = {
        _PURPOSE_ALIASES.get(str(p).strip().lower(), str(p).strip().lower())
        for p in (purposes or [])
        if str(p).strip()
    }
    if not purpose_set:
        return []

    kept: list[dict[str, Any]] = []
    for det in detections or []:
        tag = str(det.get("model_tag") or det.get("model") or "").strip().lower()
        cls = str(det.get("class_name") or "").strip().lower()
        label = str(det.get("label") or "").strip().lower()
        allow = False

        if "general_objects" in purpose_set:
            if tag in ("", "coco") and cls and cls not in _ANPR_VEHICLE_NAMES and cls not in _PLATE_NAMES:
                if cls not in _SMOKE_FIRE_NAMES and tag not in ("smoke", "weapon", "custom", "plate"):
                    allow = True

        if "custom_objects" in purpose_set and tag == "custom":
            allow = True

        if "smoke_fire" in purpose_set and (tag == "smoke" or cls in _SMOKE_FIRE_NAMES or label in _SMOKE_FIRE_NAMES):
            allow = True

        if "weapon" in purpose_set and (tag == "weapon" or cls in _WEAPON_NAMES or label in _WEAPON_NAMES):
            allow = True

        if purpose_set & {"face_recognition", "attendance"}:
            if cls in ("person", "face") or label in ("person", "face") or (
                tag in ("", "coco") and cls == "person"
            ):
                allow = True

        if "anpr" in purpose_set:
            if tag == "plate" or cls in _PLATE_NAMES or cls in _ANPR_VEHICLE_NAMES:
                allow = True

        if allow:
            kept.append(det)
    return kept


def filter_osd_detections(
    detections: list[dict[str, Any]],
    frame_h: int,
    frame_w: int,
    *,
    top_frac: float = 0.22,
    left_frac: float = 0.25,
    right_frac: float = 0.25,
    bottom_frac: float = 0.0,
) -> list[dict[str, Any]]:
    """Drop boxes in CCTV OSD bands (top strip corners / optional bottom).

    left_frac / right_frac: width of the left / right corner zones (fraction of the frame
    width) where the small-text guard below the strip applies. They are clamped to 0.45 so
    the two corners can never cover the whole width (the old 0.55 / 0.55 defaults did, and
    removed every small box just under the strip, not only corner text).
    """
    if not detections or frame_h <= 0 or frame_w <= 0:
        return detections
    y_top = frame_h * max(0.0, min(top_frac, 0.5))
    y_bottom = frame_h * (1.0 - max(0.0, min(bottom_frac, 0.5)))
    x_left = frame_w * max(0.0, min(left_frac, 0.45))
    x_right = frame_w * (1.0 - max(0.0, min(right_frac, 0.45)))
    kept: list[dict[str, Any]] = []
    small_h = frame_h * 0.06  # date / time / camera-name text is short
    for det in detections:
        x1, y1, x2, y2 = (float(v) for v in det.get("bbox", [0, 0, 0, 0])[:4])
        cx = (x1 + x2) / 2.0
        # Box ENTIRELY inside the top OSD strip (date/time text). A real object that only
        # reaches into the strip (a car on the road beyond a gate) is kept — the old centre
        # test deleted 56% of the vehicles on the gate camera.
        if y2 <= y_top:
            continue
        # Bottom OSD strip if configured: box entirely inside it
        if bottom_frac > 0 and y1 >= y_bottom:
            continue
        # Corner guard just below the strip: only small, text-sized boxes
        if y2 <= y_top * 1.35 and (y2 - y1) <= small_h and (cx <= x_left or cx >= x_right):
            continue
        kept.append(det)
    return kept


def _mjpeg_idle_timeout() -> float:
    """MJPEG stream ends when no new frame arrived for this long (ML_MJPEG_IDLE_TIMEOUT_SEC)."""
    return max(2.0, _env_float("ML_MJPEG_IDLE_TIMEOUT_SEC", 30.0))


def _mjpeg_part(jpeg: bytes) -> bytes:
    return (
        b"--frame\r\nContent-Type: image/jpeg\r\n"
        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
        + jpeg
        + b"\r\n"
    )


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes | None:
    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes() if ret else None


class _StageTimer:
    """Average time per inference step (for /live/timing) — finds what limits the infer rate."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._totals: dict[str, list[float]] = {}
        self._since = time.time()

    def add(self, stage: str, seconds: float) -> None:
        with self._lock:
            row = self._totals.setdefault(stage, [0, 0.0])
            row[0] += 1
            row[1] += seconds

    def snapshot(self, reset: bool = False) -> dict[str, Any]:
        with self._lock:
            elapsed = max(1e-6, time.time() - self._since)
            out = {
                stage: {"calls": int(n), "avg_ms": round(1000 * t / n, 1) if n else 0.0, "busy_pct": round(100 * t / elapsed, 1)}
                for stage, (n, t) in sorted(self._totals.items())
            }
            result = {"window_sec": round(elapsed, 1), "stages": out}
            if reset:
                self._totals = {}
                self._since = time.time()
            return result


class _CameraSession:
    """Per-camera buffers: latest frame mirror + result/JPEG caches."""

    def __init__(self, ip: str, stream: CameraStream, rtsp_url: str):
        self.ip = ip
        self.stream = stream
        self.rtsp_url = rtsp_url
        self.lock = threading.Lock()
        # Result Buffer + browser JPEG (written by render / infer)
        self.latest_jpeg: bytes | None = None
        self.latest_detections: list[dict[str, Any]] = []
        # Infer scheduling (newest-frame-only; never queue stale work)
        self.infer_lock = threading.Lock()
        self.infer_busy = False
        self.infer_seq_done = 0
        self.last_infer_at = 0.0
        # Render-on-view: when a browser last asked for this camera's preview / MJPEG,
        # and when the render path last encoded it (monotonic seconds).
        self.last_viewed_at = 0.0
        self.rendered_at = 0.0
        # Set when the camera is unregistered / reopened: late inference results are dropped.
        self.closed = False

    def set_frame(self, jpeg: bytes | None, detections: list[dict[str, Any]]):
        with self.lock:
            if jpeg is not None:
                self.latest_jpeg = jpeg
                self.rendered_at = time.monotonic()
            self.latest_detections = detections

    def set_results(self, detections: list[dict[str, Any]]):
        with self.lock:
            self.latest_detections = detections


class LiveStreamManager:
    """Runs triple-model inference and serves annotated MJPEG per camera key."""

    def __init__(self):
        self._sessions: dict[str, _CameraSession] = {}
        self._registry: dict[str, str] = {}
        self._purposes: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._byte_trackers = CameraByteTrackerPool(track_buffer=90)
        self._object_identity = get_object_identity_registry()
        self._motion = MotionRegistry()
        self._timing = _StageTimer()
        self._metrics = get_live_metrics()
        self._clips = MotionClipRecorder(
            ffmpeg_path=_resolve_ffmpeg_path,
            frame_sources=lambda: {key: s.stream for key, s in list(self._sessions.items())},
            pose_batch=self._clip_pose_batch,
            vehicle_batch=self._clip_vehicle_batch,
        )
        self._infer_executor: ThreadPoolExecutor | None = None
        self._infer_futures: list[Future] = []
        self._render_thread: threading.Thread | None = None
        self._running = False
        self._face_db: KnownFaceDB | None = None
        self._plate_engine: PlateEngine | None = None
        self._coco_model = None
        self._custom_model = None
        self._smoke_model = None
        self._weapon_model = None
        self._pose_model = None
        self._device: str | int = 0
        self._conf = 0.12
        self._iou = 0.45
        self._imgsz = 1280
        self._max_det = 300
        self._osd_filter = True
        self._osd_top = 0.22
        self._osd_left = 0.25
        self._osd_right = 0.25
        self._osd_bottom = 0.0
        self._infer_interval = 0.15
        # Partition cameras across workers (e.g. 5 workers → ~5 cams each when 25 cams).
        self._infer_workers = max(1, min(_env_int("ML_LIVE_INFER_WORKERS", 5), 16))
        self._offline_pause = False
        # Browser preview defaults: smaller/faster JPEGs (override via env).
        self._jpeg_quality = 50
        self._face_threshold = 0.28
        # Plate OCR: on for purpose=anpr by default; set ML_PLATE_ON_ALL=true for every camera.
        self._plate_on_all = os.getenv("ML_PLATE_ON_ALL", "false").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self._plate_ocr_every = max(1, _env_int("ML_PLATE_OCR_EVERY", 3))
        self._plate_frame_counters: dict[str, int] = {}
        self._plate_counter_lock = threading.Lock()
        self._last_plate_dets: dict[str, list[dict[str, Any]]] = {}
        # Alarm-type models (custom / smoke-fire / weapon) run every Nth frame per camera;
        # object detection + pose (what movement needs) run on every frame. Between runs the
        # last alarm-model boxes are reused so overlays and alerts do not flicker.
        self._specialist_every = max(1, _env_int("ML_SPECIALIST_EVERY_N", 3))
        # Stage 2.1: reuse face pass 1 (coco parse) in the identity step instead of running
        # YuNet + SFace again on the same box inside the shared registry lock. Off = old path.
        self._face_dedupe = os.getenv("ML_FACE_DEDUPE", "false").strip().lower() in ("1", "true", "yes")
        # Stage 2.3: face recognition after ByteTrack, once per track id (see face_track_gate.py).
        self._face_track_cache = os.getenv("ML_FACE_TRACK_CACHE", "false").strip().lower() in ("1", "true", "yes")
        self._face_gate = FaceTrackGate()
        # Stage 3: "batched" = one GPU dispatcher batches YOLO across the most-overdue cameras,
        # CPU post-processing runs in the worker pool. "legacy" = contiguous shard workers.
        self._scheduler_mode = os.getenv("ML_LIVE_SCHEDULER", "legacy").strip().lower()
        self._batch_max = max(1, min(_env_int("ML_LIVE_BATCH_MAX", 4), 16))
        self._prefetch = threading.local()
        self._dispatch_thread: threading.Thread | None = None
        self._inflight = 0
        self._inflight_lock = threading.Lock()
        # Stage 2.2: render thread encodes preview JPEGs only for cameras a viewer asked for
        # in the last ML_LIVE_RENDER_VIEWER_GRACE seconds (default ON; a single /jpeg request
        # renders on demand). ML_LIVE_RENDER_ONLY_WHEN_WATCHED=false = encode every camera every tick.
        self._render_only_watched = os.getenv("ML_LIVE_RENDER_ONLY_WHEN_WATCHED", "true").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self._render_grace = max(0.5, _env_float("ML_LIVE_RENDER_VIEWER_GRACE", 5.0))
        self._osd_enrolled_staff_only = False  # real value set in _load_infer_settings
        self._specialist_count: dict[str, int] = {}
        self._last_specialist: dict[str, tuple[list, list, list]] = {}
        # 0 = native 4K passthrough for browser preview (override via ML_LIVE_MAX_WIDTH/HEIGHT).
        self._max_width = 0
        self._max_height = 0
        self._stream_fps = max(5, min(_env_int("ML_LIVE_STREAM_FPS", 12), 30))
        self._frame_interval = 1.0 / self._stream_fps
        self._detections: dict[str, list[dict[str, Any]]] = {}
        # Same-frame evidence (raw infer frame) keyed with detections — for Django snapshots.
        # The frame is kept and JPEG-encoded only when /jpeg/evidence asks (it used to be
        # encoded after every inference of every camera, mostly never read).
        self._evidence_frame: dict[str, np.ndarray] = {}
        self._evidence_jpeg: dict[str, bytes] = {}
        self._evidence_wh: dict[str, tuple[int, int]] = {}
        self._det_lock = threading.Lock()
        self._start_lock = threading.Lock()

    def configure_from_env(self):
        boot_ips = _boot_camera_ips()
        for ip in boot_ips:
            threading.Thread(
                target=self.ensure_camera,
                args=(ip,),
                daemon=True,
                name=f"live-boot-{ip}",
            ).start()

    def _load_infer_settings(self):
        try:
            self._conf = float(os.getenv("ML_YOLO_CONF", str(self._conf)))
        except (TypeError, ValueError):
            pass
        try:
            self._iou = float(os.getenv("ML_YOLO_IOU", str(self._iou)))
        except (TypeError, ValueError):
            pass
        try:
            self._imgsz = max(320, int(os.getenv("ML_YOLO_IMGSZ", str(self._imgsz))))
        except (TypeError, ValueError):
            pass
        try:
            self._max_det = max(1, int(os.getenv("ML_YOLO_MAX_DET", str(self._max_det))))
        except (TypeError, ValueError):
            pass
        self._osd_filter = os.getenv("ML_OSD_FILTER", "true").lower() in ("true", "1", "yes")
        self._osd_enrolled_staff_only = os.getenv("ML_OSD_ENROLLED_STAFF_ONLY", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        try:
            self._osd_top = float(os.getenv("ML_OSD_TOP", str(self._osd_top)))
        except (TypeError, ValueError):
            pass
        try:
            self._osd_left = float(os.getenv("ML_OSD_LEFT", str(self._osd_left)))
        except (TypeError, ValueError):
            pass
        try:
            self._osd_right = float(os.getenv("ML_OSD_RIGHT", str(self._osd_right)))
        except (TypeError, ValueError):
            pass
        try:
            self._osd_bottom = float(os.getenv("ML_OSD_BOTTOM", str(self._osd_bottom)))
        except (TypeError, ValueError):
            pass
        self._max_width = _env_int("ML_LIVE_MAX_WIDTH", self._max_width)
        self._max_height = _env_int("ML_LIVE_MAX_HEIGHT", self._max_height)
        self._jpeg_quality = max(40, min(100, _env_int("ML_LIVE_JPEG_QUALITY", self._jpeg_quality)))
        self._stream_fps = max(5, min(_env_int("ML_LIVE_STREAM_FPS", self._stream_fps), 30))
        self._frame_interval = 1.0 / self._stream_fps
        try:
            self._infer_interval = max(0.05, float(os.getenv("ML_LIVE_INFER_INTERVAL", str(self._infer_interval))))
        except (TypeError, ValueError):
            pass
        self._infer_workers = max(1, min(_env_int("ML_LIVE_INFER_WORKERS", self._infer_workers), 16))

    def _detach_session_locked(self, key: str) -> _CameraSession | None:
        """Pop a camera's live state. Caller holds self._lock; nothing slow happens here.

        The session is marked closed so an inference still running on it does not publish
        results (or motion / clips) after the camera is gone. Call _release_detached()
        after releasing self._lock to stop the ingest.
        """
        session = self._sessions.pop(key, None)
        with self._det_lock:
            self._detections.pop(key, None)
            self._evidence_frame.pop(key, None)
            self._evidence_jpeg.pop(key, None)
            self._evidence_wh.pop(key, None)
        with self._plate_counter_lock:
            self._plate_frame_counters.pop(key, None)
        self._last_plate_dets.pop(key, None)
        self._specialist_count.pop(key, None)
        self._last_specialist.pop(key, None)
        if session is not None:
            session.closed = True
            with session.infer_lock:
                session.infer_busy = False
        return session

    def _release_detached(self, key: str, session: _CameraSession | None) -> None:
        """Slow part of closing a camera (outside self._lock): trackers, ids, ingest stop."""
        self._metrics.forget_camera(key)
        # New stream / unregistered: track ids restart, so old binds must not survive.
        self._byte_trackers.reset(key)
        try:
            self._object_identity.forget_camera(key)
        except Exception as exc:
            print(f"[live] identity reset failed [{key}]: {exc}")
        if self._plate_engine is not None:
            try:
                self._plate_engine.clear_camera_tracks(key)
            except Exception:
                pass
        # Release shared ingest (one FFmpeg per camera) — do not stop stream twice.
        try:
            from camera_session import get_camera_session_manager

            get_camera_session_manager().release(key, stream=session.stream if session is not None else None)
        except Exception:
            if session is not None:
                try:
                    session.stream.stop()
                except Exception:
                    pass

    def _reopen(self, key: str, old: _CameraSession, url: str) -> None:
        """Stop the old ingest (outside the lock), then bind a fresh one."""
        self._release_detached(key, old)
        with self._lock:
            if key not in self._sessions and self._registry.get(key):
                self._open_session_locked(key, url)

    def is_registered(self, key: str) -> bool:
        return bool(self._registry.get((key or "").strip()))

    def resolve_rtsp_url(self, key: str, rtsp_url: str | None = None) -> str | None:
        """Resolve RTSP for a key. Registered cameras only when ML_REQUIRE_REGISTERED=true."""
        key = (key or "").strip()
        if not key:
            return None
        registered = self._registry.get(key, "").strip()
        if registered:
            # Ignore client-supplied rtsp_url — assignment registry is authoritative.
            return registered
        if _require_registered():
            return None
        explicit = (rtsp_url or "").strip()
        if explicit:
            return explicit
        if not _key_may_be_rtsp_host(key):
            return None
        cfg = _rtsp_config()
        return build_rtsp_url(key, cfg["user"], cfg["password"], cfg["port"], cfg["path"])

    def register_camera(
        self,
        key: str,
        rtsp_url: str,
        purpose: str = "",
        purposes: list[str] | None = None,
    ) -> bool:
        key = (key or "").strip()
        url = (rtsp_url or "").strip()
        if not key or not url:
            return False
        if not stream_url_allowed(url):
            print(f"[live] Rejected {key}: stream URL scheme not allowed ({redact_url(url)[:60]})")
            return False
        purpose_list = self._normalize_purpose_list(purposes, primary=purpose)
        old = None
        with self._lock:
            self._registry[key] = url
            self._purposes[key] = purpose_list
            existing = self._sessions.get(key)
            if existing is not None:
                native_mismatch = bool(getattr(existing.stream, "keep_native", False)) != self._want_native_frame(key)
                if existing.rtsp_url != url or native_mismatch:
                    old = self._detach_session_locked(key)
        if old is not None:
            self._reopen(key, old, url)
        return True

    def set_camera_purposes(
        self,
        key: str,
        *,
        purpose: str = "",
        purposes: list[str] | None = None,
    ) -> list[str]:
        """Update AI purposes for an already-known camera key (e.g. from MJPEG query)."""
        key = (key or "").strip()
        if not key:
            return []
        purpose_list = self._normalize_purpose_list(purposes, primary=purpose)
        old = None
        with self._lock:
            if purpose_list:
                self._purposes[key] = purpose_list
            existing = self._sessions.get(key)
            if existing is not None:
                native_mismatch = bool(getattr(existing.stream, "keep_native", False)) != self._want_native_frame(key)
                if native_mismatch:
                    old = self._detach_session_locked(key)
            applied = list(self._purposes.get(key) or [])
        if old is not None:
            self._reopen(key, old, old.rtsp_url)
        return applied

    @staticmethod
    def _normalize_purpose_list(
        purposes: list[str] | None,
        *,
        primary: str = "",
    ) -> list[str]:
        out: list[str] = []
        for raw in purposes or []:
            code = _PURPOSE_ALIASES.get(str(raw or "").strip().lower(), str(raw or "").strip().lower())
            if code and code not in out:
                out.append(code)
        primary_code = _PURPOSE_ALIASES.get(
            (primary or "").strip().lower(),
            (primary or "").strip().lower(),
        )
        if primary_code and primary_code not in out:
            out.insert(0, primary_code)
        return out

    def register_cameras_bulk(
        self,
        entries: list[dict[str, str]],
        *,
        replace: bool = False,
    ) -> dict[str, int]:
        registered = 0
        keep: set[str] = set()
        for item in entries:
            key = str(item.get("key") or "").strip()
            url = str(item.get("rtsp_url") or "").strip()
            purpose = str(item.get("purpose") or "").strip()
            raw_purposes = item.get("purposes")
            purposes = raw_purposes if isinstance(raw_purposes, list) else []
            if not key or not url:
                continue
            keep.add(key)
            if self.register_camera(key, url, purpose=purpose, purposes=purposes):
                registered += 1
        removed = 0
        if replace:
            for key in list(self._registry.keys()):
                if key not in keep:
                    if self.unregister_camera(key):
                        removed += 1
        self.ensure_started()
        return {"registered": registered, "total": len(entries), "removed": removed}

    def get_raw_frame(self, key: str):
        """Latest decoded frame from an existing live RTSP session (no extra connection)."""
        key = (key or "").strip()
        if not key:
            return None
        with self._lock:
            session = self._sessions.get(key)
        if session is None:
            return None
        return session.stream.get_frame()

    def get_frame_meta(self, key: str) -> tuple[int, float] | None:
        """(decoder frame_seq, wall time of that frame) of a live session, else None."""
        with self._lock:
            session = self._sessions.get((key or "").strip())
        if session is None:
            return None
        stream = session.stream
        lock = getattr(stream, "lock", None)
        if lock is not None:
            with lock:
                return int(getattr(stream, "frame_seq", 0) or 0), float(getattr(stream, "frame_ts", 0.0) or 0.0)
        return int(getattr(stream, "frame_seq", 0) or 0), float(getattr(stream, "frame_ts", 0.0) or 0.0)

    def is_ready(self) -> bool:
        """True after YOLO infer loops have started. Raw RTSP can run before this."""
        return bool(self._running)

    def get_raw_jpeg_bytes(self, key: str, *, target_width: int | None = None) -> bytes | None:
        """Encode the latest raw RTSP frame (does not wait for YOLO)."""
        raw = self.get_raw_frame(key)
        if raw is None:
            return None
        if target_width:
            prepared = self._prepare_attendance_frame(raw, int(target_width))
            quality = 98 if int(target_width) >= 2560 else max(90, min(self._jpeg_quality, 98))
            return encode_jpeg(prepared, quality)
        quality = max(70, min(self._jpeg_quality, 98))
        return encode_jpeg(self._limit_size(raw), quality)

    def wait_for_raw_jpeg(
        self,
        key: str,
        *,
        timeout_sec: float = 6.0,
        target_width: int | None = None,
    ) -> bytes | None:
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        while time.monotonic() < deadline:
            jpeg = self.get_raw_jpeg_bytes(key, target_width=target_width)
            if jpeg:
                return jpeg
            time.sleep(0.05)
        return None

    def wait_for_preview_jpeg(self, key: str, *, timeout_sec: float = 2.5) -> bytes | None:
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        while time.monotonic() < deadline:
            frame = self.get_preview_jpeg(key)
            if frame:
                return frame
            jpeg = self.get_raw_jpeg_bytes(key)
            if jpeg:
                return jpeg
            time.sleep(0.05)
        return None

    def ensure_started(self) -> bool:
        """Start infer/render threads if YOLO is available but loops are not running."""
        self.start()
        return self._running

    def unregister_camera(self, key: str) -> bool:
        key = (key or "").strip()
        if not key:
            return False
        with self._lock:
            self._registry.pop(key, None)
            self._purposes.pop(key, None)
            old = self._detach_session_locked(key)
        self._release_detached(key, old)
        self._motion.reset(key)
        self._clips.reset(key)
        self._face_gate.reset(key)
        return True

    def ensure_camera(self, key: str, rtsp_url: str | None = None) -> bool:
        key = (key or "").strip()
        if not key:
            return False
        if _require_registered() and not self.is_registered(key):
            print(
                f"[live] ensure_camera rejected {key}: not registered on this ML node "
                f"(assign via Django Camera Distribution / sync)"
            )
            return False
        url = self.resolve_rtsp_url(key, rtsp_url)
        if not url:
            return False
        if not stream_url_allowed(url):
            print(f"[live] ensure_camera rejected {key}: stream URL scheme not allowed")
            return False
        with self._lock:
            existing = self._sessions.get(key)
            if existing is None:
                self._open_session_locked(key, url)
                return True
            # Never reopen just because a query-string RTSP URL differs slightly
            # (encoding / param order). That caused Opening storms + perpetual 503.
            if url and existing.rtsp_url != url:
                self._registry[key] = url
            native_mismatch = bool(getattr(existing.stream, "keep_native", False)) != self._want_native_frame(key)
            if not native_mismatch:
                return True
            old = self._detach_session_locked(key)
        self._reopen(key, old, url)
        return True

    def get_preview_jpeg(self, key: str) -> bytes | None:
        """Cached browser JPEG. With render-on-view, a stale cache is rendered once here."""
        session = self._sessions.get(key)
        if session is None:
            return None
        session.last_viewed_at = time.monotonic()
        if self._render_only_watched:
            with session.lock:
                stale = (time.monotonic() - session.rendered_at) > max(1.0, 3 * self._frame_interval)
            if stale:
                self._render_session(session)
        with session.lock:
            return session.latest_jpeg

    def start(self):
        with self._start_lock:
            if self._running:
                return
            try:
                self._load_infer_settings()
                coco_weights = resolve_coco_weights_path()
                custom_weights = resolve_custom_weights_path()
                smoke_weights = resolve_smoke_weights_path()
                weapon_weights = resolve_weapon_weights_path()
                if not any([coco_weights, custom_weights, smoke_weights, weapon_weights]):
                    print("[live] No YOLO weights — live annotated streams disabled.")
                    return

                self._coco_model = get_yolo_coco_model()
                self._custom_model = get_yolo_custom_model()
                self._smoke_model = get_yolo_smoke_model()
                self._weapon_model = get_yolo_weapon_model()
                self._pose_model = get_pose_model()
                if (
                    self._coco_model is None
                    and self._custom_model is None
                    and self._smoke_model is None
                    and self._weapon_model is None
                ):
                    print("[live] YOLO models unavailable.")
                    return

                self._device = resolve_ml_device()
                cuda = get_cuda_status()
                self._face_db = get_face_db()
                if hasattr(self._face_db, "threshold"):
                    self._face_db.threshold = self._face_threshold
                try:
                    self._plate_engine = get_plate_engine()
                except Exception as exc:
                    print(f"[live] Plate engine unavailable: {exc}")
                    self._plate_engine = None
                plate_ok = bool(self._plate_engine and self._plate_engine.available)
                self._running = True
                self._infer_futures = []
                self._infer_executor = ThreadPoolExecutor(
                    max_workers=self._infer_workers,
                    thread_name_prefix="live-infer",
                )
                if self._scheduler_mode == "batched":
                    self._dispatch_thread = threading.Thread(
                        target=self._batched_dispatch_loop, daemon=True, name="live-gpu-dispatch"
                    )
                    self._dispatch_thread.start()
                else:
                    for i in range(self._infer_workers):
                        fut = self._infer_executor.submit(self._infer_shard_loop, i)
                        self._infer_futures.append(fut)
                self._render_thread = threading.Thread(target=self._render_loop, daemon=True, name="live-render")
                self._render_thread.start()
                self._clips.start()
                self._metrics.start_gil_probe()
                max_label = (
                    "native"
                    if self._max_width <= 0 and self._max_height <= 0
                    else f"{self._max_width}x{self._max_height}"
                )
                print(
                    f"[live] scheduler={self._scheduler_mode}"
                    + (f" batch_max={self._batch_max}" if self._scheduler_mode == "batched" else "")
                )
                print(
                    "[live] Started partitioned infer workers "
                    f"(device={self._device}, gpu={cuda.get('cuda_device_name') or 'n/a'}, "
                    f"fps={self._stream_fps}, infer_interval={self._infer_interval}s, "
                    f"infer_workers={self._infer_workers} "
                    f"[Worker-i → contiguous camera shard], "
                    f"conf={self._conf}, imgsz={self._imgsz}, display={max_label}, "
                    f"osd_enrolled_staff_only={self._osd_enrolled_staff_only}, "
                    f"rtsp_decode={_rtsp_decode_backend()}, "
                    f"coco={coco_weights or 'off'}, custom={custom_weights or 'off'}, "
                    f"smoke={smoke_weights or 'off'}, weapon={weapon_weights or 'off'}, "
                    f"plate={'on' if plate_ok else 'off'}, plate_on_all={self._plate_on_all})"
                )
            except Exception as exc:
                print(f"[live] start failed: {exc}")
                self._running = False

    def stop(self):
        self._running = False
        self._clips.stop()
        if self._infer_executor is not None:
            try:
                self._infer_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                self._infer_executor.shutdown(wait=False)
            self._infer_executor = None
        for fut in self._infer_futures:
            try:
                fut.result(timeout=2.0)
            except Exception:
                pass
        self._infer_futures = []
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=2.0)
            self._dispatch_thread = None
        if self._render_thread:
            self._render_thread.join(timeout=2.0)
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.closed = True
            try:
                session.stream.stop()
            except Exception:
                pass

    def status(self) -> dict[str, Any]:
        with self._lock:
            keys = set(self._registry.keys()) | set(self._sessions.keys())
            cameras = []
            for key in sorted(keys):
                session = self._sessions.get(key)
                det_count = 0
                has_frame = False
                connected = False
                if session is not None:
                    with session.lock:
                        det_count = len(session.latest_detections)
                        # infer_seq_done > 0: frames arrive even when no one watches the preview
                        has_frame = session.latest_jpeg is not None or session.infer_seq_done > 0
                    connected = bool(session.stream.connected)
                cameras.append(
                    {
                        "ip": key,
                        "key": key,
                        "registered": key in self._registry,
                        "connected": connected,
                        "has_frame": has_frame,
                        "detections": det_count,
                        "purpose": (self._purposes.get(key) or [""])[0] if self._purposes.get(key) else "",
                        "purposes": list(self._purposes.get(key) or []),
                        "rtsp_url": redact_url((self._registry.get(key) or "").strip()),
                    }
                )
        return {
            "running": self._running,
            "inference_device": self._device,
            "plate_only_mode": False,
            "plate_model_loaded": bool(self._plate_engine and self._plate_engine.available),
            "triple_model_mode": (
                self._coco_model is not None
                and self._custom_model is not None
                and self._smoke_model is not None
            ),
            "quad_model_mode": (
                self._coco_model is not None
                and self._custom_model is not None
                and self._smoke_model is not None
                and self._weapon_model is not None
            ),
            "coco_model_loaded": self._coco_model is not None,
            "custom_model_loaded": self._custom_model is not None,
            "smoke_model_loaded": self._smoke_model is not None,
            "weapon_model_loaded": self._weapon_model is not None,
            "pose_model_loaded": self._pose_model is not None,
            "dual_model_mode": self._coco_model is not None and self._smoke_model is not None,
            "general_model_loaded": self._coco_model is not None,
            "infer_workers": self._infer_workers,
            "camera_count": len(cameras),
            "cameras": cameras,
        }

    def get_latest_jpeg(self, ip: str) -> bytes | None:
        session = self._sessions.get(ip)
        if not session:
            return None
        session.last_viewed_at = time.monotonic()
        with session.lock:
            return session.latest_jpeg

    def get_detections(self, ip: str) -> list[dict[str, Any]]:
        return list(self.get_detection_snapshot(ip).get("detections") or [])

    def get_detection_snapshot(self, ip: str) -> dict[str, Any]:
        session = self._sessions.get(ip)
        if not session:
            return {
                "detections": [],
                "frame_width": 0,
                "frame_height": 0,
                "display_width": 0,
                "display_height": 0,
                "has_evidence": False,
            }
        with self._det_lock:
            detections = list(self._detections.get(ip, []))
            evidence_wh = self._evidence_wh.get(ip)
            has_evidence = ip in self._evidence_frame or bool(self._evidence_jpeg.get(ip))
        # Prefer dimensions of the frame the detections were computed on
        if evidence_wh and evidence_wh[0] > 0 and evidence_wh[1] > 0:
            infer_w, infer_h = int(evidence_wh[0]), int(evidence_wh[1])
            limited_h, limited_w = infer_h, infer_w
            if self._max_width > 0 or self._max_height > 0:
                # display size matches _limit_size of that evidence frame
                max_w = self._max_width if self._max_width > 0 else infer_w
                max_h = self._max_height if self._max_height > 0 else infer_h
                if infer_w > max_w or infer_h > max_h:
                    scale = min(max_w / infer_w, max_h / infer_h)
                    limited_w = int(infer_w * scale)
                    limited_h = int(infer_h * scale)
            display_w, display_h = limited_w, limited_h
        else:
            raw = session.stream.get_frame()
            if raw is not None:
                infer_h, infer_w = raw.shape[:2]
                limited = self._limit_size(raw)
                display_h, display_w = limited.shape[:2]
            else:
                infer_w, infer_h = 0, 0
                display_w, display_h = 0, 0
        return {
            "detections": detections,
            "frame_width": int(infer_w),
            "frame_height": int(infer_h),
            "display_width": int(display_w),
            "display_height": int(display_h),
            "has_evidence": has_evidence,
        }

    def get_evidence_jpeg(self, ip: str) -> bytes | None:
        """Raw infer-frame JPEG stored with the current detection buffer (same frame as YOLO).

        Encoded on first request per inferred frame, then cached until the next inference.
        """
        with self._det_lock:
            data = self._evidence_jpeg.get(ip)
            if data:
                return data
            frame = self._evidence_frame.get(ip)
        if frame is None:
            return None
        # Slightly higher quality than live preview — used for detection evidence.
        q = max(75, min(92, int(self._jpeg_quality) + 10))
        try:
            data = encode_jpeg(frame, q)
        except Exception:
            return None
        if data:
            with self._det_lock:
                if self._evidence_frame.get(ip) is frame:
                    self._evidence_jpeg[ip] = data
        return data or None

    def _session_alive(self, key: str, session: _CameraSession | None) -> bool:
        return session is not None and not session.closed and self._sessions.get(key) is session

    def _mjpeg_next(self, key: str, session: _CameraSession, mode: str, target_width: int, last: Any):
        """(jpeg | None, marker) for one MJPEG tick. Blocking work (encode) happens here."""
        if mode == "live":
            frame = self.get_latest_jpeg(key)
            if frame and frame is not last:
                return frame, frame
            return None, last
        getter = getattr(session.stream, "get_latest", None)
        if callable(getter):
            raw, seq = getter()
        else:
            raw, seq = session.stream.get_frame(), object()
        if raw is None or seq == last:
            return None, last
        if mode == "attendance":
            quality = 98 if target_width >= 2560 else max(90, min(self._jpeg_quality, 98))
            jpeg = encode_jpeg(self._prepare_attendance_frame(raw, target_width), quality)
        else:
            jpeg = encode_jpeg(self._limit_size(raw), max(70, min(self._jpeg_quality, 98)))
        return jpeg, seq

    def _iter_mjpeg_sync(self, key: str, mode: str, target_width: int = 3840) -> Iterator[bytes]:
        session = self._sessions.get(key)
        idle = _mjpeg_idle_timeout()
        last: Any = None
        last_new = time.monotonic()
        while self._session_alive(key, session):
            jpeg, last = self._mjpeg_next(key, session, mode, target_width, last)
            if jpeg:
                last_new = time.monotonic()
                yield _mjpeg_part(jpeg)
                time.sleep(self._frame_interval)
            else:
                if time.monotonic() - last_new > idle:
                    return
                time.sleep(0.02)

    async def aiter_mjpeg(
        self,
        key: str,
        *,
        mode: str = "live",
        target_width: int = 3840,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[bytes]:
        """MJPEG parts without holding a worker thread per viewer.

        Ends when the camera session is closed / replaced, when no new frame arrived for
        ML_MJPEG_IDLE_TIMEOUT_SEC, or when the client disconnected. JPEG encoding of raw /
        attendance frames runs in a worker thread.
        """
        session = self._sessions.get(key)
        idle = _mjpeg_idle_timeout()
        last: Any = None
        last_new = time.monotonic()
        next_disc_check = 0.0
        while self._session_alive(key, session):
            now = time.monotonic()
            if is_disconnected is not None and now >= next_disc_check:
                next_disc_check = now + 1.0
                try:
                    if await is_disconnected():
                        return
                except Exception:
                    return
            if mode == "live":
                jpeg, last = self._mjpeg_next(key, session, mode, target_width, last)
            else:
                jpeg, last = await asyncio.to_thread(self._mjpeg_next, key, session, mode, target_width, last)
            if jpeg:
                last_new = time.monotonic()
                yield _mjpeg_part(jpeg)
                await asyncio.sleep(self._frame_interval)
            else:
                if time.monotonic() - last_new > idle:
                    return
                await asyncio.sleep(0.02)

    def iter_mjpeg(self, key: str) -> Iterator[bytes]:
        return self._iter_mjpeg_sync(key, "live")

    def iter_mjpeg_raw(self, key: str) -> Iterator[bytes]:
        return self._iter_mjpeg_sync(key, "raw")

    def _prepare_frame(self, frame: np.ndarray):
        """
        Other YOLO models stay on the live AI size (~1080p).
        Native/4K ANPR buffers are scaled here; boxes map back via sx/sy.
        """
        h, w = frame.shape[:2]
        max_w, max_h = _rtsp_scale_size()
        if max_w <= 0 and max_h <= 0:
            return frame, 1.0, 1.0
        if max_w <= 0:
            max_w = w
        if max_h <= 0:
            max_h = h
        if w <= max_w and h <= max_h:
            return frame, 1.0, 1.0
        scale = min(max_w / float(w), max_h / float(h))
        dw = max(2, int(w * scale) // 2 * 2)
        dh = max(2, int(h * scale) // 2 * 2)
        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        return small, w / float(dw), h / float(dh)

    def _want_native_frame(self, camera_key: str) -> bool:
        """
        Native 4K live is opt-in only.

        Default: always scale to ML_RTSP_SCALE_* (1280x720) — including ANPR —
        so 24+ camera MJPEG does not flood CPU/RAM. NVR recording stays 4K.
        """
        if _force_rtsp_scale():
            return False
        if not _allow_native_rtsp():
            return False
        w, h = _rtsp_scale_size()
        if w > 0 or h > 0:
            return False
        if self._plate_on_all:
            return True
        return "anpr" in self._purposes_for(camera_key)

    def _open_session_locked(self, key: str, url: str) -> None:
        """Open via CameraSessionManager so every feature shares one decode."""
        keep_native = self._want_native_frame(key)
        from camera_session import get_camera_session_manager

        ingest = get_camera_session_manager().ensure(key, url, keep_native=keep_native)
        if ingest is None:
            print(f"[live] Failed to open shared session: {key}")
            return
        self._sessions[key] = _CameraSession(key, ingest.stream, url)
        with self._det_lock:
            self._detections[key] = []
            self._evidence_frame.pop(key, None)
            self._evidence_jpeg.pop(key, None)
            self._evidence_wh.pop(key, None)
        tag = "native-4K" if keep_native else "scaled"
        print(f"[live] Bound shared session: {key} ({tag})")

    def _predict(
        self,
        model,
        frame: np.ndarray,
        *,
        min_conf: float | None = None,
        classes=None,
        imgsz: int | None = None,
        tag: str = "yolo",
    ):
        pre = getattr(self._prefetch, "results", None)
        if pre and tag in pre:
            # Batched scheduler already ran this model on this frame (same kwargs).
            return [pre.pop(tag)]
        use_half = self._device != "cpu"
        conf = max(self._conf, min_conf) if min_conf is not None else self._conf
        kwargs = {
            "device": self._device,
            "conf": conf,
            "iou": self._iou,
            "imgsz": imgsz or self._imgsz,
            "max_det": self._max_det,
            "half": use_half,
            "verbose": False,
        }
        if classes is not None:
            kwargs["classes"] = list(classes)
        t0 = time.perf_counter()
        with gpu_predict_lock():
            t1 = time.perf_counter()
            result = model.predict(frame, **kwargs)
        t2 = time.perf_counter()
        self._timing.add("gpu_lock_wait", t1 - t0)
        self._timing.add("yolo_predict", t2 - t1)
        self._metrics.record_model(tag, images=1, latency=t2 - t1, lock_wait=t1 - t0)
        try:
            sp = getattr(result[0], "speed", None) or {}
            pre = float(sp.get("preprocess") or 0.0) / 1000.0
            inf = float(sp.get("inference") or 0.0) / 1000.0
            post = float(sp.get("postprocess") or 0.0) / 1000.0
            self._metrics.record_stage(f"yolo.{tag}.1_preprocess", pre)
            self._metrics.record_stage(f"yolo.{tag}.2_inference", inf)
            self._metrics.record_stage(f"yolo.{tag}.3_postprocess", post)
            self._metrics.record_stage(f"yolo.{tag}.4_other_overhead", max(0.0, (t2 - t1) - pre - inf - post))
        except Exception:
            pass
        return result

    def _clip_pose_batch(self, images: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        """YOLO pose on a few clip frames (one short GPU turn). [] per image when pose is off."""
        if self._pose_model is None or not images:
            return [[] for _ in images]
        t0 = time.perf_counter()
        with gpu_predict_lock():
            t1 = time.perf_counter()
            results = self._pose_model.predict(
                images,
                device=self._device,
                conf=POSE_CONF,
                imgsz=POSE_IMGSZ,
                half=self._device != "cpu",
                verbose=False,
            )
        self._timing.add("gpu_lock_wait", t1 - t0)
        self._timing.add("clip_pose", time.perf_counter() - t1)
        self._metrics.record_model(
            "clip_pose", images=len(images), latency=time.perf_counter() - t1, lock_wait=t1 - t0
        )
        return [parse_pose_result(r) for r in results]

    def _clip_vehicle_batch(self, images: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        """YOLO vehicle boxes (car / motorcycle / bus / truck) on a few clip frames."""
        if self._coco_model is None or not images:
            return [[] for _ in images]
        t0 = time.perf_counter()
        with gpu_predict_lock():
            t1 = time.perf_counter()
            results = self._coco_model.predict(
                images,
                device=self._device,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                classes=sorted(_ANPR_VEHICLE_CLASS_IDS),
                half=self._device != "cpu",
                verbose=False,
            )
        self._timing.add("gpu_lock_wait", t1 - t0)
        self._timing.add("clip_vehicles", time.perf_counter() - t1)
        self._metrics.record_model(
            "clip_vehicle", images=len(images), latency=time.perf_counter() - t1, lock_wait=t1 - t0
        )
        out: list[list[dict[str, Any]]] = []
        for img, r in zip(images, results):
            dets = []
            boxes = getattr(r, "boxes", None)
            if boxes is not None and len(boxes):
                for b, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
                    dets.append({"bbox": [float(v) for v in b[:4]], "score": float(c)})
            if self._osd_filter:
                h, w = img.shape[:2]
                dets = filter_osd_detections(dets, h, w, top_frac=self._osd_top, left_frac=self._osd_left,
                                             right_frac=self._osd_right, bottom_frac=self._osd_bottom)
            out.append(dets)
        return out

    def _purposes_for(self, camera_key: str) -> list[str]:
        values = self._purposes.get(camera_key) or []
        if isinstance(values, str):
            values = [values] if values else []
        out: list[str] = []
        for raw in values:
            code = str(raw or "").strip().lower()
            code = _PURPOSE_ALIASES.get(code, code)
            if code and code not in out:
                out.append(code)
        return out

    def _should_run_plates(self, camera_key: str) -> bool:
        if self._plate_engine is None or not self._plate_engine.available:
            return False
        if self._plate_on_all:
            return True
        return "anpr" in self._purposes_for(camera_key)

    def _purpose_pipeline(self, camera_key: str) -> dict[str, Any]:
        """
        Run ONLY the models that are explicitly selected on this camera.
        Each purpose maps 1:1 to a model (union when multiple are checked).
        """
        purposes = set(self._purposes_for(camera_key))
        # No purposes registered yet → run nothing (do NOT fall back to general objects)
        if not purposes:
            return {
                "coco": False,
                "coco_classes": [],
                "recognize_faces": False,
                "custom": False,
                "smoke": False,
                "weapon": False,
                "plates": False,
                "purposes": [],
            }

        coco = False
        recognize_faces = False
        custom = False
        smoke = False
        weapon = False
        coco_ids: set[int] = set()

        # General Objects (YOLO) — non-vehicle COCO classes only
        if "general_objects" in purposes:
            coco = True
            coco_ids |= set(_GENERAL_OBJECT_CLASS_IDS)

        # Custom trained objects model
        if "custom_objects" in purposes:
            custom = True

        # Fire & Smoke specialist — nothing else
        if "smoke_fire" in purposes:
            smoke = True

        # Weapon specialist — nothing else
        if "weapon" in purposes:
            weapon = True

        # Face / Attendance — person + face recognition only
        if purposes & {"face_recognition", "attendance"}:
            coco = True
            recognize_faces = True
            coco_ids |= set(_FACE_PURPOSE_CLASS_IDS)

        # ANPR — vehicles + plate OCR only
        if "anpr" in purposes:
            coco = True
            coco_ids |= set(_ANPR_VEHICLE_CLASS_IDS)

        return {
            "coco": coco,
            "coco_classes": sorted(coco_ids) if coco_ids else ([] if coco else None),
            "recognize_faces": recognize_faces,
            "custom": custom,
            "smoke": smoke,
            "weapon": weapon,
            "plates": self._should_run_plates(camera_key),
            "purposes": sorted(purposes),
        }

    def _infer_frame(
        self,
        frame: np.ndarray,
        *,
        camera_key: str = "",
        run_plates: bool = False,
    ) -> list[dict[str, Any]]:
        t_prep = time.perf_counter()
        infer_frame, sx, sy = self._prepare_frame(frame)
        self._metrics.record_stage("pre.prepare_frame_resize", time.perf_counter() - t_prep)
        pipeline = self._purpose_pipeline(camera_key)
        # Caller may force plates off; purpose may force on
        do_plates = bool(run_plates and pipeline.get("plates"))

        coco_detections: list[dict[str, Any]] = []
        face_results: dict[tuple[int, int, int, int], tuple[str, float, list[float]]] = {}
        custom_detections: list[dict[str, Any]] = []
        smoke_detections: list[dict[str, Any]] = []
        weapon_detections: list[dict[str, Any]] = []

        mark = time.perf_counter()

        def _step(name: str) -> None:
            nonlocal mark
            now_pc = time.perf_counter()
            self._timing.add(name, now_pc - mark)
            self._metrics.record_stage(f"step.{name}", now_pc - mark)
            mark = now_pc

        # Track-gated faces: parse boxes only here; faces are labelled after ByteTrack.
        gate_faces = bool(
            self._face_track_cache and pipeline.get("recognize_faces") and self._face_db is not None
        )
        if pipeline.get("coco") and self._coco_model is not None:
            coco_classes = pipeline.get("coco_classes")
            if coco_classes is None:
                class_filter = ALLOWED_COCO_CLASS_IDS
            else:
                class_filter = frozenset(int(c) for c in coco_classes)
            if class_filter:
                results = self._predict(self._coco_model, infer_frame, classes=class_filter, tag="coco")
                t_parse = time.perf_counter()
                coco_detections = parse_yolo_result(
                    frame,
                    results[0],
                    sx=sx,
                    sy=sy,
                    recognize_faces=bool(pipeline.get("recognize_faces")),
                    smoke_model=False,
                    model_tag="coco",
                    face_db=self._face_db if pipeline.get("recognize_faces") and not gate_faces else None,
                )
                self._metrics.record_stage(
                    "post.coco_parse_incl_face", time.perf_counter() - t_parse
                )
                if self._face_dedupe and not gate_faces and pipeline.get("recognize_faces") and self._face_db is not None:
                    # parse_yolo_result ran label_detection_detail on every coco person/face box.
                    for d in coco_detections:
                        cname = str(d.get("class_name") or "").lower()
                        if int(d.get("class_id", -1)) in (0, 80) or cname in ("person", "face"):
                            bkey = face_bbox_key(d.get("bbox") or [])
                            if bkey is not None:
                                face_results[bkey] = face_result_from_detection(d)
            _step("coco_and_faces")

            # Pose (17 body points) only when this frame has people; other objects have no pose
            if self._pose_model is not None and any(is_person(d) for d in coco_detections):
                try:
                    pose_results = self._predict(
                        self._pose_model, infer_frame, min_conf=POSE_CONF, imgsz=POSE_IMGSZ, tag="pose"
                    )
                    attach_poses(coco_detections, parse_pose_result(pose_results[0], sx=sx, sy=sy))
                except Exception as exc:
                    print(f"[pose] Pose error [{camera_key}]: {exc}")
                _step("pose")

        n = self._specialist_count.get(camera_key, 0)
        self._specialist_count[camera_key] = n + 1
        run_specialists = n % self._specialist_every == 0
        if not run_specialists and camera_key in self._last_specialist:
            # Reused boxes from the last specialist run: marked stale so they can keep an
            # overlay / running danger clip alive but never start a new alert or clip.
            last_custom, last_smoke, last_weapon = self._last_specialist[camera_key]
            custom_detections = [{**d, "stale": True} for d in last_custom]
            smoke_detections = [{**d, "stale": True} for d in last_smoke]
            weapon_detections = [{**d, "stale": True} for d in last_weapon]
        else:
            run_specialists = True

        if run_specialists and pipeline.get("custom") and self._custom_model is not None:
            custom_ids = custom_only_class_ids(self._custom_model)
            results = self._predict(
                self._custom_model,
                infer_frame,
                classes=custom_ids or None,
                tag="custom",
            )
            custom_detections = keep_custom_classes_only(
                parse_yolo_result(
                    frame,
                    results[0],
                    sx=sx,
                    sy=sy,
                    recognize_faces=False,
                    smoke_model=False,
                    model_tag="custom",
                    face_db=None,
                )
            )
            _step("custom_model")

        if run_specialists and pipeline.get("smoke") and self._smoke_model is not None:
            results = self._predict(self._smoke_model, infer_frame, min_conf=SMOKE_FIRE_MIN_CONF, tag="smoke")
            smoke_detections = parse_yolo_result(
                frame,
                results[0],
                sx=sx,
                sy=sy,
                recognize_faces=False,
                smoke_model=True,
                model_tag="smoke",
                face_db=None,
            )
            _step("smoke_model")

        if run_specialists and pipeline.get("weapon") and self._weapon_model is not None:
            results = self._predict(self._weapon_model, infer_frame, min_conf=WEAPON_MIN_CONF, tag="weapon")
            weapon_detections = parse_yolo_result(
                frame,
                results[0],
                sx=sx,
                sy=sy,
                recognize_faces=False,
                weapon_model=True,
                model_tag="weapon",
                face_db=None,
            )
            _step("weapon_model")

        if run_specialists:
            self._last_specialist[camera_key] = (
                [dict(d) for d in custom_detections],
                [dict(d) for d in smoke_detections],
                [dict(d) for d in weapon_detections],
            )

        detections = merge_triple_detections(coco_detections, custom_detections, smoke_detections)
        detections.extend(weapon_detections)

        vehicle_boxes: list[list[float]] = []
        for det in coco_detections:
            cls = str(det.get("class_name") or det.get("label") or "").strip().lower()
            if cls in _ANPR_VEHICLE_NAMES:
                bbox = det.get("bbox") or []
                if len(bbox) >= 4:
                    vehicle_boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])

        if do_plates and self._plate_engine is not None:
            with self._plate_counter_lock:
                counter = self._plate_frame_counters.get(camera_key, 0) + 1
                self._plate_frame_counters[camera_key] = counter
            # Plate YOLO every N infer cycles; PaddleOCR PP-OCRv5 runs inside PlateEngine.
            if counter % self._plate_ocr_every == 0:
                t_plate = time.perf_counter()
                try:
                    plate_dets = self._plate_engine.detect_and_read(
                        frame,
                        camera_key=camera_key,
                        save=True,
                        vehicle_boxes=vehicle_boxes,
                    )
                    self._last_plate_dets[camera_key] = plate_dets
                    detections.extend(plate_dets)
                except Exception as exc:
                    print(f"[live] Plate OCR error [{camera_key}]: {exc}")
                    detections.extend(self._last_plate_dets.get(camera_key, []))
                self._metrics.record_model(
                    "plate_total", images=1, latency=time.perf_counter() - t_plate, kind="mixed"
                )
            else:
                # Keep last plate boxes on screen between detect cycles
                detections.extend(self._last_plate_dets.get(camera_key, []))
            _step("plates")

        if self._osd_filter:
            h, w = frame.shape[:2]
            detections = filter_osd_detections(
                detections,
                h,
                w,
                top_frac=self._osd_top,
                left_frac=self._osd_left,
                right_frac=self._osd_right,
                bottom_frac=self._osd_bottom,
            )
        # Final hard gate — never draw/save classes outside selected purposes
        detections = filter_detections_for_purposes(
            detections,
            pipeline.get("purposes") or self._purposes_for(camera_key),
        )
        _step("filters")
        detections = self._byte_trackers.assign_track_ids(camera_key, detections, frame)
        _step("bytetrack")
        if gate_faces:
            self._face_gate.label(camera_key, frame, detections, self._face_db)
            # Identity step reuses these results instead of running face recognition again.
            for d in detections:
                if is_face_target(d):
                    bkey = face_bbox_key(d.get("bbox") or [])
                    if bkey is not None:
                        face_results[bkey] = face_result_from_detection(d)
            _step("faces_track_gated")
        detections = self._object_identity.enrich_detections(
            camera_key,
            frame,
            detections,
            face_db=self._face_db,
            face_results=face_results or None,
        )
        _step("identity_reid")
        detections = assign_overlay_ids(detections)
        return detections

    def _publish_results(
        self,
        camera_key: str,
        detections: list[dict[str, Any]],
        frame: np.ndarray | None = None,
    ) -> None:
        """Write Result Buffer (API snapshot + same-frame evidence frame, encoded on request)."""
        has_frame = frame is not None and bool(getattr(frame, "size", 0))
        with self._det_lock:
            self._detections[camera_key] = detections
            if has_frame:
                h, w = frame.shape[:2]
                self._evidence_frame[camera_key] = frame
                self._evidence_jpeg.pop(camera_key, None)
                self._evidence_wh[camera_key] = (int(w), int(h))
        session = self._sessions.get(camera_key)
        if session is not None:
            session.set_results(detections)

    def timing(self, reset: bool = False) -> dict[str, Any]:
        return self._timing.snapshot(reset=reset)

    def sla_sec(self) -> float:
        """Max allowed time between inferences for a healthy camera (ML_LIVE_SLA_SEC)."""
        try:
            return max(0.1, float(os.getenv("ML_LIVE_SLA_SEC", "") or max(1.0, 2.0 * self._infer_interval)))
        except (TypeError, ValueError):
            return max(1.0, 2.0 * self._infer_interval)

    def metrics(self, reset: bool = False, include_cameras: bool = True) -> dict[str, Any]:
        """Per-camera / per-model / system inference metrics (observation only)."""
        with self._lock:
            sessions = dict(self._sessions)
            purposes = {k: self._purposes_for(k) for k in sessions}
        meta = {
            key: {"connected": bool(getattr(s.stream, "connected", False)), "purposes": purposes.get(key, [])}
            for key, s in sessions.items()
        }
        snap = self._metrics.snapshot(
            cameras=meta,
            sla_sec=self.sla_sec(),
            target_interval_sec=self._infer_interval,
        )
        snap["config"] = {
            "running": self._running,
            "device": str(self._device),
            "infer_workers": self._infer_workers,
            "infer_interval_sec": self._infer_interval,
            "specialist_every_n": self._specialist_every,
            "plate_ocr_every": self._plate_ocr_every,
            "scheduler": self._scheduler_mode,
            "batch_max": self._batch_max,
            "imgsz": self._imgsz,
            "stream_fps": self._stream_fps,
            "offline_pause": self._offline_pause,
        }
        if not include_cameras:
            snap.pop("cameras", None)
        if reset:
            self._metrics.reset()
        return snap

    def motion_status(self) -> dict[str, Any]:
        return {**self._motion.status(), "clips": self._clips.status()}

    @property
    def motion_clips(self) -> MotionClipRecorder:
        return self._clips

    def _sessions_for_worker(self, worker_id: int) -> list[_CameraSession]:
        """
        Contiguous camera shards across workers:
          Worker-0 → cams[0 : chunk)
          Worker-1 → cams[chunk : 2*chunk)
          ...
        With 25 cams and 5 workers → ~5 cams each (cam1-5, cam6-10, ...).
        """
        n_workers = max(1, self._infer_workers)
        worker_id = max(0, min(int(worker_id), n_workers - 1))
        with self._lock:
            sessions = sorted(self._sessions.values(), key=lambda s: s.ip)
        total = len(sessions)
        if total == 0:
            return []
        base = total // n_workers
        rem = total % n_workers
        # First `rem` workers get one extra camera so every cam is covered.
        start = 0
        for i in range(worker_id):
            start += base + (1 if i < rem else 0)
        size = base + (1 if worker_id < rem else 0)
        return sessions[start : start + size]

    def pause_for_offline(self) -> None:
        """Stop live YOLO while an offline video job owns the GPU."""
        self._offline_pause = True

    def resume_after_offline(self) -> None:
        self._offline_pause = False

    def _run_camera_inference(self, session: _CameraSession) -> None:
        """Infer newest frame only for one camera; write Result Buffer. Never encodes JPEG."""
        if self._offline_pause:
            return
        now = time.time()
        with session.infer_lock:
            if session.infer_busy:
                return
            if (now - session.last_infer_at) < self._infer_interval:
                return

        frame_ts = 0.0
        meta_getter = getattr(session.stream, "get_latest_meta", None)
        getter = getattr(session.stream, "get_latest", None)
        if callable(meta_getter):
            t_get = time.perf_counter()
            frame, seq, frame_ts = meta_getter()
            self._metrics.record_stage("acquire.latest_frame_copy", time.perf_counter() - t_get)
        elif callable(getter):
            frame, seq = getter()
        else:
            frame = session.stream.get_frame()
            seq = session.infer_seq_done + 1 if frame is not None else session.infer_seq_done
        if frame is None:
            self._metrics.record_camera_skip(session.ip, "no_frame")
            return

        prev_seq = 0
        with session.infer_lock:
            if session.infer_busy:
                return
            if (now - session.last_infer_at) < self._infer_interval:
                return
            stalled = seq <= session.infer_seq_done
            if not stalled:
                session.infer_busy = True
                session.last_infer_at = now
                prev_seq = session.infer_seq_done
        if stalled:
            # Due for inference but the reader has no newer frame (stream frozen / slow).
            self._metrics.record_camera_skip(session.ip, "no_new_frame")
            return

        self._infer_and_publish(session, frame, seq, frame_ts, prev_seq, now)

    def _infer_and_publish(
        self,
        session: _CameraSession,
        frame: np.ndarray,
        seq: int,
        frame_ts: float,
        prev_seq: int,
        now: float,
        started_at: float | None = None,
    ) -> None:
        """Inference + Result Buffer for one claimed frame. Clears session.infer_busy."""
        ok = False
        err = ""
        started_wall = started_at if started_at is not None else time.time()
        try:
            t_infer = time.perf_counter()
            detections = self._infer_frame(
                frame,
                camera_key=session.ip,
                run_plates=self._should_run_plates(session.ip),
            )
            self._timing.add("infer_frame_total", time.perf_counter() - t_infer)
            fh, fw = frame.shape[:2]
            for det in detections:
                det["frame_width"] = int(fw)
                det["frame_height"] = int(fh)
            if session.closed or self._sessions.get(session.ip) is not session:
                # Camera unregistered / reopened while this frame was being inferred.
                self._metrics.record_stage("event.dropped_closed_session", 0.0)
                ok = True
                return
            t_m = time.perf_counter()
            motion_events = self._motion.update(session.ip, detections, now)
            t_p = time.perf_counter()
            self._publish_results(session.ip, detections, frame=frame)
            t_c = time.perf_counter()
            self._clips.submit(session.ip, detections, motion_events, now)
            t_e = time.perf_counter()
            self._metrics.record_stage("event.motion_update", t_p - t_m)
            self._metrics.record_stage("event.publish_incl_evidence_jpeg", t_c - t_p)
            self._metrics.record_stage("event.clip_submit", t_e - t_c)
            with session.infer_lock:
                session.infer_seq_done = seq
            ok = True
        except Exception as exc:
            err = str(exc)
            print(f"[live] Inference error [{session.ip}]: {exc}")
        finally:
            with session.infer_lock:
                session.infer_busy = False
            try:
                self._metrics.record_camera_inference(
                    session.ip,
                    started_at=started_wall,
                    finished_at=time.time(),
                    frame_ts=frame_ts,
                    seq=seq,
                    prev_seq=prev_seq,
                    ok=ok,
                    error=err,
                )
            except Exception:
                pass

    # -- Stage 3: batched scheduler (ML_LIVE_SCHEDULER=batched) -----------------------
    def _claim_due_sessions(self, now: float, limit: int) -> list[dict[str, Any]]:
        """Most-overdue cameras first (fair across all cameras, no fixed shards)."""
        with self._lock:
            sessions = list(self._sessions.values())
        sessions.sort(key=lambda s: s.last_infer_at)
        claimed: list[dict[str, Any]] = []
        for session in sessions:
            if len(claimed) >= limit:
                break
            with session.infer_lock:
                if session.infer_busy or (now - session.last_infer_at) < self._infer_interval:
                    continue
                if now < getattr(session, "next_poll_at", 0.0):
                    continue
            frame_ts = 0.0
            meta_getter = getattr(session.stream, "get_latest_meta", None)
            getter = getattr(session.stream, "get_latest", None)
            if callable(meta_getter):
                frame, seq, frame_ts = meta_getter()
            elif callable(getter):
                frame, seq = getter()
            else:
                frame = session.stream.get_frame()
                seq = session.infer_seq_done + 1 if frame is not None else session.infer_seq_done
            if frame is None or seq <= session.infer_seq_done:
                # Same skip accounting as the legacy loop, at most once per interval per camera.
                session.next_poll_at = now + self._infer_interval
                self._metrics.record_camera_skip(session.ip, "no_frame" if frame is None else "no_new_frame")
                continue
            with session.infer_lock:
                if session.infer_busy:
                    continue
                session.infer_busy = True
                session.last_infer_at = now
                prev_seq = session.infer_seq_done
            claimed.append(
                {"session": session, "frame": frame, "seq": seq, "frame_ts": frame_ts, "prev_seq": prev_seq}
            )
        return claimed

    def _predict_batch(
        self,
        model,
        frames: list[np.ndarray],
        *,
        min_conf: float | None = None,
        classes=None,
        imgsz: int | None = None,
        tag: str = "yolo",
    ) -> list:
        """One YOLO call on several frames (same kwargs as _predict)."""
        conf = max(self._conf, min_conf) if min_conf is not None else self._conf
        kwargs = {
            "device": self._device,
            "conf": conf,
            "iou": self._iou,
            "imgsz": imgsz or self._imgsz,
            "max_det": self._max_det,
            "half": self._device != "cpu",
            "verbose": False,
            "batch": len(frames),
        }
        if classes is not None:
            kwargs["classes"] = sorted(int(c) for c in classes)
        t0 = time.perf_counter()
        with gpu_predict_lock():
            t1 = time.perf_counter()
            results = model.predict(frames, **kwargs)
        t2 = time.perf_counter()
        self._timing.add("gpu_lock_wait", t1 - t0)
        self._timing.add("yolo_predict", t2 - t1)
        self._metrics.record_model(tag, images=len(frames), latency=t2 - t1, lock_wait=t1 - t0)
        return list(results)

    @staticmethod
    def _keep_classes(result, class_ids: frozenset[int]):
        """Drop boxes outside one camera's class filter (the batch ran with the union)."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or not len(boxes):
            return result
        cls = boxes.cls.int().tolist()
        idx = [i for i, c in enumerate(cls) if c in class_ids]
        if len(idx) == len(cls):
            return result
        return result[idx]

    def _gpu_prefetch(self, claimed: list[dict[str, Any]]) -> None:
        """Run each needed model once over all claimed frames; per-item results go to item['pre']."""
        groups: dict[str, list[dict[str, Any]]] = {"coco": [], "custom": [], "smoke": [], "weapon": []}
        for item in claimed:
            key = item["session"].ip
            item["pre"] = {}
            item["infer_frame"], _sx, _sy = self._prepare_frame(item["frame"])
            pipeline = self._purpose_pipeline(key)
            if pipeline.get("coco") and self._coco_model is not None:
                ids = pipeline.get("coco_classes")
                class_filter = ALLOWED_COCO_CLASS_IDS if ids is None else frozenset(int(c) for c in ids)
                if class_filter:
                    item["coco_classes"] = frozenset(class_filter)
                    groups["coco"].append(item)
            # Same rule _infer_frame applies (it increments the counter itself).
            n = self._specialist_count.get(key, 0)
            specialists_due = n % self._specialist_every == 0 or key not in self._last_specialist
            if specialists_due:
                if pipeline.get("custom") and self._custom_model is not None:
                    groups["custom"].append(item)
                if pipeline.get("smoke") and self._smoke_model is not None:
                    groups["smoke"].append(item)
                if pipeline.get("weapon") and self._weapon_model is not None:
                    groups["weapon"].append(item)

        if groups["coco"]:
            union: set[int] = set()
            for item in groups["coco"]:
                union |= item["coco_classes"]
            results = self._predict_batch(
                self._coco_model, [i["infer_frame"] for i in groups["coco"]], classes=union, tag="coco"
            )
            pose_items = []
            for item, r in zip(groups["coco"], results):
                r = self._keep_classes(r, item["coco_classes"])
                item["pre"]["coco"] = r
                boxes = getattr(r, "boxes", None)
                if (
                    self._pose_model is not None
                    and boxes is not None
                    and len(boxes)
                    and 0 in boxes.cls.int().tolist()
                ):
                    pose_items.append(item)
            if pose_items:
                results = self._predict_batch(
                    self._pose_model,
                    [i["infer_frame"] for i in pose_items],
                    min_conf=POSE_CONF,
                    imgsz=POSE_IMGSZ,
                    tag="pose",
                )
                for item, r in zip(pose_items, results):
                    item["pre"]["pose"] = r

        if groups["custom"]:
            custom_ids = custom_only_class_ids(self._custom_model)
            results = self._predict_batch(
                self._custom_model,
                [i["infer_frame"] for i in groups["custom"]],
                classes=custom_ids or None,
                tag="custom",
            )
            for item, r in zip(groups["custom"], results):
                item["pre"]["custom"] = r
        for tag, model, min_conf in (
            ("smoke", self._smoke_model, SMOKE_FIRE_MIN_CONF),
            ("weapon", self._weapon_model, WEAPON_MIN_CONF),
        ):
            if groups[tag]:
                results = self._predict_batch(
                    model, [i["infer_frame"] for i in groups[tag]], min_conf=min_conf, tag=tag
                )
                for item, r in zip(groups[tag], results):
                    item["pre"][tag] = r

    def _batched_post(self, item: dict[str, Any], claimed_at: float) -> None:
        """CPU side for one camera: parse / faces / tracking / identity / publish."""
        self._prefetch.results = item.get("pre") or {}
        try:
            self._infer_and_publish(
                item["session"],
                item["frame"],
                item["seq"],
                item["frame_ts"],
                item["prev_seq"],
                claimed_at,
                started_at=claimed_at,
            )
        finally:
            unused = self._prefetch.results
            self._prefetch.results = None
            if unused:
                self._metrics.record_stage("batch.prefetch_unused", 0.0)
            with self._inflight_lock:
                self._inflight -= 1

    def _batched_dispatch_loop(self) -> None:
        # At most one extra batch waiting beyond the workers, so frames do not go stale.
        max_inflight = self._infer_workers + self._batch_max
        while self._running:
            try:
                self._batched_dispatch_once(max_inflight)
            except Exception as exc:
                print(f"[live] batched dispatch loop error (continuing): {exc}")
                time.sleep(0.2)

    def _batched_dispatch_once(self, max_inflight: int) -> None:
        """One dispatcher step: claim due cameras, batch their GPU work, hand off CPU work."""
        if self._offline_pause:
            time.sleep(0.1)
            return
        with self._inflight_lock:
            room = max_inflight - self._inflight
        limit = min(self._batch_max, room)
        if limit <= 0:
            time.sleep(0.005)
            return
        now = time.time()
        claimed = self._claim_due_sessions(now, limit)
        if not claimed:
            time.sleep(0.01)
            return
        try:
            self._gpu_prefetch(claimed)
        except Exception as exc:
            # Fall back to per-frame predicts inside _infer_frame for this batch.
            print(f"[live] batched GPU stage failed ({len(claimed)} cams): {exc}")
            for item in claimed:
                item["pre"] = {}
        executor = self._infer_executor
        for item in claimed:
            with self._inflight_lock:
                self._inflight += 1
            try:
                if executor is None:
                    raise RuntimeError("executor stopped")
                executor.submit(self._batched_post, item, now)
            except Exception:
                with self._inflight_lock:
                    self._inflight -= 1
                with item["session"].infer_lock:
                    item["session"].infer_busy = False

    def _infer_shard_loop(self, worker_id: int) -> None:
        """
        Dedicated worker loop for one contiguous camera shard.
        Runs in parallel with other workers via ThreadPoolExecutor.
        """
        while self._running:
            loop_start = time.time()
            sessions: list[_CameraSession] = []
            try:
                sessions = self._sessions_for_worker(worker_id)
                for session in sessions:
                    if not self._running:
                        break
                    try:
                        self._run_camera_inference(session)
                    except Exception as exc:
                        print(f"[live] infer worker {worker_id} error [{session.ip}] (continuing): {exc}")
                        with session.infer_lock:
                            session.infer_busy = False
                        time.sleep(0.05)
            except Exception as exc:
                print(f"[live] infer worker {worker_id} loop error (continuing): {exc}")
                time.sleep(0.2)
            elapsed = time.time() - loop_start
            wait = self._infer_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            elif not sessions:
                time.sleep(0.05)

    def _limit_size(self, frame: np.ndarray) -> np.ndarray:
        """Downscale only when ML_LIVE_MAX_WIDTH/HEIGHT cap is set; 0 = native passthrough."""
        h, w = frame.shape[:2]
        max_w = self._max_width
        max_h = self._max_height
        if max_w <= 0 and max_h <= 0:
            return frame
        if max_w <= 0:
            max_w = w
        if max_h <= 0:
            max_h = h
        if w <= max_w and h <= max_h:
            return frame
        scale = min(max_w / w, max_h / h)
        return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    def _prepare_attendance_frame(self, frame: np.ndarray, target_width: int) -> np.ndarray:
        """Native main-stream frame; only downscale when wider than target (never upscale)."""
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return frame
        target_width = max(640, min(4096, int(target_width or 3840)))
        if w <= target_width:
            return frame
        scale = target_width / float(w)
        return cv2.resize(
            frame,
            (target_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def iter_mjpeg_attendance(self, key: str, *, target_width: int = 3840) -> Iterator[bytes]:
        """Full main-stream MJPEG for attendance clips (higher quality than raw preview)."""
        return self._iter_mjpeg_sync(key, "attendance", target_width)

    def _render_session(self, session: _CameraSession) -> bool:
        """Latest frame + last Result Buffer -> preview JPEG for one camera."""
        t_g = time.perf_counter()
        live = session.stream.get_frame()
        if live is None:
            return False
        with self._det_lock:
            boxes = list(self._detections.get(session.ip, []))
        if self._osd_enrolled_staff_only:
            boxes = filter_enrolled_staff_detections(boxes)
        t_d = time.perf_counter()
        # draw_detections copies when there are boxes; otherwise just resize+encode.
        framed = draw_detections(live, boxes) if boxes else live
        t_e = time.perf_counter()
        jpeg = encode_jpeg(self._limit_size(framed), self._jpeg_quality)
        t_s = time.perf_counter()
        session.set_frame(jpeg, boxes)
        self._metrics.record_stage("render.frame_copy", t_d - t_g)
        self._metrics.record_stage("render.draw", t_e - t_d)
        self._metrics.record_stage("render.jpeg_encode", t_s - t_e)
        return True

    def _render_loop(self):
        """Render Thread: latest frame + last Result Buffer -> JPEG. Never waits on inference."""
        while self._running:
            loop_start = time.time()
            with self._lock:
                sessions = list(self._sessions.values())
            now_m = time.monotonic()
            for session in sessions:
                if self._render_only_watched and (now_m - session.last_viewed_at) > self._render_grace:
                    continue
                try:
                    self._render_session(session)
                except Exception as exc:
                    print(f"[live] render error [{session.ip}] (continuing): {exc}")
                    time.sleep(0.01)
            self._metrics.record_stage("render.loop_all_cams", time.time() - loop_start)
            elapsed = time.time() - loop_start
            wait = self._frame_interval - elapsed
            if wait > 0:
                time.sleep(wait)


_manager: LiveStreamManager | None = None
_manager_lock = threading.Lock()


def get_live_manager() -> LiveStreamManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = LiveStreamManager()
        return _manager
