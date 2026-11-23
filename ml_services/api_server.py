"""
FastAPI inference server for Custom VMS integration.

Run (Server 2 / local ML host):
  cd ml_services
  python api_server.py

Or:
  uvicorn api_server:app --host 0.0.0.0 --port 8100

Security:
  ML_API_KEY            shared secret; every route except / and /health then requires the
                        header X-ML-API-Key (or ?key= on the /live/cam/... image / MJPEG
                        routes that browsers load through <img>). Unset = open API (a loud
                        WARNING is logged at startup) so existing deployments keep working.
  ML_CORS_ORIGINS       comma list of allowed browser origins (default "*", without credentials).
  ML_MAX_IMAGE_UPLOAD_MB / ML_MAX_AUDIO_UPLOAD_MB / ML_MAX_VIDEO_UPLOAD_MB  upload limits.
  ML_LOCAL_VIDEO_ROOTS  folders local video_path values must resolve into (os.pathsep list).
"""
from __future__ import annotations

import hmac
import logging
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from load_env import load_ml_env

load_ml_env()

from inference_engine import (  # noqa: E402
    decode_image,
    detect_image,
    extract_face_embedding,
    health_status,
    recognize_face,
    reload_face_db,
    validate_human_face,
    warmup_all_models,
)
from live_stream import get_live_manager, redact_url, stream_url_allowed  # noqa: E402

_log = logging.getLogger("ml.api")
_live = get_live_manager()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ---------- auth ----------

_API_KEY = os.getenv("ML_API_KEY", "").strip()
# Fail closed: with no ML_API_KEY every route except "/" is refused, unless this is explicitly a
# throw-away dev box (ML_ALLOW_NO_API_KEY=true).
_ALLOW_NO_KEY = _env_flag("ML_ALLOW_NO_API_KEY", False)
# /health shows GPU, camera and stream details, so it needs the key too (the backend's probes send
# it). ML_HEALTH_PUBLIC=true reopens it for an external uptime monitor.
_OPEN_PATHS = frozenset({"/", "/health"}) if _env_flag("ML_HEALTH_PUBLIC", False) else frozenset({"/"})
# Browser <img> / MJPEG routes may pass the key as ?key= (an <img> cannot send headers).
_QUERY_KEY_PREFIX = "/live/cam/"


def _key_ok(given: str | None) -> bool:
    return bool(given) and hmac.compare_digest(str(given).encode(), _API_KEY.encode())


def require_api_key(request: Request) -> None:
    """X-ML-API-Key must match ML_API_KEY. With no key configured, refuse (fail closed)."""
    if request.url.path in _OPEN_PATHS or request.method == "OPTIONS":
        return
    if not _API_KEY:
        if _ALLOW_NO_KEY:
            return
        raise HTTPException(
            status_code=503,
            detail="ML API is locked: ML_API_KEY is not configured on this server.",
        )
    if _key_ok(request.headers.get("x-ml-api-key")):
        return
    if request.url.path.startswith(_QUERY_KEY_PREFIX) and request.method == "GET":
        if _key_ok(request.query_params.get("key")):
            return
    raise HTTPException(status_code=401, detail="Missing or invalid X-ML-API-Key")


# ---------- upload limits ----------

_MB = 1024 * 1024
MAX_IMAGE_BYTES = int(max(1.0, _env_float("ML_MAX_IMAGE_UPLOAD_MB", 25)) * _MB)
MAX_AUDIO_BYTES = int(max(1.0, _env_float("ML_MAX_AUDIO_UPLOAD_MB", 25)) * _MB)
# Backend allows 8 GB recordings (ML_VIDEO_SEARCH_MAX_BYTES); keep the same default.
MAX_VIDEO_BYTES = int(max(1.0, _env_float("ML_MAX_VIDEO_UPLOAD_MB", 8192)) * _MB)
_VIDEO_UPLOAD_PATHS = ("/search/video", "/analyze/video")


def _read_limited(upload: UploadFile, limit: int, what: str = "File") -> bytes:
    """Bounded read of an UploadFile (sync endpoints)."""
    size = getattr(upload, "size", None)
    if size is not None and size > limit:
        raise HTTPException(status_code=413, detail=f"{what} too large (max {limit // _MB} MB).")
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"{what} too large (max {limit // _MB} MB).")
    return data


def _check_upload_size(upload: UploadFile | None, limit: int, what: str = "Video") -> None:
    if upload is None:
        return
    size = getattr(upload, "size", None)
    if size is None:
        try:
            pos = upload.file.tell()
            upload.file.seek(0, os.SEEK_END)
            size = upload.file.tell()
            upload.file.seek(pos)
        except Exception:
            size = None
    if size is not None and size > limit:
        raise HTTPException(status_code=413, detail=f"{what} too large (max {limit // _MB} MB).")


# ---------- local video paths ----------

def _local_video_roots() -> list[Path]:
    """Folders a local video_path must resolve into.

    Defaults match where the backend stores uploads when the ML node is on the same host
    (backend/media/video_search|video_analyze/<job>/source.mp4) plus this service's own tmp dirs.
    """
    raw = os.getenv("ML_LOCAL_VIDEO_ROOTS", "").strip()
    if raw:
        items = [Path(p) for p in raw.split(os.pathsep) if p.strip()]
    else:
        here = Path(__file__).resolve().parent
        items = [
            here.parent / "backend" / "media" / "video_search",
            here.parent / "backend" / "media" / "video_analyze",
            here / "tmp" / "video_search",
            here / "tmp" / "video_analyze",
        ]
    out = []
    for p in items:
        try:
            out.append(p.resolve())
        except OSError:
            continue
    return out


def _resolve_local_video(path_text: str) -> str:
    """Real path of an allowed local video, else HTTP 400."""
    # The backend sends local paths when it reaches this node via localhost; keep the default on.
    if not _env_flag("ML_ALLOW_LOCAL_VIDEO_PATHS", True):
        raise HTTPException(status_code=400, detail="Local video paths are disabled.")
    try:
        real = Path(path_text).resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid local video path.") from None
    if not real.is_file() or not any(real.is_relative_to(root) for root in _local_video_roots()):
        raise HTTPException(status_code=400, detail="Invalid local video path.")
    return str(real)


# ---------- app ----------

def _boot_live_streams():
    try:
        warmup_all_models()
    except Exception as exc:
        print(f"[live] Model warmup failed (continuing): {exc}")
    try:
        from plate_recognizer import get_plate_engine

        get_plate_engine()
    except Exception as exc:
        print(f"[live] Plate engine failed (ANPR off): {exc}")
    try:
        _live.configure_from_env()
        _live.ensure_started()
    except Exception as exc:
        print(f"[live] Background boot failed: {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not _API_KEY:
        msg = (
            "[security] ML_API_KEY is not set. "
            + ("ML_ALLOW_NO_API_KEY=true: the API is OPEN to anyone who can reach this port."
               if _ALLOW_NO_KEY else
               "Every API route is refused (503) until ML_API_KEY is set here and in the backend.")
        )
        _log.warning(msg)
        print(msg)
    # Sync endpoints (and sync MJPEG fallbacks) each hold an anyio worker thread; the default
    # 40 tokens let a handful of viewers block every other request.
    try:
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = max(
            40, int(_env_float("ML_HTTP_THREADS", 200))
        )
    except Exception as exc:
        print(f"[api] Could not raise thread limit: {exc}")
    threading.Thread(target=_boot_live_streams, daemon=True, name="live-boot").start()
    yield
    _live.stop()


app = FastAPI(
    title="Custom ML Inference",
    version="1.3.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)

_cors_origins = [o.strip() for o in os.getenv("ML_CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Credentials with a wildcard origin let any site make cookie-bearing calls; only allow
    # them when explicit origins are configured.
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _content_length_limit(request: Request, call_next):
    """Reject oversized file uploads before they are spooled (Content-Length check).

    Only multipart uploads are limited here; JSON bodies (e.g. /reload/faces with every
    staff embedding) are left alone.
    """
    raw = request.headers.get("content-length")
    is_upload = request.headers.get("content-type", "").lower().startswith("multipart/")
    if is_upload and raw and raw.isdigit():
        limit = MAX_VIDEO_BYTES if request.url.path.startswith(_VIDEO_UPLOAD_PATHS) else (
            max(MAX_IMAGE_BYTES, MAX_AUDIO_BYTES) + _MB
        )
        if int(raw) > limit:
            return JSONResponse(status_code=413, content={"detail": f"Request body too large (max {limit // _MB} MB)."})
    return await call_next(request)


class FaceEmbeddingEntry(BaseModel):
    identity: str
    embedding: list[float] = Field(default_factory=list)
    quality: float | None = None


class ThresholdCalibrationRequest(BaseModel):
    genuine_scores: list[float] = Field(default_factory=list)
    impostor_scores: list[float] = Field(default_factory=list)
    target_far: float = 0.01


class CameraRegisterEntry(BaseModel):
    key: str
    rtsp_url: str
    purpose: str = ""
    purposes: list[str] = Field(default_factory=list)


def _resolve_live_stream(
    key: str,
    rtsp_url: str | None = None,
    *,
    purpose: str = "",
    purposes: str = "",
    require_engine: bool = True,
) -> str:
    # Do not block HTTP on YOLO load. Raw JPEG/RTSP can serve before infer is ready.
    if require_engine and not _live.is_ready():
        raise HTTPException(status_code=503, detail="Live engine still starting")
    key = (key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="Missing camera key")
    if not _live.is_registered(key):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Camera {key} is not assigned to this ML server. "
                "Assign it in Camera Distribution and sync; unassigned cameras stay idle."
            ),
        )
    purpose_list = [p.strip() for p in (purposes or "").split(",") if p.strip()]
    if purpose_list or (purpose or "").strip():
        applied = _live.set_camera_purposes(key, purpose=purpose, purposes=purpose_list)
        print(f"[live] Purposes for {key}: {applied}")
    resolved = _live.resolve_rtsp_url(key, rtsp_url)
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail=f"Camera {key} is not registered. Sync cameras from Django backend.",
        )
    if not _live.ensure_camera(key, resolved):
        raise HTTPException(status_code=503, detail=f"Could not open camera stream for {key}")
    return resolved


def _mjpeg_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "X-Accel-Buffering": "no",
    }


def _redact_sessions(data: Any) -> Any:
    if isinstance(data, dict) and isinstance(data.get("sessions"), list):
        for row in data["sessions"]:
            if isinstance(row, dict) and row.get("rtsp_url"):
                row["rtsp_url"] = redact_url(row["rtsp_url"])
    return data


@app.get("/")
def root():
    return {"service": "Custom ML Inference", "health": "/health"}


@app.get("/health")
def health():
    data = health_status()
    data["live_streams"] = _live.status()  # rtsp_url already redacted
    try:
        from camera_session import get_camera_session_manager

        data["camera_sessions"] = _redact_sessions(get_camera_session_manager().status())
    except Exception as exc:
        data["camera_sessions"] = {"error": str(exc)}
    data["api_key_required"] = bool(_API_KEY)
    return data


@app.get("/camera-sessions/status")
def camera_sessions_status():
    """One FFmpeg/NVDEC ingest per camera — shared frame buffer status."""
    from camera_session import get_camera_session_manager

    return _redact_sessions(get_camera_session_manager().status())


@app.post("/reload/faces")
def reload_faces(payload: list[FaceEmbeddingEntry] | None = None):
    entries = [
        {
            "identity": item.identity,
            "embedding": item.embedding,
            **({"quality": item.quality} if item.quality is not None else {}),
        }
        for item in (payload or [])
    ]
    count = reload_face_db(entries)
    return {"reloaded": True, "known_faces": count, "db_embeddings": len(entries)}


@app.post("/faces/calibrate")
def calibrate_face_threshold(payload: ThresholdCalibrationRequest):
    from face_calibration import suggest_threshold

    return suggest_threshold(
        payload.genuine_scores,
        payload.impostor_scores,
        target_far=payload.target_far,
    )


# Inference endpoints are plain `def`: FastAPI runs them in the worker thread pool, so a
# blocking YOLO / face call no longer freezes the event loop (and every MJPEG stream).


@app.post("/faces/extract")
def extract_face_embedding_endpoint(image: UploadFile = File(...)):
    data = _read_limited(image, MAX_IMAGE_BYTES, "Image")
    try:
        frame = decode_image(data)
        result = extract_face_embedding(frame)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/reid/extract")
def extract_reid_embedding_endpoint(image: UploadFile = File(...)):
    from reid_extractor import extract_reid_embedding

    data = _read_limited(image, MAX_IMAGE_BYTES, "Image")
    try:
        frame = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    embedding = extract_reid_embedding(frame)
    if not embedding:
        raise HTTPException(status_code=400, detail="Could not extract appearance embedding from image.")
    return {"embedding": embedding, "dim": len(embedding)}


@app.post("/detect/image")
def detect(
    image: UploadFile = File(...),
    conf: float = 0.25,
    iou: float = 0.45,
    recognize_faces: bool = True,
):
    data = _read_limited(image, MAX_IMAGE_BYTES, "Image")
    try:
        frame = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    detections = detect_image(
        frame,
        conf=conf,
        iou=iou,
        recognize_faces=recognize_faces,
    )
    return {"detections": detections, "count": len(detections)}


@app.post("/camera-health/analyze")
def camera_health_analyze(
    image: UploadFile | None = File(None),
    purposes: str = Form(""),
    roi_json: str = Form(""),
    detections_json: str = Form(""),
    previous_fingerprint_json: str = Form(""),
    rtsp_available: bool = Form(True),
    fps: float | None = Form(None),
    frame_age_sec: float | None = Form(None),
    dropped_frames: int | None = Form(None),
    camera_key: str = Form(""),
):
    """Analyze one camera frame for health / visibility (OpenCV + purpose checks)."""
    import json

    from cam_health import analyze_camera_health

    frame = None
    if image is not None:
        data = _read_limited(image, MAX_IMAGE_BYTES, "Image")
        if data:
            try:
                frame = decode_image(data)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    purpose_list = [p.strip() for p in purposes.split(",") if p.strip()]
    roi = None
    detections = None
    prev_fp = None
    try:
        if roi_json.strip():
            roi = json.loads(roi_json)
        if detections_json.strip():
            detections = json.loads(detections_json)
        if previous_fingerprint_json.strip():
            prev_fp = json.loads(previous_fingerprint_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON field: {exc}") from exc

    # Prefer live detections when camera_key is registered
    key = (camera_key or "").strip()
    # Freeze evidence from the live decoder: frame_seq / frame_ts of the live session, compared
    # with the values seen on the previous call for this camera (one-off uploads: None).
    meta = _live.get_frame_meta(key) if key else None
    with _health_prev_lock:
        prev = dict(_health_prev.get(key) or {}) if meta is not None else {}
    if key and _live.is_ready() and detections is None:
        try:
            snap = _live.get_detection_snapshot(key)
            detections = snap.get("detections") or []
            if frame is None:
                jpeg = _live.get_raw_jpeg_bytes(key)
                if jpeg:
                    frame = decode_image(jpeg)
        except Exception:
            pass

    result = analyze_camera_health(
        frame,
        purposes=purpose_list,
        roi=roi if isinstance(roi, dict) else None,
        detections=detections if isinstance(detections, list) else None,
        previous_fingerprint=prev_fp if isinstance(prev_fp, list) else None,
        rtsp_available=bool(rtsp_available),
        fps=fps,
        frame_age_sec=frame_age_sec,
        dropped_frames=dropped_frames,
        frame_seq=meta[0] if meta is not None else None,
        previous_frame_seq=prev.get("seq"),
        frame_ts=meta[1] if meta is not None else None,
        previous_frame_ts=prev.get("ts"),
        previous_identical_streak=prev.get("streak"),
    )
    if meta is not None:
        stream = result.get("stream") if isinstance(result, dict) else None
        streak = stream.get("identical_streak") if isinstance(stream, dict) else None
        now = time.time()
        with _health_prev_lock:
            _health_prev[key] = {"seq": meta[0], "ts": meta[1], "streak": streak, "at": now}
            if len(_health_prev) > 2000:  # drop cameras not checked for an hour
                for k in [k for k, v in _health_prev.items() if now - v.get("at", 0) > 3600]:
                    _health_prev.pop(k, None)
    return result


# camera key → {"seq", "ts", "streak", "at"} from the previous /camera-health/analyze call
_health_prev: dict[str, dict[str, Any]] = {}
_health_prev_lock = threading.Lock()


@app.post("/plates/detect")
def detect_plates(
    image: UploadFile = File(...),
    conf: float | None = None,
    save: bool = True,
    camera_key: str = "",
):
    """License plate YOLO + PaddleOCR PP-OCRv5. Saves accepted plates under media/licence plates/."""
    from plate_recognizer import get_plate_engine

    data = _read_limited(image, MAX_IMAGE_BYTES, "Image")
    try:
        frame = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    engine = get_plate_engine()
    if not engine.available:
        raise HTTPException(
            status_code=503,
            detail="Plate model/OCR unavailable. Install paddleocr (+ paddlepaddle-gpu) and ensure plate weights exist.",
        )
    detections = engine.detect_and_read(
        frame,
        camera_key=camera_key.strip(),
        conf=conf,
        save=save,
        force_save=True,
    )
    return {
        "detections": detections,
        "count": len(detections),
        "accepted": sum(1 for d in detections if d.get("accepted")),
        "media_dir": "licence plates",
    }


@app.post("/voice/transcribe")
def voice_transcribe(
    audio: UploadFile = File(...),
    language: str = Form(""),
    prompt: str = Form(""),
):
    """On-prem Whisper STT for the CIIS voice agent (short WAV utterances)."""
    import speech_to_text

    if not speech_to_text.available():
        raise HTTPException(status_code=503, detail="faster-whisper is not installed on this ML node.")
    data = _read_limited(audio, MAX_AUDIO_BYTES, "Audio")
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio.")
    try:
        return speech_to_text.transcribe(data, language=language.strip() or None, prompt=prompt.strip())
    except Exception as exc:
        logging.getLogger("ml.stt").exception("voice transcribe failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc


@app.post("/recognize/face")
def recognize(image: UploadFile = File(...)):
    data = _read_limited(image, MAX_IMAGE_BYTES, "Image")
    try:
        frame = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return recognize_face(frame)


@app.post("/validate/human-face")
def validate_face(image: UploadFile = File(...)):
    data = _read_limited(image, MAX_IMAGE_BYTES, "Image")
    try:
        frame = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return validate_human_face(frame)


@app.get("/live/status")
def live_status():
    return _live.status()  # rtsp_url values are redacted (user:***@host)


@app.get("/live/timing")
def live_timing(reset: bool = False):
    """Average time per live inference step since start (or last reset)."""
    return _live.timing(reset=reset)


@app.get("/live/metrics")
def live_metrics(reset: bool = False, cameras: bool = True):
    """Per-camera / per-model inference metrics, SLA coverage, GPU/CPU/RAM.

    reset=true clears the rolling counters after returning them (start of a load-test step).
    cameras=false drops the per-camera list (summary + models + system only).
    """
    return _live.metrics(reset=reset, include_cameras=cameras)


@app.get("/live/motion/status")
def live_motion_status():
    return _live.motion_status()


@app.get("/live/motion/clips")
def live_motion_clips(limit: int = 50):
    """Finished motion clips waiting for the backend (oldest first)."""
    return {"clips": _live.motion_clips.list_ready(limit=max(1, min(int(limit), 200)))}


@app.get("/live/motion/clips/{clip_id}/{kind}")
def live_motion_clip_file(clip_id: str, kind: str):
    if kind not in ("video", "poster", "end"):
        raise HTTPException(status_code=404, detail="Unknown file")
    path = _live.motion_clips.file_path(clip_id, kind)
    if path is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    return FileResponse(path, media_type="video/mp4" if kind == "video" else "image/jpeg")


@app.delete("/live/motion/clips/{clip_id}")
def live_motion_clip_delete(clip_id: str):
    """Backend calls this after it stored the clip."""
    return {"deleted": _live.motion_clips.delete_clip(clip_id) > 0}


@app.post("/live/register/bulk")
def register_cameras_bulk(
    payload: list[CameraRegisterEntry],
    replace: bool = False,
):
    entries = []
    rejected: list[str] = []
    for item in payload:
        if not item.key.strip() or not item.rtsp_url.strip():
            continue
        if not stream_url_allowed(item.rtsp_url):
            # Only rtsp:// / rtsps:// (ML_ALLOWED_STREAM_SCHEMES) — never local files or other URLs.
            rejected.append(item.key.strip())
            continue
        entry = {
            "key": item.key.strip(),
            "rtsp_url": item.rtsp_url.strip(),
        }
        if item.purpose.strip():
            entry["purpose"] = item.purpose.strip()
        purposes = [p.strip() for p in (item.purposes or []) if str(p).strip()]
        if purposes:
            entry["purposes"] = purposes
        entries.append(entry)
    if rejected:
        print(f"[live] register/bulk rejected {len(rejected)} camera(s) with a disallowed URL scheme: {rejected[:10]}")
    result = _live.register_cameras_bulk(entries, replace=bool(replace))
    if rejected:
        result = {**result, "rejected": rejected}
    if not _live.ensure_started():
        print("[live] Warning: camera registry updated but infer loops did not start")
    return result


@app.delete("/live/cam/{camera_key}/register")
def unregister_camera(camera_key: str):
    key = camera_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="camera_key required")
    removed = _live.unregister_camera(key)
    return {"removed": removed, "key": key}


_EMBEDDING_KEYS = ("face_embedding", "reid_embedding")


def _public_detections(detections: list[dict[str, Any]], include_embeddings: bool) -> list[dict[str, Any]]:
    """HTTP view of live detections: no embeddings unless asked, motion_path capped."""
    try:
        max_path = max(0, int(os.getenv("ML_DETECTIONS_MAX_PATH_POINTS", "30")))
    except ValueError:
        max_path = 30
    out = []
    for det in detections or []:
        row = dict(det)
        if not include_embeddings:
            for k in _EMBEDDING_KEYS:
                row.pop(k, None)
        path = row.get("motion_path")
        if isinstance(path, list) and len(path) > max_path:
            row["motion_path"] = path[-max_path:] if max_path else []
        out.append(row)
    return out


@app.get("/live/cam/{camera_key}/detections")
def live_detections(
    camera_key: str,
    rtsp_url: str | None = None,
    purpose: str = "",
    purposes: str = "",
    include_embeddings: bool | None = None,
):
    """Latest detections. face_embedding / reid_embedding only with include_embeddings=1
    (default from ML_DETECTIONS_INCLUDE_EMBEDDINGS, false)."""
    key = camera_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="camera_key required")
    # Warm-up: return empty 200 instead of 503 so clients can keep polling quietly.
    if not _live.is_ready():
        return {
            "ip": key,
            "key": key,
            "detections": [],
            "purposes": [],
            "frame_width": 0,
            "frame_height": 0,
            "display_width": 0,
            "display_height": 0,
            "has_evidence": False,
            "count": 0,
        }
    if include_embeddings is None:
        include_embeddings = _env_flag("ML_DETECTIONS_INCLUDE_EMBEDDINGS", False)
    _resolve_live_stream(key, rtsp_url, purpose=purpose, purposes=purposes)
    snapshot = _live.get_detection_snapshot(key)
    detections = _public_detections(snapshot.get("detections") or [], bool(include_embeddings))
    return {
        "ip": key,
        "key": key,
        "detections": detections,
        "purposes": _live._purposes_for(key),
        "frame_width": snapshot.get("frame_width") or 0,
        "frame_height": snapshot.get("frame_height") or 0,
        "display_width": snapshot.get("display_width") or 0,
        "display_height": snapshot.get("display_height") or 0,
        "has_evidence": bool(snapshot.get("has_evidence")),
        "count": len(detections),
    }


@app.get("/live/cam/{camera_key}/jpeg")
def live_jpeg(
    camera_key: str,
    rtsp_url: str | None = None,
    purpose: str = "",
    purposes: str = "",
):
    """Single latest JPEG — annotated when ready, otherwise raw (fast first paint)."""
    key = camera_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="camera_key required")
    _resolve_live_stream(key, rtsp_url, purpose=purpose, purposes=purposes, require_engine=False)
    if _live.is_ready():
        frame = _live.wait_for_preview_jpeg(key, timeout_sec=2.5)
    else:
        frame = _live.wait_for_raw_jpeg(key, timeout_sec=2.5)
    if not frame:
        raise HTTPException(status_code=503, detail="No frame yet")
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers=_mjpeg_headers(),
    )


@app.get("/live/cam/{camera_key}/jpeg/raw")
def live_jpeg_raw(
    camera_key: str,
    rtsp_url: str | None = None,
    purpose: str = "",
    purposes: str = "",
):
    """Single raw JPEG from the live RTSP session. Does not wait for YOLO warmup."""
    key = camera_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="camera_key required")
    _resolve_live_stream(key, rtsp_url, purpose=purpose, purposes=purposes, require_engine=False)
    frame = _live.wait_for_raw_jpeg(key, timeout_sec=2.0)
    if not frame:
        raise HTTPException(status_code=503, detail="No frame yet")
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers=_mjpeg_headers(),
    )


@app.get("/live/cam/{camera_key}/jpeg/evidence")
def live_jpeg_evidence(
    camera_key: str,
    rtsp_url: str | None = None,
    purpose: str = "",
    purposes: str = "",
):
    """
    Raw JPEG of the SAME frame YOLO last ran on (evidence buffer).
    Use this for detection snapshots — do not re-open RTSP in Django.
    """
    key = camera_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="camera_key required")
    _resolve_live_stream(key, rtsp_url, purpose=purpose, purposes=purposes, require_engine=False)
    frame = _live.get_evidence_jpeg(key)
    if not frame:
        raise HTTPException(status_code=503, detail="No evidence frame yet")
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers=_mjpeg_headers(),
    )


@app.get("/live/cam/{camera_key}/jpeg/attendance")
def live_jpeg_attendance(
    camera_key: str,
    rtsp_url: str | None = None,
    width: int = 3840,
):
    """Single JPEG from the same RTSP session used for attendance clips.

    Note: the live ingest is scaled (ML_RTSP_SCALE_*, default 1280x720), so "width" only
    downsizes; the default 3840 does not give a 4K frame unless native ingest is enabled.
    """
    key = camera_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="camera_key required")
    _resolve_live_stream(key, rtsp_url, require_engine=False)
    target_width = max(640, min(4096, int(width or 3840)))
    frame = _live.wait_for_raw_jpeg(key, timeout_sec=2.5, target_width=target_width)
    if not frame:
        raise HTTPException(status_code=503, detail="No frame yet")
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers=_mjpeg_headers(),
    )


def _mjpeg_response(request: Request, key: str, mode: str, target_width: int = 3840) -> StreamingResponse:
    # Async generator: no worker thread is held per viewer; it ends when the session is
    # closed, the stream idles (ML_MJPEG_IDLE_TIMEOUT_SEC) or the client disconnects.
    return StreamingResponse(
        _live.aiter_mjpeg(key, mode=mode, target_width=target_width, is_disconnected=request.is_disconnected),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers=_mjpeg_headers(),
    )


@app.get("/live/cam/{camera_key}/mjpeg")
def live_mjpeg(
    request: Request,
    camera_key: str,
    rtsp_url: str | None = None,
    purpose: str = "",
    purposes: str = "",
):
    key = camera_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="camera_key required")
    _resolve_live_stream(key, rtsp_url, purpose=purpose, purposes=purposes)
    return _mjpeg_response(request, key, "live")


@app.get("/live/cam/{camera_key}/mjpeg/raw")
def live_mjpeg_raw(
    request: Request,
    camera_key: str,
    rtsp_url: str | None = None,
    purpose: str = "",
    purposes: str = "",
):
    key = camera_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="camera_key required")
    _resolve_live_stream(key, rtsp_url, purpose=purpose, purposes=purposes, require_engine=False)
    return _mjpeg_response(request, key, "raw")


@app.get("/live/cam/{camera_key}/mjpeg/attendance")
def live_mjpeg_attendance(
    request: Request,
    camera_key: str,
    rtsp_url: str | None = None,
    width: int = 3840,
):
    """Main-stream MJPEG for attendance clip capture (reuses ML RTSP session; see jpeg/attendance note)."""
    key = camera_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="camera_key required")
    _resolve_live_stream(key, rtsp_url, require_engine=False)
    target_width = max(640, min(4096, int(width or 3840)))
    return _mjpeg_response(request, key, "attendance", target_width)


# ---------- Person Journey (CIIS) — parallel pipeline, does not alter live inference ----------


class JourneyCameraEntry(BaseModel):
    key: str
    rtsp_url: str
    camera_id: int | None = None
    zone: str = ""
    name: str = ""


class JourneyRegisterBulk(BaseModel):
    cameras: list[JourneyCameraEntry] = Field(default_factory=list)
    backend_ingest_url: str = ""
    ingest_token: str = ""


@app.get("/journey/status")
def journey_status():
    from journey_manager import get_journey_manager

    return get_journey_manager().status()


@app.post("/journey/register/bulk")
def journey_register_bulk(payload: JourneyRegisterBulk):
    from journey_manager import get_journey_manager

    data = payload.model_dump()
    rejected = [c["key"] for c in data.get("cameras") or [] if not stream_url_allowed(c.get("rtsp_url"))]
    if rejected:
        data["cameras"] = [c for c in data["cameras"] if stream_url_allowed(c.get("rtsp_url"))]
        print(f"[journey] register/bulk rejected {len(rejected)} camera(s) with a disallowed URL scheme")
    mgr = get_journey_manager()
    result = mgr.register_cameras_bulk(data)
    if rejected and isinstance(result, dict):
        result = {**result, "rejected": rejected}
    return result


@app.delete("/journey/cam/{camera_key}")
def journey_unregister_camera(camera_key: str):
    from journey_manager import get_journey_manager

    key = camera_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="camera_key required")
    removed = get_journey_manager().unregister_camera(key)
    return {"removed": removed, "key": key}


@app.post("/journey/stop-all")
def journey_stop_all():
    from journey_manager import get_journey_manager

    mgr = get_journey_manager()
    before = mgr.status()
    mgr.stop_all()
    return {"stopped": before.get("running_pipelines", 0), "cameras": before.get("cameras") or []}


@app.post("/search/video")
def search_video(
    image: UploadFile = File(...),
    video: UploadFile | None = File(default=None),
    video_path: str = Form(default=""),
    face_threshold: float = 0.45,
    reid_threshold: float = 0.88,
    sample_fps: float = 0.0,
    clip_seconds: float = 4.0,
):
    """Find the uploaded image in a camera recording (up to 1 hour). Returns a job id."""
    from video_search_jobs import copy_upload, new_upload_dir, start_search_job

    # Validate everything before creating the job folder (a 400 used to leave a tmp dir behind).
    local_video = (video_path or "").strip()
    stored_video = ""
    cleanup_video = True
    if local_video:
        stored_video = _resolve_local_video(local_video)
        cleanup_video = False
    elif video is None or not video.filename:
        raise HTTPException(status_code=400, detail="video file is required.")
    else:
        _check_upload_size(video, MAX_VIDEO_BYTES)
    _check_upload_size(image, MAX_IMAGE_BYTES, "Image")

    upload_dir = new_upload_dir()
    try:
        image_path = upload_dir / "query.jpg"
        copy_upload(image.file, image_path)
        if not stored_video:
            stored_path = upload_dir / "source.mp4"
            copy_upload(video.file, stored_path)
            stored_video = str(stored_path)
        job_id = start_search_job(
            str(image_path),
            stored_video,
            face_threshold=face_threshold,
            reid_threshold=reid_threshold,
            sample_fps=sample_fps,
            clip_seconds=clip_seconds,
            cleanup_video=cleanup_video,
        )
    except BaseException:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    return {"job_id": job_id, "status": "queued", "progress": 1, "message": "Queued"}


@app.get("/search/video/{job_id}")
def search_video_status(job_id: str):
    from video_search_jobs import get_search_job

    row = get_search_job(job_id.strip())
    if not row:
        raise HTTPException(status_code=404, detail="Search job not found.")
    return row


def _form_flag(value: str, default: bool = True) -> bool:
    raw = (value or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@app.post("/analyze/video")
def analyze_video(
    video: UploadFile | None = File(default=None),
    video_path: str = Form(default=""),
    person: str = Form(default="true"),
    vehicle: str = Form(default="true"),
    weapon: str = Form(default="true"),
    fire: str = Form(default="true"),
    match_staff: str = Form(default="true"),
    plate: str = Form(default="false"),
    sample_fps: float = Form(default=0.5),
):
    """Tag an uploaded video with person / vehicle / weapon / fire / licence plate detections."""
    from video_analyze_jobs import copy_upload, new_upload_dir, start_analyze_job

    want_person = _form_flag(person)
    want_vehicle = _form_flag(vehicle)
    want_weapon = _form_flag(weapon)
    want_fire = _form_flag(fire)
    want_plate = _form_flag(plate)
    if not any([want_person, want_vehicle, want_weapon, want_fire, want_plate]):
        raise HTTPException(status_code=400, detail="Select at least one detection type.")

    # Validate before creating the job folder (a 400 used to leave an empty tmp dir behind).
    local_video = (video_path or "").strip()
    stored_video = ""
    cleanup_video = True
    if local_video:
        stored_video = _resolve_local_video(local_video)
        cleanup_video = False
    elif video is None or not video.filename:
        raise HTTPException(status_code=400, detail="video file is required.")
    else:
        _check_upload_size(video, MAX_VIDEO_BYTES)

    upload_dir: Path | None = None
    try:
        if not stored_video:
            upload_dir = new_upload_dir()
            stored_path = upload_dir / "source.mp4"
            copy_upload(video.file, stored_path)
            stored_video = str(stored_path)
        job_id = start_analyze_job(
            stored_video,
            person=want_person,
            vehicle=want_vehicle,
            weapon=want_weapon,
            fire=want_fire,
            match_staff=_form_flag(match_staff),
            sample_fps=max(0.5, min(float(sample_fps or 0.5), 4.0)),
            cleanup_video=cleanup_video,
            plate=want_plate,
        )
    except BaseException:
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    return {"job_id": job_id, "status": "queued", "progress": 1, "message": "Queued"}


@app.get("/analyze/video/{job_id}")
def analyze_video_status(job_id: str):
    from video_analyze_jobs import get_analyze_job

    row = get_analyze_job(job_id.strip())
    if not row:
        raise HTTPException(status_code=404, detail="Analyze job not found.")
    public = dict(row)
    public.pop("output_path", None)
    # Server file path is also inside the finished result — never return it.
    if isinstance(public.get("result"), dict):
        public["result"] = {k: v for k, v in public["result"].items() if k != "output_path"}
    return public


@app.get("/analyze/video/{job_id}/file")
def analyze_video_file(job_id: str):
    from video_analyze_jobs import get_analyze_job, get_output_path

    row = get_analyze_job(job_id.strip())
    if not row:
        raise HTTPException(status_code=404, detail="Analyze job not found.")
    if str(row.get("status") or "") != "done":
        raise HTTPException(status_code=409, detail="Tagged video is not ready yet.")
    path = get_output_path(job_id.strip())
    if path is None:
        raise HTTPException(status_code=404, detail="Tagged video file is missing.")
    return FileResponse(str(path), media_type="video/mp4", filename=path.name)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("ML_API_HOST", "0.0.0.0")
    port = int(os.getenv("ML_API_PORT", "8100"))
    uvicorn.run("api_server:app", host=host, port=port, reload=False)
