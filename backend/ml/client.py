"""HTTP client for the external ML inference service (ml_services/api_server.py)."""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlencode

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


# ——— Redaction: never let RTSP credentials, API keys or tokens reach logs / clients ———

_URL_CREDENTIALS_RE = re.compile(r"(?P<scheme>\b[a-zA-Z][a-zA-Z0-9+.\-]*://)(?P<creds>[^/\s@'\"<>]+@)")
_SENSITIVE_PARAM_RE = re.compile(
    r"(?P<sep>[?&])(?P<name>key|api_key|apikey|token|rtsp_url|rtsp|password|passwd|pwd|pass|user|username|auth)="
    r"(?P<value>[^&\s'\"<>]*)",
    re.IGNORECASE,
)
# A URL-encoded rtsp://user:pass@ (rtsp%3A%2F%2Fuser%3Apass%40host)
_ENCODED_CREDENTIALS_RE = re.compile(r"(%3A%2F%2F)[^%\s&'\"]+(%3A[^%\s&'\"]*)?%40", re.IGNORECASE)


def redact(text: Any) -> str:
    """Strip URL credentials and sensitive query values (key, token, rtsp_url, …) from text."""
    out = str(text or "")
    if not out:
        return ""
    out = _URL_CREDENTIALS_RE.sub(lambda m: f"{m.group('scheme')}***@", out)
    out = _SENSITIVE_PARAM_RE.sub(lambda m: f"{m.group('sep')}{m.group('name')}=***", out)
    out = _ENCODED_CREDENTIALS_RE.sub(r"\1***%40", out)
    return out


def redact_url(url: str) -> str:
    """Log-safe URL: no credentials and no query string."""
    text = redact(url)
    return text.split("?", 1)[0]


class MLServiceError(Exception):
    """ML call failure. The message is always redacted (no RTSP credentials / keys)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(redact(message))
        self.status_code = status_code


_GENERIC_ERROR_MESSAGES = {
    400: "The detection engine rejected the request.",
    404: "The detection engine does not know this camera. Sync cameras and try again.",
    409: "Camera is not assigned to an ML server. Assign it in Camera Distribution first.",
    503: "Detection engine temporarily unavailable.",
}


_INPUT_ERROR_CODES = frozenset({400, 413, 415, 422})


def public_error_message(exc: Exception) -> str:
    """Short client-facing message; technical detail stays in the server log.

    Input errors (bad image, no face, …) keep the ML node's own (redacted) explanation;
    connectivity / server failures never expose hosts, URLs or stack text.
    """
    code = int(getattr(exc, "status_code", None) or 503)
    if code in _INPUT_ERROR_CODES:
        text = redact(exc)
        if text and "://" not in text:
            return text[:300]
    return _GENERIC_ERROR_MESSAGES.get(code, "Detection engine request failed.")


# ——— Auth (optional X-ML-API-Key, see ml_services ML_API_KEY) ———


def ml_api_key() -> str:
    return str(getattr(settings, "ML_API_KEY", "") or os.environ.get("ML_API_KEY", "") or "").strip()


def ml_auth_headers() -> dict[str, str]:
    """Headers for every backend → ML request (empty when no ML_API_KEY is configured)."""
    key = ml_api_key()
    return {"X-ML-API-Key": key} if key else {}


def _auth_query() -> dict[str, str]:
    """?key= for JPEG/MJPEG URLs fetched by plain requests.get / ffmpeg (no custom headers)."""
    key = ml_api_key()
    return {"key": key} if key else {}


def _with_query(path: str, params: dict[str, str] | None = None) -> str:
    merged = dict(params or {})
    merged.update(_auth_query())
    if not merged:
        return path
    return f"{path}?{urlencode(merged)}"


def known_ml_base_urls() -> list[str]:
    """
    Every ML node the backend should control:
      - ML_SERVICE_URL (optional hub)
      - active RemoteServer rows in ML mode (ops camera distribution)

    Loopback URLs (127.0.0.1 / localhost) are allowed — local and remote ML hosts are both valid.
    """
    urls: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        url = (raw or "").strip().rstrip("/")
        if not url or url in seen:
            return
        seen.add(url)
        urls.append(url)

    _add(getattr(settings, "ML_SERVICE_URL", "") or "")
    try:
        from ops_central.models import RemoteServer

        for server in RemoteServer.objects.filter(is_active=True):
            if hasattr(server, "is_ml_mode") and not server.is_ml_mode():
                continue
            _add(server.resolved_ml_base_url() or "")
    except Exception:
        logger.debug("Could not enumerate RemoteServer ML URLs", exc_info=True)
    return urls


def ml_service_enabled() -> bool:
    """True when at least one ML endpoint is configured (hub and/or RemoteServer)."""
    return bool(known_ml_base_urls())


def _base_url() -> str:
    urls = known_ml_base_urls()
    if not urls:
        raise MLServiceError(
            "ML service is not configured. Set ML_SERVICE_URL or add an active ML RemoteServer.",
            503,
        )
    return urls[0]


def _timeout() -> int:
    return int(getattr(settings, "ML_SERVICE_TIMEOUT", 60))


def _live_timeout() -> tuple[float, float]:
    return (3.0, min(float(_timeout()), 30.0))


def _send(method: str, root: str, path: str, *, timeout, **kwargs) -> requests.Response:
    url = f"{root}{path}"
    headers = dict(kwargs.pop("headers", None) or {})
    headers.update(ml_auth_headers())
    try:
        return requests.request(method, url, timeout=timeout, headers=headers, **kwargs)
    except requests.RequestException as exc:
        logger.debug("ML request %s %s failed: %s", method, redact_url(url), redact(exc))
        raise MLServiceError(f"ML service unreachable at {redact_url(root)} ({exc})", 503) from exc


def _request(method: str, path: str, *, timeout: float | tuple[float, float] | None = None, **kwargs) -> requests.Response:
    req_timeout = _timeout() if timeout is None else timeout
    return _send(method, _base_url(), path, timeout=req_timeout, **kwargs)


def _request_at(
    base_url: str,
    method: str,
    path: str,
    *,
    timeout: float | tuple[float, float] | None = None,
    **kwargs,
) -> requests.Response:
    root = (base_url or "").strip().rstrip("/")
    if not root:
        raise MLServiceError("ML base URL is empty.", 503)
    req_timeout = _timeout() if timeout is None else timeout
    return _send(method, root, path, timeout=req_timeout, **kwargs)


# ——— Streaming multipart (large video uploads are never loaded into memory) ———


class _MultipartStream:
    """File-like multipart/form-data body. requests streams it via read() with a Content-Length."""

    _CHUNK = 256 * 1024

    def __init__(
        self,
        fields: list[tuple[str, str]] | None = None,
        files: list[tuple[str, str, str | bytes, str]] | None = None,
    ):
        self.boundary = uuid.uuid4().hex
        parts: list[bytes | Path] = []
        for name, value in fields or []:
            parts.append(
                (
                    f"--{self.boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                ).encode("utf-8")
                + str(value).encode("utf-8")
                + b"\r\n"
            )
        for name, filename, source, content_type in files or []:
            parts.append(
                (
                    f"--{self.boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8")
            )
            parts.append(source if isinstance(source, bytes) else Path(source))
            parts.append(b"\r\n")
        parts.append(f"--{self.boundary}--\r\n".encode("utf-8"))
        self._parts = parts
        self._length = sum(len(p) if isinstance(p, bytes) else p.stat().st_size for p in parts)
        self._index = 0
        self._buffer = b""
        self._fh = None

    @property
    def content_type(self) -> str:
        return f"multipart/form-data; boundary={self.boundary}"

    def __len__(self) -> int:
        return self._length

    def _fill(self, size: int) -> None:
        while len(self._buffer) < size and self._index < len(self._parts):
            part = self._parts[self._index]
            if isinstance(part, bytes):
                self._buffer += part
                self._index += 1
                continue
            if self._fh is None:
                self._fh = part.open("rb")
            chunk = self._fh.read(max(size - len(self._buffer), self._CHUNK))
            if chunk:
                self._buffer += chunk
            else:
                self._fh.close()
                self._fh = None
                self._index += 1

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._length
        self._fill(size)
        out, self._buffer = self._buffer[:size], self._buffer[size:]
        return out

    def __iter__(self) -> Iterator[bytes]:
        while True:
            chunk = self.read(self._CHUNK)
            if not chunk:
                return
            yield chunk

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def _post_multipart(
    path: str,
    *,
    fields: list[tuple[str, str]] | None = None,
    files: list[tuple[str, str, str | bytes, str]] | None = None,
    params: dict[str, str] | None = None,
    timeout: float | tuple[float, float] | None = None,
) -> requests.Response:
    body = _MultipartStream(fields=fields, files=files)
    try:
        return _request(
            "POST",
            path,
            data=body,
            params=params,
            headers={"Content-Type": body.content_type},
            timeout=timeout,
        )
    finally:
        body.close()


# ——— Health / faces / single-image endpoints ———


def ml_health() -> dict[str, Any]:
    """Health of the first reachable ML node (hub preference order)."""
    return ml_health_any()


def ml_health_any() -> dict[str, Any]:
    """Return health from the first reachable known ML node."""
    last_exc: Exception | None = None
    urls = known_ml_base_urls()
    if not urls:
        raise MLServiceError(
            "ML service is not configured. Set ML_SERVICE_URL or add an active ML RemoteServer.",
            503,
        )
    for base in urls:
        try:
            data = ml_health_at(base)
            data.setdefault("ml_base_url", base)
            return data
        except MLServiceError as exc:
            last_exc = exc
            logger.debug("ML health failed at %s: %s", redact_url(base), exc)
    raise MLServiceError(
        f"No reachable ML service among {len(urls)} configured node(s). Last error: {last_exc}",
        getattr(last_exc, "status_code", None) or 503,
    ) from last_exc


def ml_health_at(base_url: str) -> dict[str, Any]:
    res = _request_at(base_url, "GET", "/health", timeout=5.0)
    if res.status_code != 200:
        raise MLServiceError(f"ML health check failed ({res.status_code})", res.status_code)
    return res.json()


def ml_reload_faces(embeddings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Push known faces to every connected ML node."""
    return ml_reload_faces_all(embeddings)


def ml_reload_faces_all(embeddings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = [
        {"identity": entry["identity"], "embedding": entry["embedding"]}
        for entry in (embeddings or [])
    ]
    urls = known_ml_base_urls()
    if not urls:
        raise MLServiceError(
            "ML service is not configured. Set ML_SERVICE_URL or add an active ML RemoteServer.",
            503,
        )
    by_server: dict[str, Any] = {}
    ok = 0
    for base in urls:
        try:
            res = _request_at(base, "POST", "/reload/faces", json=payload)
            if res.status_code != 200:
                by_server[base] = {
                    "ok": False,
                    "error": f"HTTP {res.status_code}",
                    "detail": redact(res.text[:200]),
                }
                logger.warning("[face-sync] reload/faces failed at %s (%s)", redact_url(base), res.status_code)
                continue
            data = res.json() if res.content else {}
            by_server[base] = {"ok": True, **(data if isinstance(data, dict) else {"result": data})}
            ok += 1
        except MLServiceError as exc:
            by_server[base] = {"ok": False, "error": str(exc)}
            logger.warning("[face-sync] reload/faces unreachable at %s: %s", redact_url(base), exc)
    if ok == 0:
        raise MLServiceError(
            f"Failed to reload faces on all {len(urls)} ML node(s).",
            503,
        )
    return {
        "known_faces": len(payload),
        "db_embeddings": len(payload),
        "ok_count": ok,
        "total": len(urls),
        "by_server": by_server,
    }


def ml_extract_face_embedding(file_bytes: bytes, filename: str = "face.jpg") -> dict[str, Any]:
    """Extract embedding via the first reachable ML node."""
    last_exc: Exception | None = None
    urls = known_ml_base_urls()
    if not urls:
        raise MLServiceError(
            "ML service is not configured. Set ML_SERVICE_URL or add an active ML RemoteServer.",
            503,
        )
    for base in urls:
        try:
            res = _request_at(
                base,
                "POST",
                "/faces/extract",
                files={"image": (filename, file_bytes, "application/octet-stream")},
            )
            if res.status_code != 200:
                last_exc = MLServiceError(
                    f"Face embedding extraction failed: {res.text[:300]}",
                    res.status_code,
                )
                continue
            return res.json()
        except MLServiceError as exc:
            last_exc = exc
            logger.debug("Face extract failed at %s: %s", redact_url(base), exc)
    raise MLServiceError(
        f"Face embedding extraction failed on all ML nodes. Last error: {last_exc}",
        getattr(last_exc, "status_code", None) or 503,
    ) from last_exc


def ml_detect_image(
    file_bytes: bytes,
    filename: str = "image.jpg",
    *,
    conf: float = 0.25,
    recognize_faces: bool = True,
    timeout: float | tuple[float, float] | None = None,
) -> dict[str, Any]:
    res = _request(
        "POST",
        "/detect/image",
        files={"image": (filename, file_bytes, "application/octet-stream")},
        params={"conf": conf, "recognize_faces": str(recognize_faces).lower()},
        timeout=timeout,
    )
    if res.status_code != 200:
        detail = res.text[:300]
        raise MLServiceError(f"Detection failed: {detail}", res.status_code)
    return res.json()


def ml_validate_human_face(file_bytes: bytes, filename: str = "face.jpg") -> dict[str, Any]:
    res = _request(
        "POST",
        "/validate/human-face",
        files={"image": (filename, file_bytes, "application/octet-stream")},
    )
    if res.status_code != 200:
        detail = res.text[:300]
        raise MLServiceError(f"Face validation failed: {detail}", res.status_code)
    return res.json()


# ——— Per-camera live endpoints ———
#
# rtsp_url is never sent to the ML node: api_server._resolve_live_stream() rejects unregistered
# keys (409) and LiveStreamManager.resolve_rtsp_url() ignores a client URL for registered keys,
# so the query param only leaked NVR credentials into URLs and logs. Unknown cameras are
# (re-)registered via POST /live/register/bulk (JSON body) — see ml.camera_sync.


def camera_ml_base_url(camera) -> str:
    """Assigned ML node URL, or empty when unassigned (must not fall back to ML_SERVICE_URL)."""
    if not getattr(camera, "ml_server_id", None):
        return ""
    server = getattr(camera, "ml_server", None)
    if server is None:
        return ""
    return (server.resolved_ml_base_url() or "").strip().rstrip("/")


def require_camera_ml_url(camera) -> str:
    url = camera_ml_base_url(camera)
    if not url:
        raise MLServiceError(
            "Camera is not assigned to an ML server. Assign it in Camera Distribution first.",
            409,
        )
    return url


def ml_live_detections_at(
    base_url: str,
    stream_key: str,
    rtsp_url: str | None = None,
    *,
    purpose: str = "",
    purposes: list[str] | None = None,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    del rtsp_url  # never sent (see note above)
    key = (stream_key or "").strip()
    params: dict[str, str] = {}
    purpose_list = [str(p).strip() for p in (purposes or []) if str(p).strip()]
    if purpose_list:
        params["purposes"] = ",".join(purpose_list)
    primary = (purpose or "").strip()
    if primary:
        params["purpose"] = primary
    if include_embeddings:
        # ML strips face/ReID vectors from this endpoint unless asked (biometric data).
        params["include_embeddings"] = "1"
    res = _request_at(
        base_url,
        "GET",
        f"/live/cam/{key}/detections",
        params=params or None,
        timeout=_live_timeout(),
    )
    if res.status_code != 200:
        detail = res.text[:300]
        raise MLServiceError(
            f"Live detections failed ({res.status_code}): {detail}",
            res.status_code,
        )
    return res.json()


def ml_live_detections_for_camera(
    camera,
    *,
    purpose: str = "",
    purposes: list[str] | None = None,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    """Poll detections only on the camera's assigned ML node."""
    base = require_camera_ml_url(camera)
    return ml_live_detections_at(
        base,
        camera.stream_key,
        purpose=purpose or camera.purpose,
        purposes=purposes if purposes is not None else camera.purpose_list(),
        include_embeddings=include_embeddings,
    )


def ml_assigned_mjpeg_public_url(camera, *, kind: str = "live") -> str:
    """
    Browser MJPEG via Ops proxy to the assigned ML server.
    Empty string when unassigned (no stream until Camera Distribution assigns).
    """
    if not getattr(camera, "ml_server_id", None) or not camera.is_active or not camera.nvr_id:
        return ""
    key = (camera.stream_key or "").strip()
    if not key:
        return ""
    kind_q = "raw" if (kind or "").strip().lower() == "raw" else "live"
    return f"/api/ops/servers/{camera.ml_server_id}/mjpeg/?stream_key={key}&kind={kind_q}"


# Server-side JPEG/MJPEG URLs. Callers fetch them with plain requests.get (some outside this app),
# so the optional API key travels as ?key= — always log these through redact_url().


def ml_live_mjpeg_url_for_camera(camera) -> str:
    """Absolute MJPEG URL on the assigned ML node (server-side fetch)."""
    base = require_camera_ml_url(camera)
    return _with_query(f"{base}/live/cam/{camera.stream_key}/mjpeg")


def ml_live_mjpeg_raw_url_for_camera(camera) -> str:
    base = require_camera_ml_url(camera)
    return _with_query(f"{base}/live/cam/{camera.stream_key}/mjpeg/raw")


def ml_live_jpeg_raw_url(stream_key: str, rtsp_url: str | None = None) -> str:
    """Single-frame raw JPEG on the hub ML node (use this for snapshots; do not hang on MJPEG)."""
    del rtsp_url  # never sent
    key = (stream_key or "").strip()
    return _with_query(f"{_base_url()}/live/cam/{key}/jpeg/raw")


def ml_live_jpeg_raw_url_for_camera(camera) -> str:
    base = require_camera_ml_url(camera)
    return _with_query(f"{base}/live/cam/{camera.stream_key}/jpeg/raw")


def ml_live_jpeg_evidence_url_for_camera(camera) -> str:
    """Same-frame evidence JPEG (YOLO infer frame) — preferred for detection snapshots."""
    base = require_camera_ml_url(camera)
    return _with_query(f"{base}/live/cam/{camera.stream_key}/jpeg/evidence")


def ml_live_mjpeg_attendance_url_for_camera(camera, *, width: int = 1280) -> str:
    """HD main-stream MJPEG for attendance snapshots (same RTSP session as live AI)."""
    base = require_camera_ml_url(camera)
    width_q = str(max(640, min(4096, int(width or 3840))))
    return _with_query(f"{base}/live/cam/{camera.stream_key}/mjpeg/attendance", {"width": width_q})


# ——— Camera registry on ML nodes ———


def _build_register_payload(entries: list[dict[str, str]]) -> list[dict[str, Any]]:
    payload = []
    for item in entries:
        key = str(item.get("key") or "").strip()
        rtsp_url = str(item.get("rtsp_url") or "").strip()
        if not key or not rtsp_url:
            continue
        entry: dict[str, Any] = {"key": key, "rtsp_url": rtsp_url}
        purpose = str(item.get("purpose") or "").strip()
        if purpose:
            entry["purpose"] = purpose
        purposes = item.get("purposes")
        if isinstance(purposes, list) and purposes:
            entry["purposes"] = [str(p).strip() for p in purposes if str(p).strip()]
        elif purpose:
            entry["purposes"] = [purpose]
        payload.append(entry)
    return payload


def ml_register_cameras_bulk_at(
    base_url: str,
    entries: list[dict[str, str]],
    *,
    timeout: float | tuple[float, float] | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    payload = _build_register_payload(entries)
    if not payload and not replace:
        return {"registered": 0, "total": 0, "removed": 0}
    params = {"replace": "true"} if replace else None
    res = _request_at(
        base_url,
        "POST",
        "/live/register/bulk",
        json=payload,
        params=params,
        timeout=timeout,
    )
    if res.status_code != 200:
        raise MLServiceError(
            f"Failed to register cameras with ML service at {redact_url(base_url)}.",
            res.status_code,
        )
    return res.json()


def ml_unregister_camera_at(
    base_url: str,
    stream_key: str,
    *,
    timeout: float | tuple[float, float] | None = None,
) -> dict[str, Any]:
    key = (stream_key or "").strip()
    if not key:
        return {"removed": False}
    res = _request_at(
        base_url, "DELETE", f"/live/cam/{key}/register", timeout=timeout
    )
    if res.status_code != 200:
        raise MLServiceError(f"Failed to unregister camera {key} at {redact_url(base_url)}.", res.status_code)
    return res.json()


# ——— Video search / analyze jobs ———


def _video_search_timeout() -> int:
    return int(getattr(settings, "ML_VIDEO_SEARCH_TIMEOUT", 3600))


def _ml_is_local() -> bool:
    url = _base_url().lower()
    return "127.0.0.1" in url or "localhost" in url or "://[::1]" in url


def ml_start_video_search(
    *,
    image_path: str,
    video_path: str,
    face_threshold: float = 0.45,
    reid_threshold: float = 0.88,
    sample_fps: float = 0.0,
    clip_seconds: float = 4.0,
) -> dict[str, Any]:
    params = {
        "face_threshold": str(face_threshold),
        "reid_threshold": str(reid_threshold),
        "sample_fps": str(sample_fps),
        "clip_seconds": str(clip_seconds),
    }
    fields: list[tuple[str, str]] = []
    files: list[tuple[str, str, str | bytes, str]] = [
        ("image", "query.jpg", image_path, "application/octet-stream"),
    ]
    if _ml_is_local():
        fields.append(("video_path", video_path))
    else:
        files.append(("video", "source.mp4", video_path, "application/octet-stream"))
    res = _post_multipart(
        "/search/video",
        fields=fields,
        files=files,
        params=params,
        timeout=_video_search_timeout(),
    )
    if res.status_code != 200:
        detail = res.text[:400]
        raise MLServiceError(f"Video search failed: {detail}", res.status_code)
    return res.json()


def ml_poll_video_search(job_id: str) -> dict[str, Any]:
    res = _request("GET", f"/search/video/{job_id.strip()}", timeout=30)
    if res.status_code != 200:
        detail = res.text[:400]
        raise MLServiceError(f"Video search status failed: {detail}", res.status_code)
    return res.json()


def ml_start_video_analyze(
    *,
    video_path: str,
    person: bool = True,
    vehicle: bool = True,
    weapon: bool = True,
    fire: bool = True,
    match_staff: bool = True,
    plate: bool = False,
    sample_fps: float = 1.0,
) -> dict[str, Any]:
    fields: list[tuple[str, str]] = [
        ("person", str(bool(person)).lower()),
        ("vehicle", str(bool(vehicle)).lower()),
        ("weapon", str(bool(weapon)).lower()),
        ("fire", str(bool(fire)).lower()),
        ("match_staff", str(bool(match_staff)).lower()),
        ("plate", str(bool(plate)).lower()),
        ("sample_fps", str(sample_fps)),
    ]
    files: list[tuple[str, str, str | bytes, str]] = []
    if _ml_is_local():
        fields.append(("video_path", video_path))
    else:
        files.append(("video", "source.mp4", video_path, "application/octet-stream"))
    res = _post_multipart(
        "/analyze/video",
        fields=fields,
        files=files,
        timeout=_video_search_timeout(),
    )
    if res.status_code != 200:
        detail = res.text[:400]
        raise MLServiceError(f"Video AI test failed: {detail}", res.status_code)
    return res.json()


def ml_poll_video_analyze(job_id: str) -> dict[str, Any]:
    res = _request("GET", f"/analyze/video/{job_id.strip()}", timeout=30)
    if res.status_code != 200:
        detail = res.text[:400]
        raise MLServiceError(f"Video AI status failed: {detail}", res.status_code)
    return res.json()


def ml_download_video_analyze(job_id: str, dest: str) -> None:
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    res = _request(
        "GET",
        f"/analyze/video/{job_id.strip()}/file",
        timeout=_video_search_timeout(),
        stream=True,
    )
    if res.status_code != 200:
        detail = res.text[:400]
        raise MLServiceError(f"Tagged video download failed: {detail}", res.status_code)
    with dest_path.open("wb") as out:
        for chunk in res.iter_content(chunk_size=1024 * 256):
            if chunk:
                out.write(chunk)
