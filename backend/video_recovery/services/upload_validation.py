"""Upload validation — signature, format, size, basic integrity."""

from __future__ import annotations

import os
from typing import Any

from django.conf import settings

from .forensic_analyzer import detect_file_signature, run_forensic_analysis

ALLOWED_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
MIN_BYTES = 512
DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
MAX_BYTES = DEFAULT_MAX_BYTES  # kept for callers; max_upload_bytes() reads the setting


def max_upload_bytes() -> int:
    """VIDEO_RECOVERY_MAX_UPLOAD_BYTES (default 2 GB)."""
    try:
        value = int(getattr(settings, "VIDEO_RECOVERY_MAX_UPLOAD_BYTES", DEFAULT_MAX_BYTES) or DEFAULT_MAX_BYTES)
    except (TypeError, ValueError):
        value = DEFAULT_MAX_BYTES
    return max(MIN_BYTES, value)


def header_format(header: bytes) -> str | None:
    """Container named by the first bytes (same signatures the pipeline accepts), else None."""
    if len(header) >= 8 and header[4:8] == b"ftyp":
        return "mp4"
    if header[:4] == b"\x1a\x45\xdf\xa3":
        return "mkv"
    if header[:4] == b"RIFF" and b"AVI" in header[8:16]:
        return "avi"
    return None


def precheck_upload(upload) -> list[str]:
    """Cheap checks on an incoming upload before anything is written to disk:
    extension whitelist, size limits and container magic bytes. Returns error messages."""
    errors: list[str] = []
    name = getattr(upload, "name", "") or ""
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        errors.append(f"Unsupported file type '{ext or 'none'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
    size = int(getattr(upload, "size", 0) or 0)
    if size < MIN_BYTES:
        errors.append(f"File too small ({size} bytes).")
    if size > max_upload_bytes():
        errors.append(f"File exceeds the {max_upload_bytes() // (1024 * 1024)} MB limit.")
    try:
        upload.seek(0)
        head = upload.read(32)
        upload.seek(0)
    except Exception:
        head = b""
    if not errors and header_format(head or b"") is None:
        errors.append("Not a recognised video container (MP4/MOV, MKV/WebM or AVI).")
    return errors


def validate_upload(path: str, filename: str = "") -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not os.path.isfile(path):
        return {"valid": False, "errors": ["File not found"], "warnings": []}

    size = os.path.getsize(path)
    if size < MIN_BYTES:
        errors.append(f"File too small ({size} bytes)")
    if size > max_upload_bytes():
        errors.append(f"File exceeds maximum size ({size} bytes)")

    ext = os.path.splitext(filename or path)[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        warnings.append(f"Unusual extension '{ext}' — proceeding with content inspection")

    signature = detect_file_signature(path)
    if signature.get("primary_format") == "unknown":
        errors.append("Unrecognized video file signature")

    forensic = run_forensic_analysis(path)
    corruption = forensic.get("corruption_detection") or {}
    if corruption.get("corruption_detected"):
        warnings.extend(corruption.get("issues") or [])

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings[:8],
        "file_signature": signature,
        "size_bytes": size,
        "format": signature.get("primary_format"),
    }
