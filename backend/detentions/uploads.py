"""Upload whitelist for detention / seizure documents (extension + MIME type + size cap)."""

from __future__ import annotations

import os

from django.conf import settings
from rest_framework.exceptions import ValidationError

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})
VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp", ".mpeg", ".mpg"})
PDF_EXTS = frozenset({".pdf"})
OFFICE_EXTS = frozenset(
    {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".txt", ".csv", ".rtf"}
)

KIND_IMAGE = "image"
KIND_VIDEO = "video"
KIND_PDF = "pdf"
KIND_DOCUMENT = "document"  # pdf, office docs, images (scans)
KIND_ANY = "any"  # image, video, pdf, office

_EXTS_BY_KIND = {
    KIND_IMAGE: IMAGE_EXTS,
    KIND_VIDEO: VIDEO_EXTS,
    KIND_PDF: PDF_EXTS,
    KIND_DOCUMENT: PDF_EXTS | OFFICE_EXTS | IMAGE_EXTS,
    KIND_ANY: IMAGE_EXTS | VIDEO_EXTS | PDF_EXTS | OFFICE_EXTS,
}

_OFFICE_MIMES = frozenset(
    {
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/rtf",
        "text/rtf",
        "text/plain",
        "text/csv",
        "application/csv",
        "application/vnd.ms-office",
        "application/zip",  # some browsers label docx/xlsx as zip
        "application/x-zip-compressed",
    }
)

# Browsers sometimes send these for valid files (e.g. .mkv, .heic, .docx on Windows).
_GENERIC_MIMES = frozenset({"", "application/octet-stream", "binary/octet-stream"})


def _limits() -> dict[str, int]:
    mb = 1024 * 1024
    return {
        KIND_IMAGE: int(getattr(settings, "SEIZURE_UPLOAD_MAX_IMAGE_BYTES", 15 * mb)),
        KIND_VIDEO: int(getattr(settings, "SEIZURE_UPLOAD_MAX_VIDEO_BYTES", 300 * mb)),
        "doc": int(getattr(settings, "SEIZURE_UPLOAD_MAX_DOCUMENT_BYTES", 25 * mb)),
    }


def _ext(name: str) -> str:
    return os.path.splitext((name or "").replace("\\", "/"))[1].lower()


def _category(ext: str) -> str:
    if ext in IMAGE_EXTS:
        return KIND_IMAGE
    if ext in VIDEO_EXTS:
        return KIND_VIDEO
    if ext in PDF_EXTS:
        return KIND_PDF
    return KIND_DOCUMENT


def _mime_ok(category: str, mime: str) -> bool:
    mime = (mime or "").split(";")[0].strip().lower()
    if mime in _GENERIC_MIMES:
        return True
    if category == KIND_IMAGE:
        return mime.startswith("image/")
    if category == KIND_VIDEO:
        return mime.startswith("video/") or mime == "application/x-matroska"
    if category == KIND_PDF:
        return mime in ("application/pdf", "application/x-pdf")
    return mime in _OFFICE_MIMES


def _looks_like_pdf(uploaded) -> bool:
    try:
        pos = uploaded.tell()
    except Exception:  # noqa: BLE001
        pos = None
    try:
        uploaded.seek(0)
        head = uploaded.read(1024) or b""
        return b"%PDF" in head
    except Exception:  # noqa: BLE001
        return False
    finally:
        try:
            uploaded.seek(pos or 0)
        except Exception:  # noqa: BLE001
            pass


def validate_upload(uploaded, kind: str = KIND_ANY, *, field: str = "file") -> str:
    """Raise ValidationError unless ``uploaded`` is an allowed type/size. Returns its category."""
    name = getattr(uploaded, "name", "") or ""
    ext = _ext(name)
    allowed = _EXTS_BY_KIND.get(kind, _EXTS_BY_KIND[KIND_ANY])
    if ext not in allowed:
        raise ValidationError(
            {field: f"File type '{ext or 'unknown'}' is not allowed for {name or 'this upload'}."}
        )
    category = _category(ext)
    mime = getattr(uploaded, "content_type", "") or ""
    if not _mime_ok(category, mime):
        raise ValidationError({field: f"{name}: content type '{mime}' does not match a {category} file."})
    limits = _limits()
    cap = limits.get(category, limits["doc"])
    size = int(getattr(uploaded, "size", 0) or 0)
    if size <= 0:
        raise ValidationError({field: f"{name} is empty."})
    if size > cap:
        raise ValidationError(
            {field: f"{name} is too large ({size // (1024 * 1024)} MB; limit {cap // (1024 * 1024)} MB)."}
        )
    if category == KIND_PDF and not _looks_like_pdf(uploaded):
        raise ValidationError({field: f"{name} is not a valid PDF file."})
    if category == KIND_IMAGE and not _is_readable_image(uploaded):
        raise ValidationError({field: f"{name} is not a valid image file."})
    return category


def _is_readable_image(uploaded) -> bool:
    from PIL import Image

    try:
        uploaded.seek(0)
        with Image.open(uploaded) as img:
            img.verify()
        return True
    except Exception:  # noqa: BLE001 — includes DecompressionBombError
        return False
    finally:
        try:
            uploaded.seek(0)
        except Exception:  # noqa: BLE001
            pass


def validate_uploads(files, kind: str = KIND_ANY, *, field: str = "file") -> None:
    for f in files:
        validate_upload(f, kind, field=field)
