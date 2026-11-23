"""Serve /media/ only to logged-in users (token header or auth cookie)."""

from __future__ import annotations

import logging
import mimetypes
import os
import posixpath
import re

from django.conf import settings
from django.db.models import Q
from django.core.exceptions import SuspiciousFileOperation
from django.http import Http404, HttpResponse, HttpResponseRedirect, StreamingHttpResponse
from django.utils._os import safe_join
from django.views.static import serve
from urllib.parse import quote

MEDIA_AUTH_COOKIE = "tekeye_auth_token"

logger = logging.getLogger(__name__)


def attach_media_auth_cookie(response: HttpResponse, token_key: str, request=None) -> HttpResponse:
    """Allow same-origin <img> / new-tab requests to send the API token.

    HttpOnly: page scripts cannot read it, so an injected script cannot steal it.
    """
    response.set_cookie(
        MEDIA_AUTH_COOKIE,
        token_key,
        httponly=True,
        samesite="Lax",
        path="/",
        secure=bool(request and request.is_secure()),
    )
    return response


def clear_media_auth_cookie(response: HttpResponse) -> HttpResponse:
    response.delete_cookie(MEDIA_AUTH_COOKIE, path="/", samesite="Lax")
    return response


def _token_key_from_request(request) -> str:
    auth = (request.META.get("HTTP_AUTHORIZATION") or "").strip()
    if auth.lower().startswith("token "):
        return auth[6:].strip()
    return (request.COOKIES.get(MEDIA_AUTH_COOKIE) or "").strip()


def media_user_from_request(request):
    key = _token_key_from_request(request)
    if not key:
        return None
    from users.token_auth import user_for_token_key

    return user_for_token_key(key)


def _safe_media_next(path: str) -> str | None:
    value = (path or "").strip()
    if not value.startswith("/media/"):
        return None
    if value.startswith("//") or "\\" in value or "://" in value:
        return None
    return value


def _login_redirect(request) -> HttpResponseRedirect:
    next_path = _safe_media_next(request.get_full_path() or "") or "/media/"
    from django.conf import settings

    frontend = getattr(settings, "FRONTEND_ORIGIN", "").rstrip("/") or ""
    login_base = f"{frontend}/login" if frontend else "/login"
    return HttpResponseRedirect(login_base + "?next=" + quote(next_path, safe="/"))


def _unauthorized(request) -> HttpResponse:
    accept = (request.headers.get("Accept") or "").lower()
    if "text/html" in accept:
        return _login_redirect(request)
    return HttpResponse("Authentication required", status=401, content_type="text/plain")


# Staff documents (CNIC, contracts, CVs) and face-training images: only for the
# people who may use the HR / recognition APIs that hand out these URLs.
HR_ONLY_MEDIA_PREFIXES = ("staff_docs/", "dataset/", "face_samples/")


def _same_file_path(a: str, b: str) -> bool:
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except (OSError, ValueError):
        return False


def _hq_media_source_path(path: str) -> str | None:
    """source_path of the HqMediaObject stored at exactly this media path, or None."""
    from city_sync.models import HqMediaObject

    try:
        full = safe_join(settings.MEDIA_ROOT, path)
    except (SuspiciousFileOperation, ValueError):
        return None
    name = posixpath.basename(path.replace("\\", "/"))
    if not name:
        return None
    # Basename narrows the search; the full stored path must then match exactly
    # (m1.jpg from Kohat is a different file from m1.jpg from Mardan).
    for storage_path, source_path in HqMediaObject.objects.filter(storage_path__iendswith=name).values_list(
        "storage_path", "source_path"
    )[:50]:
        if storage_path and _same_file_path(storage_path, full):
            return source_path or ""
    return None


def _video_recovery_allowed(user, clean: str, path: str) -> bool:
    """Evidence footage: the job's uploader, or forensic roles (Super Admin, superuser,
    investigation officers). Working files are never served."""
    if clean.startswith("video_recovery/work/"):
        return False
    from video_recovery.permissions import can_see_all_jobs
    from video_recovery.models import VideoRecoveryJob

    if can_see_all_jobs(user):
        return True
    rel = posixpath.normpath(path.replace("\\", "/")).lstrip("/")
    match = Q(original_file__iexact=rel) | Q(recovered_file__iexact=rel)
    return VideoRecoveryJob.objects.filter(match, uploaded_by=user).exists()


def _support_attachment_allowed(user, path: str) -> bool:
    """Chat attachment: only people who can see the ticket; internal notes only for handlers."""
    from django.http import QueryDict

    from requests_support.models import TicketComment
    from requests_support.permissions import can_access_support_module, is_handler_user, is_support_user
    from requests_support.views import SupportTicketViewSet

    if not can_access_support_module(user):
        return False
    rel = posixpath.normpath(path.replace("\\", "/")).lstrip("/")
    comment = TicketComment.objects.filter(attachment__iexact=rel).select_related("ticket").first()
    if comment is None:
        return False
    if comment.is_internal and not (is_support_user(user) or is_handler_user(user)):
        return False

    class _Req:  # what SupportTicketViewSet.get_queryset reads
        pass

    req = _Req()
    req.user = user
    req.query_params = QueryDict("")
    view = SupportTicketViewSet()
    view.request = req
    view.action = "retrieve"
    view.kwargs = {}
    view.format_kwarg = None
    return view.get_queryset().filter(pk=comment.ticket_id).exists()


def media_path_allowed(user, path: str) -> bool:
    from users.permissions import has_hr_api_access

    # Same normalising as django.views.static.serve, so "a/../staff_docs/x" is caught.
    clean = posixpath.normpath(path.replace("\\", "/")).lstrip("/").lower()
    if clean.startswith("hq_sync_media/"):
        # HQ keeps city files by date; the city's own path tells what the file is.
        source = _hq_media_source_path(path)
        if source is None:
            return False  # not a file HQ received through sync: deny
        source = "/" + source.replace("\\", "/").lower()
        if any("/" + prefix in source for prefix in HR_ONLY_MEDIA_PREFIXES):
            return has_hr_api_access(user)
        # City copies of restricted files: the owning job / ticket lives on the city server,
        # so only the roles that may see every such file get them here.
        if "/video_recovery/" in source:
            from video_recovery.permissions import can_see_all_jobs

            return "/video_recovery/work/" not in source and can_see_all_jobs(user)
        if "/support_chat/" in source or source.endswith("/recording_manifest.json"):
            return False
        return True
    if clean.startswith(HR_ONLY_MEDIA_PREFIXES):
        return has_hr_api_access(user)
    if clean.startswith("distribution/") and posixpath.basename(clean) == "recording_manifest.json":
        return False  # older manifests embed RTSP URLs with NVR passwords
    try:
        if clean.startswith("video_recovery/"):
            return _video_recovery_allowed(user, clean, path)
        if clean.startswith("support_chat/"):
            return _support_attachment_allowed(user, path)
    except Exception:
        logger.exception("Media permission check failed for %s", clean)
        return False
    return True


_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def _iter_file_range(fh, start: int, length: int, chunk: int = 64 * 1024):
    try:
        fh.seek(start)
        left = length
        while left > 0:
            data = fh.read(min(chunk, left))
            if not data:
                break
            left -= len(data)
            yield data
    finally:
        fh.close()


def _range_response(request, path: str) -> HttpResponse | None:
    """206 partial content for a single "Range: bytes=a-b" (video seeking). None = serve normally."""
    match = _RANGE_RE.match((request.META.get("HTTP_RANGE") or "").strip())
    if not match:
        return None
    full = safe_join(settings.MEDIA_ROOT, path)
    if not os.path.isfile(full):
        return None
    size = os.path.getsize(full)
    first, last = match.groups()
    if first == "" and last == "":
        return None
    if first == "":
        start, end = max(0, size - int(last)), size - 1  # "bytes=-500" = last 500 bytes
    else:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        response = HttpResponse(status=416)
        response["Content-Range"] = f"bytes */{size}"
        return response
    length = end - start + 1
    content_type = mimetypes.guess_type(full)[0] or "application/octet-stream"
    if request.method == "HEAD":
        response = HttpResponse(status=206, content_type=content_type)
    else:
        response = StreamingHttpResponse(
            _iter_file_range(open(full, "rb"), start, length), status=206, content_type=content_type
        )
    response["Content-Range"] = f"bytes {start}-{end}/{size}"
    response["Content-Length"] = str(length)
    return response


# Shown in the page (img / video / audio tags). Everything else downloads: an HTML,
# SVG or XML file served inline from our origin could run script as the viewer.
_INLINE_MEDIA_TYPES = ("image/", "video/", "audio/")
_NEVER_INLINE_EXTENSIONS = (".svg", ".svgz", ".html", ".htm", ".xhtml", ".xml", ".js", ".mjs")


def _serve_inline(path: str, content_type: str) -> bool:
    lower = path.lower()
    if lower.endswith(_NEVER_INLINE_EXTENSIONS):
        return False
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype == "image/svg+xml":
        return False
    if ctype.startswith(_INLINE_MEDIA_TYPES):
        return True
    # Optional: HR previews staff PDFs in an <iframe>. Off by default (PDFs download).
    return ctype == "application/pdf" and bool(getattr(settings, "MEDIA_INLINE_PDF", False))


def _harden_media_response(response, path: str):
    ctype = response.get("Content-Type", "") or mimetypes.guess_type(path)[0] or "application/octet-stream"
    response["X-Content-Type-Options"] = "nosniff"
    inline = _serve_inline(path, ctype)
    filename = posixpath.basename(path.replace("\\", "/")) or "download"
    quoted = quote(filename)
    if inline:
        response["Content-Disposition"] = f"inline; filename*=UTF-8''{quoted}"
    else:
        response["Content-Disposition"] = f"attachment; filename*=UTF-8''{quoted}"
    is_pdf = ctype.split(";")[0].strip().lower() == "application/pdf"
    if not (inline and is_pdf):
        # Even if a browser renders it, it runs in an opaque origin with no script.
        # (Chrome's PDF viewer refuses sandboxed documents, so an inline PDF skips this.)
        response["Content-Security-Policy"] = "sandbox"
    return response


def protected_media(request, path: str):
    if request.method not in ("GET", "HEAD"):
        return HttpResponse(status=405)
    user = media_user_from_request(request)
    if user is None:
        return _unauthorized(request)
    if not path or path.endswith("/"):
        raise Http404("Not found")
    if not media_path_allowed(user, path):
        return HttpResponse("Not allowed", status=403, content_type="text/plain")
    try:
        response = _range_response(request, path) or serve(request, path, document_root=settings.MEDIA_ROOT)
    except SuspiciousFileOperation:
        raise Http404("Not found")
    _harden_media_response(response, path)
    response["Accept-Ranges"] = "bytes"
    response["Cache-Control"] = "private, no-store"
    # PDFs are previewed in an <iframe> on our own pages; other sites still cannot frame them.
    response["X-Frame-Options"] = "SAMEORIGIN"
    return response
