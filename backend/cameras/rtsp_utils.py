"""Build RTSP URLs dynamically from NVR credentials + channel (never stored in DB)."""

from __future__ import annotations

from urllib.parse import quote

from .models import Nvr, NvrBrand


def _encode_credentials(username: str, password: str) -> tuple[str, str]:
    return quote(username or "admin", safe=""), quote(password or "", safe="")


def nvr_stream_path(nvr: Nvr, channel: int) -> str:
    """
    Return the RTSP path for a channel on the given NVR.
    channel may be a logical channel (1, 2, 3) or Hikvision stream ID (101, 201).
    """
    ch = int(channel)
    template = (nvr.stream_path_template or "").strip()
    if template:
        try:
            return template.format(channel=ch)
        except (KeyError, IndexError, ValueError, AttributeError):
            # Bad template saved before validation existed — fall back to the brand default.
            pass

    brand = nvr.brand

    if brand == NvrBrand.HIKVISION:
        stream_id = ch if ch >= 100 else ch * 100 + 1
        return f"/Streaming/Channels/{stream_id}"

    if brand == NvrBrand.DAHUA:
        logical = ch if ch <= 64 else ch
        return f"/cam/realmonitor?channel={logical}&subtype=0"

    if brand == NvrBrand.UNIVIEW:
        logical = ch if ch <= 64 else ch
        return f"/media/video{logical}"

    stream_id = ch if ch >= 100 else ch * 100 + 1
    return f"/Streaming/Channels/{stream_id}"


def build_rtsp_url_from_nvr(nvr: Nvr, channel: int) -> str:
    """Construct full RTSP URL at runtime from NVR record + channel number."""
    ip = (nvr.ip_address or "").strip()
    if not ip:
        return ""
    user_enc, pass_enc = _encode_credentials(nvr.username, nvr.password)
    port = int(nvr.port or 554)
    path = nvr_stream_path(nvr, channel)
    if not path.startswith("/"):
        path = f"/{path}"
    return f"rtsp://{user_enc}:{pass_enc}@{ip}:{port}{path}"


def validate_stream_path_template(template: str) -> str:
    """Raise ValueError unless the template formats with only {channel} into a URL path."""
    text = (template or "").strip()
    if not text:
        return ""
    if any(ch.isspace() for ch in text) or "@" in text or "://" in text:
        raise ValueError("Path template must be a URL path (no spaces, credentials or scheme).")
    try:
        rendered = text.format(channel=1)
    except (KeyError, IndexError, ValueError, AttributeError):
        raise ValueError("Path template may only use the {channel} placeholder.") from None
    if not rendered.startswith("/"):
        text = f"/{text}"
    return text


def build_rtsp_url_for_preview(nvr: Nvr, channel: int) -> str:
    return build_rtsp_url_from_nvr(nvr, channel)
