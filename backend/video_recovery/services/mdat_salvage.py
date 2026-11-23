"""Salvage video from truncated MP4/MOV files missing moov (mdat AVCC → Annex-B decode)."""

from __future__ import annotations

import mmap
import os
import re
from typing import Any

from .ffmpeg_utils import run_ffmpeg

_NAL_START = re.compile(b"\x00\x00(?:00\x01|01)")


_WRITE_CHUNK = 8 * 1024 * 1024


def _mdat_bounds(data) -> tuple[int, int] | None:
    """(start, end) of the mdat payload; handles truncated files (box extends past EOF).

    Works on bytes or an mmap, so a multi-GB upload is never read into RAM."""
    offset = 0
    file_len = len(data)
    while offset + 8 <= file_len:
        size = int.from_bytes(data[offset : offset + 4], "big")
        box_type = data[offset + 4 : offset + 8]
        header = 8
        if size == 1 and offset + 16 <= file_len:
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header = 16
        if size < header:
            break
        payload_start = offset + header
        if box_type == b"mdat":
            if size == 0 or offset + size > file_len:
                return payload_start, file_len
            return payload_start, offset + size
        if size == 0:
            break
        offset += size
    # Fallback: search for mdat marker when box structure is broken
    idx = data.find(b"mdat")
    if idx >= 8:
        return idx + 8, file_len
    return None


def extract_mdat_payload(data: bytes) -> bytes | None:
    """Return mdat box payload bytes (small in-memory inputs; see _mdat_bounds)."""
    bounds = _mdat_bounds(data)
    return data[bounds[0] : bounds[1]] if bounds else None


def find_avcc_offset(payload: bytes) -> int:
    """Locate first valid AVCC length-prefixed NAL run (skips ffmpeg metadata prefix)."""
    limit = min(len(payload), 800000)
    for start in range(0, limit - 8):
        nal_len = int.from_bytes(payload[start : start + 4], "big")
        if nal_len < 4 or nal_len > 2_000_000:
            continue
        if start + 4 + nal_len > len(payload):
            continue
        nal_type = payload[start + 4] & 0x1F
        if nal_type not in (1, 5, 6, 7, 8, 9):
            continue
        nxt = start + 4 + nal_len
        if nxt + 8 > len(payload):
            return start
        nal_len2 = int.from_bytes(payload[nxt : nxt + 4], "big")
        if nal_len2 < 4 or nxt + 4 + nal_len2 > len(payload):
            continue
        nal_type2 = payload[nxt + 4] & 0x1F
        if nal_type2 in (1, 5, 6, 7, 8, 9):
            return start
    return 0


def _looks_like_avcc_nal(data: bytes, pos: int) -> bool:
    if pos + 8 > len(data):
        return False
    nal_len = int.from_bytes(data[pos : pos + 4], "big")
    if nal_len < 4 or nal_len > 2_000_000 or pos + 4 + nal_len > len(data):
        return False
    nal_type = data[pos + 4] & 0x1F
    return nal_type in (1, 5, 6, 7, 8, 9)


def avcc_to_annex_b(payload: bytes, out=None):
    """Convert length-prefixed NAL units (MP4 mdat) to Annex-B start codes.

    With ``out`` (a binary file) the result is streamed there and the NAL count returned;
    without it the converted bytes are returned (b"" when nothing was found)."""
    start = find_avcc_offset(payload)
    data = payload[start:]
    buf = bytearray()
    i = 0
    length = len(data)
    valid_nals = 0
    max_resync_scan = 8192

    while i + 4 < length:
        if not _looks_like_avcc_nal(data, i):
            found = False
            scan_end = min(i + max_resync_scan, length - 8)
            for j in range(i + 1, scan_end):
                if _looks_like_avcc_nal(data, j):
                    i = j
                    found = True
                    break
            if not found:
                break
            continue

        nal_len = int.from_bytes(data[i : i + 4], "big")
        i += 4
        buf.extend(b"\x00\x00\x00\x01")
        buf.extend(data[i : i + nal_len])
        i += nal_len
        valid_nals += 1
        if out is not None and len(buf) >= _WRITE_CHUNK:
            out.write(buf)
            buf.clear()

    if out is not None:
        if buf:
            out.write(buf)
        return valid_nals
    if valid_nals >= 1:
        return bytes(buf)
    return b""


def scan_annex_b(payload: bytes) -> bytes:
    """Keep raw bytes if Annex-B start codes already present in mdat."""
    if _NAL_START.search(payload[:4096]):
        return payload
    return b""


def _count_start_codes(mm, start: int, end: int) -> int:
    count, pos = 0, mm.find(b"\x00\x00\x00\x01", start, end)
    while pos != -1:
        count += 1
        pos = mm.find(b"\x00\x00\x00\x01", pos + 4, end)
    return count


def _write_annex_b(mm, raw_path: str) -> tuple[int, int]:
    """Write the mdat's H.264/H.265 stream to raw_path. Returns (nal_count, bytes_written);
    nal_count is -1 when the file has no usable mdat. Views into the mmap are released on
    return, so the caller can close the mapping."""
    bounds = _mdat_bounds(mm)
    if not bounds or bounds[1] - bounds[0] < 256:
        return -1, 0
    view = memoryview(mm)
    try:
        payload = view[bounds[0] : bounds[1]]
        try:
            with open(raw_path, "wb") as fh:
                nals = avcc_to_annex_b(payload, out=fh)
            if nals < 1 and _NAL_START.search(payload[:4096]):
                # Already Annex-B: copy the payload as-is, in chunks.
                with open(raw_path, "wb") as fh:
                    for pos in range(0, len(payload), _WRITE_CHUNK):
                        fh.write(payload[pos : pos + _WRITE_CHUNK])
                nals = max(1, _count_start_codes(mm, bounds[0], bounds[1]))
            written = os.path.getsize(raw_path) if nals >= 1 else 0
            return nals, written
        finally:
            payload.release()
    finally:
        view.release()


def salvage_mdat_stream(src_path: str, work_dir: str) -> dict[str, Any]:
    """
    Extract decodable H.264 from truncated MP4 when moov/index is missing.
    Re-encodes to a playable MP4.
    """
    os.makedirs(work_dir, exist_ok=True)
    raw_h264 = os.path.join(work_dir, "salvaged.h264")
    out_mp4 = os.path.join(work_dir, "salvaged.mp4")
    if os.path.getsize(src_path) == 0:
        nal_count, written = -1, 0
    else:
        # Memory-mapped: the OS pages the file in as needed instead of loading it all.
        with open(src_path, "rb") as src, mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            nal_count, written = _write_annex_b(mm, raw_h264)
    if nal_count < 0:
        return {
            "success": False,
            "method": "mdat_salvage",
            "error": "No mdat payload found in file",
        }
    if nal_count < 1 or written < 128:
        return {
            "success": False,
            "method": "mdat_salvage",
            "error": "Could not extract NAL units from mdat (file may be too severely truncated)",
        }

    proc = None
    for attempt, extra in enumerate(
        (
            ["-f", "h264"],
            ["-f", "hevc"],
            [],
        )
    ):
        cmd = [
            "-y",
            "-err_detect",
            "ignore_err",
            *extra,
            "-i",
            raw_h264,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_mp4,
        ]
        proc = run_ffmpeg(cmd, timeout=600)
        if proc.returncode == 0 and os.path.isfile(out_mp4) and os.path.getsize(out_mp4) > 0:
            return {
                "success": True,
                "method": "mdat_salvage",
                "output_path": out_mp4,
                "header_rebuilt": True,
                "index_rebuilt": True,
                "attempt": attempt + 1,
                "nal_units_extracted": nal_count,
            }

    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")[:400] if proc else ""
    return {
        "success": False,
        "method": "mdat_salvage",
        "error": stderr or "mdat salvage encode failed",
    }
