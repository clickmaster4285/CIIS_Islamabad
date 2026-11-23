"""FFmpeg / FFprobe helpers for the video recovery module."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any


def ffmpeg_path() -> str:
    from django.conf import settings

    custom = getattr(settings, "FFMPEG_PATH", "").strip()
    if custom and os.path.isfile(custom):
        return custom
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError(
        "ffmpeg not found. Set FFMPEG_PATH in backend/.env or install ffmpeg."
    )


def ffprobe_path() -> str:
    ffmpeg = ffmpeg_path()
    if ffmpeg.lower().endswith("ffmpeg.exe"):
        probe = ffmpeg[:-10] + "ffprobe.exe"
        if os.path.isfile(probe):
            return probe
    if ffmpeg.lower().endswith("ffmpeg"):
        probe = ffmpeg[:-6] + "ffprobe"
        if os.path.isfile(probe):
            return probe
    found = shutil.which("ffprobe")
    if found:
        return found
    raise FileNotFoundError(
        "ffprobe not found. Install ffprobe alongside ffmpeg or set FFMPEG_PATH."
    )


def ffmpeg_available() -> bool:
    try:
        ffmpeg_path()
        ffprobe_path()
        return True
    except FileNotFoundError:
        return False


# Inputs here are uploaded, possibly hostile files. Only local files may be opened (no
# http/tcp/etc.), and only plain media demuxers may parse them: no HLS / DASH / concat
# playlists that could make ffmpeg read other files or URLs named inside the upload.
PROTOCOL_WHITELIST = "file"
FORMAT_WHITELIST = (
    "mov,matroska,avi,flv,mpegts,mpeg,asf,h264,hevc,"
    "image2,jpeg_pipe,png_pipe,wav,aac,mp3,ogg"
)
# Our own frame lists (video_reconstruction) are the only concat inputs.
CONCAT_FORMAT_WHITELIST = FORMAT_WHITELIST + ",concat"


def _input_guard(forced_format: str = "") -> list[str]:
    formats = CONCAT_FORMAT_WHITELIST if forced_format == "concat" else FORMAT_WHITELIST
    return ["-protocol_whitelist", PROTOCOL_WHITELIST, "-format_whitelist", formats]


def guard_inputs(cmd_args: list[str]) -> list[str]:
    """Insert the protocol/demuxer whitelist in front of every ``-i`` input."""
    out: list[str] = []
    forced = ""
    i = 0
    while i < len(cmd_args):
        arg = cmd_args[i]
        if arg == "-f" and i + 1 < len(cmd_args):
            forced = str(cmd_args[i + 1])
        if arg == "-i":
            out.extend(_input_guard(forced))
            forced = ""
        out.append(arg)
        i += 1
    return out


def run_ffprobe(path: str) -> dict[str, Any]:
    cmd = [
        ffprobe_path(),
        "-v",
        "quiet",
        *_input_guard(),
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "ffprobe failed").strip()}
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {"error": "Invalid ffprobe JSON output"}


def run_ffmpeg(cmd_args: list[str], *, timeout: int = 600) -> subprocess.CompletedProcess:
    cmd = [ffmpeg_path(), "-nostdin", "-hide_banner", "-loglevel", "error", *guard_inputs(list(cmd_args))]
    return subprocess.run(cmd, capture_output=True, timeout=timeout)
