"""Freeze / stream continuity helpers."""

from __future__ import annotations

from typing import Any

from .metrics import fingerprints_identical

# Consecutive identical health samples needed before a fingerprint-only freeze is "critical".
FREEZE_MIN_IDENTICAL_STREAK = 3
# Score for one identical pair when no streak / frame counter is known: suspicious, not critical.
_SINGLE_IDENTICAL_SCORE = 0.7
# Ceiling for frames that differ (or whose counter advanced): a static scene is not a freeze.
_MOVING_CEILING = 0.5


def identical_streak(
    diff: float,
    previous_streak: int | None = None,
) -> int:
    """Consecutive samples (including this one) whose fingerprint matched the previous one."""
    if not fingerprints_identical(diff):
        return 0
    return max(0, int(previous_streak or 0)) + 1


def freeze_score(
    diff: float,
    fps: float | None = None,
    *,
    frame_seq: int | None = None,
    previous_frame_seq: int | None = None,
    frame_ts: float | None = None,
    previous_frame_ts: float | None = None,
    streak: int | None = None,
    min_streak: int = FREEZE_MIN_IDENTICAL_STREAK,
) -> float:
    """
    Higher = more frozen (>= 0.85 is reported as a frozen stream).

    Evidence, strongest first:
      1. decoder frame counter / timestamp did not advance between samples → 1.0;
         it did advance → at most 0.5 (new frames are arriving);
      2. identical fingerprints for ``min_streak`` consecutive samples → 1.0;
         a single identical pair (no history) → 0.7 (possible freeze, not critical);
      3. otherwise ≤ 0.5, scaled by similarity (a quiet scene is not a freeze).

    ``fps`` is accepted for compatibility but low FPS alone never implies a freeze — that is
    reported separately as ``low_fps``.
    """
    del fps  # low fps is its own alert, never a freeze signal
    similarity = max(0.0, min(1.0, 1.0 - float(diff)))
    moving = round(_MOVING_CEILING * similarity, 4)

    if frame_seq is not None and previous_frame_seq is not None:
        return 1.0 if int(frame_seq) <= int(previous_frame_seq) else moving
    if frame_ts is not None and previous_frame_ts is not None and float(previous_frame_ts) > 0:
        return 1.0 if float(frame_ts) <= float(previous_frame_ts) else moving

    if not fingerprints_identical(diff):
        return moving
    run = int(streak or 1)
    if run >= max(1, int(min_streak)):
        return 1.0
    # Identical, but not for long enough yet: climbing, still below the 0.85 critical bar.
    return round(min(0.8, _SINGLE_IDENTICAL_SCORE + 0.1 * (run - 1)), 4)


def stream_metrics(
    *,
    rtsp_available: bool,
    fps: float | None,
    frame_age_sec: float | None,
    dropped_frames: int | None,
    freeze: float,
    identical_streak: int | None = None,
) -> dict[str, Any]:
    latency = None if frame_age_sec is None else round(float(frame_age_sec), 3)
    out = {
        "rtsp_available": bool(rtsp_available),
        "fps": None if fps is None else round(float(fps), 2),
        "dropped_frames": dropped_frames,
        "freeze_score": freeze,
        "stream_latency": latency,
        "frame_arrival_delay": latency,
    }
    if identical_streak is not None:
        out["identical_streak"] = int(identical_streak)
    return out
