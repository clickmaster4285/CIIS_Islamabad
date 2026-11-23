"""Main camera health analyzer — V1."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .freeze import freeze_score, identical_streak, stream_metrics
from .metrics import compute_image_metrics, frame_diff_score
from .recommendations import build_recommendations
from .roi import apply_roi
from .scoring import score_health
from .visibility import evaluate_visibility


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def analyze_camera_health(
    frame: np.ndarray | None,
    *,
    purposes: list[str] | None = None,
    roi: dict[str, Any] | None = None,
    detections: list[dict[str, Any]] | None = None,
    previous_fingerprint: list[float] | None = None,
    rtsp_available: bool = True,
    fps: float | None = None,
    frame_age_sec: float | None = None,
    dropped_frames: int | None = None,
    frame_seq: int | None = None,
    previous_frame_seq: int | None = None,
    frame_ts: float | None = None,
    previous_frame_ts: float | None = None,
    previous_identical_streak: int | None = None,
) -> dict[str, Any]:
    """Freeze evidence: decoder ``frame_seq`` / ``frame_ts`` vs the previous sample when given;
    otherwise the fingerprint. Feed ``stream.identical_streak`` from the previous result back as
    ``previous_identical_streak`` so a freeze is only critical after several identical samples."""
    purposes = [str(p).strip().lower() for p in (purposes or []) if str(p).strip()]

    if frame is None or getattr(frame, "size", 0) == 0:
        stream = stream_metrics(
            rtsp_available=False,
            fps=fps,
            frame_age_sec=frame_age_sec,
            dropped_frames=dropped_frames,
            freeze=1.0,
        )
        image: dict[str, Any] = {
            "sharpness": 0,
            "brightness": 0,
            "contrast": 0,
            "dark_ratio": 1.0,
            "bright_ratio": 0.0,
            "overexposure": 0.0,
            "noise": 0,
            "glare": 0,
            "color_abnormality": 0,
            "resolution": {"width": 0, "height": 0, "megapixels": 0},
            "fingerprint": [],
        }
        visibility = evaluate_visibility([], purposes)
        scores = score_health(image, stream, visibility, purposes)
        alerts = build_recommendations(image, stream, visibility, scores)
        return {
            "image": image,
            "stream": stream,
            "visibility": visibility,
            "scores": scores,
            "alerts": alerts,
            "roi_applied": None,
            "overall_score": scores["overall_score"],
            "status": scores["status"],
        }

    cropped, roi_box = apply_roi(frame, roi)
    image = compute_image_metrics(cropped)
    diff = frame_diff_score(image.get("fingerprint") or [], previous_fingerprint)
    streak = identical_streak(diff, previous_identical_streak)
    freeze = freeze_score(
        diff,
        fps=fps,
        frame_seq=frame_seq,
        previous_frame_seq=previous_frame_seq,
        frame_ts=frame_ts,
        previous_frame_ts=previous_frame_ts,
        streak=streak,
    )
    stream = stream_metrics(
        rtsp_available=bool(rtsp_available),
        fps=fps,
        frame_age_sec=frame_age_sec,
        dropped_frames=dropped_frames,
        freeze=freeze,
        identical_streak=streak,
    )
    visibility = evaluate_visibility(detections, purposes)
    scores = score_health(image, stream, visibility, purposes)
    alerts = build_recommendations(image, stream, visibility, scores)

    # Keep fingerprint for storage but avoid huge payloads in API responses if needed
    return {
        "image": image,
        "stream": stream,
        "visibility": visibility,
        "scores": scores,
        "alerts": alerts,
        "roi_applied": list(roi_box) if roi_box else None,
        "overall_score": scores["overall_score"],
        "status": scores["status"],
        "frame_diff": round(diff, 4),
    }
