"""Lighting helpers for InsightFace enrollment and live probes."""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe_bgr(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, green, red = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    merged = cv2.merge((clahe.apply(lightness), green, red))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    gamma = float(np.clip(gamma, 0.2, 4.0))
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def lighting_variants(image: np.ndarray) -> list[np.ndarray]:
    """Original photo plus gamma and CLAHE copies used at enrollment."""
    return [image, _gamma(image, 0.7), _gamma(image, 1.4), apply_clahe_bgr(image)]


def correct_exposure(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mean = float(gray.mean())
    if mean < 1.0 or mean > 254.0:
        return image
    gamma = float(np.log(0.5) / np.log(mean / 255.0))
    return _gamma(image, gamma)


def screen_moire_score(image: np.ndarray) -> float:
    """High-frequency energy. A phone or monitor replay scores higher than a live face."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, (64, 64))
    spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    magnitude = np.abs(spectrum)
    cy, cx = 32, 32
    magnitude[cy - 4 : cy + 5, cx - 4 : cx + 5] = 0
    peak = float(magnitude.max()) + 1e-6
    return float(np.clip(magnitude.mean() / peak, 0.0, 1.0))


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    if inter <= 0:
        return 0.0
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def closest_face(faces, bbox, *, min_iou: float = 0.3):
    """
    The face in ``faces`` that is the same person as the one at ``bbox`` (same image coordinates).

    Picks the highest IoU with the original box; returns None when nothing overlaps enough, so a
    re-detection never silently switches to a different person in a crowded frame.
    """
    if not faces:
        return None
    if bbox is None:
        return max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
    best, best_iou = None, 0.0
    for face in faces:
        fb = getattr(face, "bbox", None)
        if fb is None:
            continue
        iou = _box_iou(fb, bbox)
        if iou > best_iou:
            best, best_iou = face, iou
    return best if best_iou >= min_iou else None
