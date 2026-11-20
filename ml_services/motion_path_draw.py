"""Draw an object's movement path: line from where it started (A) to where it is, arrow at the end.

Shared by the live overlay and the movement clip videos so both look the same.
Path points are [x, y, t] in source-frame pixels (t = epoch seconds).
"""

from __future__ import annotations

import cv2
import numpy as np

PATH_COLOR = (0, 230, 255)  # yellow (BGR)
START_COLOR = (0, 160, 255)  # orange "A"
OUTLINE = (0, 0, 0)


def draw_motion_path(
    img: np.ndarray,
    path: list,
    thickness: int = 3,
    *,
    scale: float = 1.0,
    until: float | None = None,
) -> np.ndarray:
    """Draw in place (and return img). `until`: only the part of the path up to this time
    (clip videos grow the line as the video plays). `scale`: source px → image px."""
    pts = []
    for p in path or []:
        try:
            x, y = float(p[0]) * scale, float(p[1]) * scale
            t = float(p[2]) if len(p) > 2 else None
        except (TypeError, ValueError, IndexError):
            continue
        if until is not None and t is not None and t > until:
            break
        pts.append((int(round(x)), int(round(y))))
    if not pts:
        return img
    thickness = max(1, int(thickness))
    if len(pts) >= 2:
        poly = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        # Dark outline under the yellow line keeps it visible on light floors / walls
        cv2.polylines(img, [poly], False, OUTLINE, thickness + 2, cv2.LINE_AA)
        cv2.polylines(img, [poly], False, PATH_COLOR, thickness, cv2.LINE_AA)
        # Arrow head on the last real segment (direction of travel)
        tail = next((q for q in reversed(pts[:-1]) if q != pts[-1]), None)
        if tail is not None:
            seg = max(1.0, float(np.hypot(pts[-1][0] - tail[0], pts[-1][1] - tail[1])))
            tip = min(0.9, 18.0 * max(1.0, thickness / 2) / seg)
            cv2.arrowedLine(img, tail, pts[-1], OUTLINE, thickness + 2, cv2.LINE_AA, tipLength=tip)
            cv2.arrowedLine(img, tail, pts[-1], PATH_COLOR, thickness, cv2.LINE_AA, tipLength=tip)
    # Start point "A"
    r = thickness + 4
    cv2.circle(img, pts[0], r, OUTLINE, -1, cv2.LINE_AA)
    cv2.circle(img, pts[0], r - 1, START_COLOR, -1, cv2.LINE_AA)
    font_scale = max(0.4, thickness * 0.18)
    cv2.putText(img, "A", (pts[0][0] + r + 2, pts[0][1] - r), cv2.FONT_HERSHEY_SIMPLEX, font_scale, OUTLINE,
                thickness + 1, cv2.LINE_AA)
    cv2.putText(img, "A", (pts[0][0] + r + 2, pts[0][1] - r), cv2.FONT_HERSHEY_SIMPLEX, font_scale, START_COLOR,
                max(1, thickness - 1), cv2.LINE_AA)
    return img


# COCO skeleton: face, arms, torso, legs (keypoint index pairs)
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
SKELETON_COLOR = (255, 200, 0)  # cyan-blue (BGR), distinct from green / orange / red boxes


def draw_skeleton(img: np.ndarray, kps: list, thickness: int = 2, min_conf: float = 0.3,
                  color: tuple[int, int, int] = SKELETON_COLOR) -> np.ndarray:
    """17 COCO keypoints [x, y, conf] → lines + joints, drawn in place."""
    pts = []
    for p in (kps or [])[:17]:
        try:
            pts.append((int(p[0]), int(p[1])) if float(p[2]) >= min_conf else None)
        except (IndexError, TypeError, ValueError):
            pts.append(None)
    for a, b in SKELETON:
        if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], color, thickness, cv2.LINE_AA)
    for p in pts:
        if p:
            cv2.circle(img, p, thickness + 1, color, -1, cv2.LINE_AA)
    return img
