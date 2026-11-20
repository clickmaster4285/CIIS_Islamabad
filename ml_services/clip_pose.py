"""Frame-by-frame YOLO pose for movement clips.

Live inference sees each camera only every few seconds, so a clip has just a few body
points. When a clip is finished, YOLO pose runs on EVERY clip frame (8 fps): each moving
person is followed frame to frame (pose box overlap with the previous frame), giving
a skeleton per frame and a dense path of the hips — the real walking route.

Pure functions; the pose model is injected as `pose_batch(images) -> [[pose, ...], ...]`
where a pose is {"bbox": [x1, y1, x2, y2], "score": f, "keypoints": [[x, y, c] * 17]}.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

PoseBatch = Callable[[list[np.ndarray]], list[list[dict[str, Any]]]]

L_HIP, R_HIP, L_SHOULDER, R_SHOULDER = 11, 12, 5, 6


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def body_point(pose: dict[str, Any], min_conf: float = 0.3) -> tuple[float, float]:
    """Hip centre; shoulders if hips are hidden; else box centre."""
    kps = pose.get("keypoints") or []

    def kp(i):
        try:
            return (float(kps[i][0]), float(kps[i][1])) if float(kps[i][2]) >= min_conf else None
        except (IndexError, TypeError, ValueError):
            return None

    for a, b in ((L_HIP, R_HIP), (L_SHOULDER, R_SHOULDER)):
        pa, pb = kp(a), kp(b)
        if pa and pb:
            return (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
        if pa or pb:
            return pa or pb
    x1, y1, x2, y2 = pose["bbox"][:4]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_of(det: dict[str, Any]) -> tuple[float, float]:
    """People (keypoints): hip centre. Vehicles / objects (no keypoints): bottom centre of the
    box — where the wheels touch the ground, so the path follows the road, not the roof."""
    if det.get("keypoints"):
        return body_point(det)
    x1, _y1, x2, y2 = det["bbox"][:4]
    return (x1 + x2) / 2.0, float(y2)


def run_pose(images: list[np.ndarray], pose_batch: PoseBatch, batch_size: int = 8) -> list[list[dict[str, Any]]]:
    """Pose for every image, in small batches (each batch holds the GPU only briefly)."""
    out: list[list[dict[str, Any]]] = []
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        res = pose_batch(chunk) or []
        out.extend(list(res) + [[] for _ in range(len(chunk) - len(res))])
    return out


def track_person(
    poses_per_frame: list[list[dict[str, Any]]],
    times: list[float],
    seed_box: Sequence[float],
    *,
    min_iou: float = 0.2,
    max_gap: int = 8,
    taken: list[set[int]] | None = None,
) -> list[dict[str, Any] | None]:
    """Follow one person through the frames, starting from `seed_box` (clip pixels).

    Each frame picks the pose that overlaps the person's last box best. Missed for up
    to `max_gap` frames (~1 s at 8 fps) → keep looking near the last box; longer → lost.
    `taken` marks poses already given to another moving person in the same clip.
    """
    track: list[dict[str, Any] | None] = []
    box = list(seed_box)
    missed = 0
    for fi, poses in enumerate(poses_per_frame):
        best, best_iou, best_j = None, min_iou, -1
        for j, pose in enumerate(poses):
            if taken is not None and j in taken[fi]:
                continue
            score = _iou(box, pose["bbox"])
            if score > best_iou:
                best, best_iou, best_j = pose, score, j
        if best is not None:
            box = list(best["bbox"])
            missed = 0
            if taken is not None:
                taken[fi].add(best_j)
            track.append({"t": times[fi], "pose": best, "point": point_of(best)})
        else:
            missed += 1
            track.append(None)
            if missed > max_gap:
                # Lost: keep the remaining frames empty rather than jump to someone else
                track.extend([None] * (len(poses_per_frame) - fi - 1))
                break
    return track


def dense_path(track: list[dict[str, Any] | None], start_time: float | None = None) -> list[list[float]]:
    """[x, y, t] of the body point for every tracked frame from the movement start on."""
    path: list[list[float]] = []
    for item in track:
        if item is None:
            continue
        if start_time is not None and item["t"] < start_time - 0.5:
            continue
        x, y = item["point"]
        if path and math.hypot(path[-1][0] - x, path[-1][1] - y) < 1.0:
            continue
        path.append([round(x, 1), round(y, 1), round(item["t"], 2)])
    return path


def smooth_path(path: list[list[float]], window: int = 3) -> list[list[float]]:
    """Moving average of x, y (pose jitter between frames) — keeps first and last point."""
    if len(path) <= 2 or window <= 1:
        return path
    half = window // 2
    out = [path[0]]
    for i in range(1, len(path) - 1):
        lo, hi = max(0, i - half), min(len(path), i + half + 1)
        xs = [p[0] for p in path[lo:hi]]
        ys = [p[1] for p in path[lo:hi]]
        out.append([round(sum(xs) / len(xs), 1), round(sum(ys) / len(ys), 1), path[i][2]])
    out.append(path[-1])
    return out


def _centre(b: Sequence[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def seed_pose(poses: list[dict[str, Any]], box: Sequence[float], max_rel: float = 0.75,
              exclude: set[int] | None = None) -> int | None:
    """Index of the pose closest to `box` (centre distance ≤ max_rel × box height), else None.

    Detector boxes and pose boxes differ in shape, so overlap alone is unreliable here.
    """
    cx, cy = _centre(box)
    limit = max_rel * max(1.0, box[3] - box[1])
    best, best_d = None, limit
    for j, pose in enumerate(poses):
        if exclude and j in exclude:
            continue
        px, py = _centre(pose["bbox"])
        d = math.hypot(px - cx, py - cy)
        if d <= best_d:
            best, best_d = j, d
    return best


def track_from_seed(
    poses_per_frame: list[list[dict[str, Any]]],
    times: list[float],
    seed_frame: int,
    seed_index: int,
    *,
    taken: list[set[int]] | None = None,
    min_iou: float = 0.2,
    max_gap: int = 8,
) -> list[dict[str, Any] | None]:
    """Follow one person backward and forward from a known pose in `seed_frame`."""
    n = len(poses_per_frame)
    seed = poses_per_frame[seed_frame][seed_index]
    if taken is not None:
        taken[seed_frame].add(seed_index)
    after = track_person(poses_per_frame[seed_frame + 1 :], times[seed_frame + 1 :], seed["bbox"],
                         min_iou=min_iou, max_gap=max_gap, taken=taken[seed_frame + 1 :] if taken else None)
    before_rev = track_person(poses_per_frame[:seed_frame][::-1], times[:seed_frame][::-1], seed["bbox"],
                              min_iou=min_iou, max_gap=max_gap,
                              taken=taken[:seed_frame][::-1] if taken else None)
    middle = {"t": times[seed_frame], "pose": seed, "point": point_of(seed)}
    track = before_rev[::-1] + [middle] + after
    return track[:n]


def nearest_frame(times: list[float], t: float | None) -> int:
    if t is None or not times:
        return len(times) - 1
    return min(range(len(times)), key=lambda i: abs(times[i] - t))
