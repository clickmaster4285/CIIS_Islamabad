"""Repeat a live match before it can punch attendance."""

from __future__ import annotations

import threading
import time
from collections import defaultdict

CONFIRM_WEBCAM = 2
CONFIRM_WEBCAM_WINDOW = 12.0
# CCTV: this many per-frame matches on one track must agree with the track's pooled identity.
CONFIRM_CCTV = 3


class MatchConfirmer:
    """Webcam: 2 hits in 12 seconds, from DIFFERENT frames. Used by IdentifyFaceView, not by the CCTV worker."""

    def __init__(self):
        self._hits: dict[str, list[tuple[float, str | None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def observe(
        self,
        key: str,
        *,
        needed: int = CONFIRM_WEBCAM,
        window: float = CONFIRM_WEBCAM_WINDOW,
        frame_hash: str | None = None,
    ) -> bool:
        """Record one match. Re-sending the same image (same ``frame_hash``) does not count twice."""
        now = time.monotonic()
        with self._lock:
            recent = [(stamp, h) for stamp, h in self._hits[key] if now - stamp <= window]
            if frame_hash is None or all(h != frame_hash for _, h in recent):
                recent.append((now, frame_hash))
            self._hits[key] = recent[-8:]
            # Hits without a hash (legacy callers) each count; hashed hits count once per distinct frame.
            distinct = {h if h is not None else f"_{i}" for i, (_, h) in enumerate(self._hits[key])}
            return len(distinct) >= needed

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


_confirmer = MatchConfirmer()


def get_match_confirmer() -> MatchConfirmer:
    return _confirmer
