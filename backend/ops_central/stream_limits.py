"""Concurrent live-stream caps for the hub's MJPEG / view proxies.

Each open MJPEG response holds a worker thread, an upstream socket and (for the view
path) an ffmpeg process, so the hub limits how many run at once:

* ``OPS_MAX_CONCURRENT_STREAMS`` (default 24) for the whole process
* ``OPS_MAX_STREAMS_PER_USER`` (default 6) for one user

The counters are per process (each gunicorn/uvicorn worker has its own).
"""

from __future__ import annotations

import threading
from collections import defaultdict

from django.conf import settings
from django.http import StreamingHttpResponse
from rest_framework import status
from rest_framework.response import Response

_lock = threading.Lock()
_total = 0
_per_user: dict[str, int] = defaultdict(int)


def _limits() -> tuple[int, int]:
    return (
        int(getattr(settings, "OPS_MAX_CONCURRENT_STREAMS", 24)),
        int(getattr(settings, "OPS_MAX_STREAMS_PER_USER", 6)),
    )


def _user_key(user) -> str:
    if user is None or not getattr(user, "is_authenticated", False):
        return ""
    return f"u{user.pk}"


def try_acquire(user) -> bool:
    global _total
    max_total, max_user = _limits()
    key = _user_key(user)
    with _lock:
        if max_total > 0 and _total >= max_total:
            return False
        if key and max_user > 0 and _per_user[key] >= max_user:
            return False
        _total += 1
        if key:
            _per_user[key] += 1
    return True


def release(user) -> None:
    global _total
    key = _user_key(user)
    with _lock:
        _total = max(0, _total - 1)
        if key:
            _per_user[key] = max(0, _per_user[key] - 1)
            if not _per_user[key]:
                _per_user.pop(key, None)


def limit_response() -> Response:
    return Response(
        {"detail": "Too many live streams are open. Close some camera tiles and try again."},
        status=status.HTTP_429_TOO_MANY_REQUESTS,
    )


class _ReleasingIterator:
    """Wraps a stream body; the slot is released once when the response closes."""

    def __init__(self, iterable, user):
        self._it = iter(iterable)
        self._source = iterable
        self._user = user
        self._released = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        try:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()
        finally:
            if not self._released:
                self._released = True
                release(self._user)


def limited_streaming_response(iterable, user, **kwargs) -> StreamingHttpResponse:
    """StreamingHttpResponse whose slot (taken with ``try_acquire``) is freed on close."""
    return StreamingHttpResponse(_ReleasingIterator(iterable, user), **kwargs)
