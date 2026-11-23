"""Stop password guessing without letting an attacker lock real users out.

Two counters per username, both reset by a successful login:
- per username + client IP: LOGIN_MAX_FAILURES wrong passwords lock that IP out
  of that account, so someone guessing cannot lock the owner out from their own PC;
- per username from all IPs: LOGIN_MAX_FAILURES_TOTAL is a ceiling against
  guessing spread over many machines.

Counts live in the "login_guard" cache (files on disk), which every gunicorn
worker shares; the default in-memory cache is per process.
"""

from __future__ import annotations

import ipaddress
import time

from django.conf import settings
from django.core.cache import caches


def _cache():
    return caches["login_guard"]


def _window_seconds() -> int:
    return max(1, int(getattr(settings, "LOGIN_LOCKOUT_MINUTES", 15))) * 60


def _name(username: str) -> str:
    return (username or "").strip().lower()


def _ip_key(username: str, ip: str) -> str:
    return f"login-fail:{_name(username)}:{ip or '-'}"


def _total_key(username: str) -> str:
    return f"login-fail-all:{_name(username)}"


def _is_loopback(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr).is_loopback
    except ValueError:
        return False


def client_ip(request) -> str:
    """The browser's IP. Proxy headers are trusted only from nginx on this machine."""
    remote = (request.META.get("REMOTE_ADDR") or "").strip()
    if _is_loopback(remote):
        real = (request.META.get("HTTP_X_REAL_IP") or "").strip()
        forwarded = (request.META.get("HTTP_X_FORWARDED_FOR") or "").split(",")[-1].strip()
        for candidate in (real, forwarded):
            try:
                return str(ipaddress.ip_address(candidate))
            except ValueError:
                continue
    return remote


def _lock_key(key: str) -> str:
    return f"{key}:locked"


def _limits() -> tuple[int, int]:
    per_ip = int(getattr(settings, "LOGIN_MAX_FAILURES", 5))
    total = int(getattr(settings, "LOGIN_MAX_FAILURES_TOTAL", 50))
    return per_ip, total


def is_locked_out(username: str, ip: str = "") -> bool:
    if not _name(username):
        return False
    per_ip, total = _limits()
    cache = _cache()
    ip_key, total_key = _ip_key(username, ip), _total_key(username)
    if per_ip > 0 and (cache.get(_lock_key(ip_key)) or int(cache.get(ip_key) or 0) >= per_ip):
        return True
    return total > 0 and bool(cache.get(_lock_key(total_key)) or int(cache.get(total_key) or 0) >= total)


def _bump(key: str, limit: int) -> None:
    """Count one failure. Every write carries the lockout window as its timeout: the
    cache's incr() re-saves with the default 300 s timeout, which shortened the window."""
    cache = _cache()
    window = _window_seconds()
    if cache.add(key, 1, window):
        count = 1
    else:
        count = int(cache.get(key) or 0) + 1
        cache.set(key, count, window)
    if limit > 0 and count >= limit:
        # The lockout runs for a full window from the moment it starts, independent of the counter.
        cache.set(_lock_key(key), int(time.time()), window)


def record_failure(username: str, ip: str = "") -> None:
    if not _name(username):
        return
    per_ip, total = _limits()
    _bump(_ip_key(username, ip), per_ip)
    _bump(_total_key(username), total)


def clear_failures(username: str, ip: str = "") -> None:
    ip_key, total_key = _ip_key(username, ip), _total_key(username)
    _cache().delete_many([ip_key, total_key, _lock_key(ip_key), _lock_key(total_key)])
