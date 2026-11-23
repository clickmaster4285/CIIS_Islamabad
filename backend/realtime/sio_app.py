"""Socket.IO realtime bus for CIIS (WSGI)."""

from __future__ import annotations

import logging
import os
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import parse_qs

import socketio

logger = logging.getLogger(__name__)


def _cors_origins() -> list[str] | str:
    """SOCKETIO_CORS_ORIGINS, else the API's own CORS / CSRF allow-lists, else "*" (with a warning)."""
    raw = os.getenv("SOCKETIO_CORS_ORIGINS", "").strip()
    if raw and raw != "*":
        return [o.strip() for o in raw.split(",") if o.strip()]
    if raw == "*":
        return "*"
    try:
        from django.conf import settings

        origins: list[str] = []
        for name in ("CORS_ALLOWED_ORIGINS", "CSRF_TRUSTED_ORIGINS"):
            for origin in getattr(settings, name, None) or []:
                if origin and origin not in origins:
                    origins.append(origin)
        frontend = (getattr(settings, "FRONTEND_ORIGIN", "") or "").strip().rstrip("/")
        if frontend and frontend not in origins:
            origins.append(frontend)
        if origins:
            return origins
    except Exception:
        logger.debug("[realtime] could not read CORS settings", exc_info=True)
    logger.warning(
        "[realtime] No CORS_ALLOWED_ORIGINS / CSRF_TRUSTED_ORIGINS / SOCKETIO_CORS_ORIGINS set: "
        "Socket.IO accepts any origin (connections still need a valid login token)."
    )
    return "*"


def _client_manager():
    """Redis-backed manager when a Redis URL is configured, so every gunicorn worker /
    detection worker reaches every browser. Without it, an emit only reaches browsers
    connected to the same process."""
    url = os.getenv("SOCKETIO_REDIS_URL", "").strip()
    if not url:
        try:
            from django.conf import settings

            url = (getattr(settings, "SOCKETIO_REDIS_URL", "") or getattr(settings, "REDIS_URL", "") or "").strip()
        except Exception:
            url = ""
    if not url:
        return None
    try:
        return socketio.RedisManager(url)
    except Exception:
        logger.warning("[realtime] Redis manager unavailable (%s); using in-process rooms", url, exc_info=True)
        return None


# threading + Django WSGI (runserver / gunicorn sync) cannot upgrade to WebSocket
# (wsgiref has no raw socket → "Cannot obtain socket from WSGI environment").
# Stay on Engine.IO long-polling unless SOCKETIO_ALLOW_WEBSOCKET=true and a
# websocket-capable server (eventlet/gevent) is used.
_allow_ws = os.getenv("SOCKETIO_ALLOW_WEBSOCKET", "").strip().lower() in ("1", "true", "yes")

_server_kwargs: dict[str, Any] = {}
_manager = _client_manager()
if _manager is not None:
    _server_kwargs["client_manager"] = _manager

sio = socketio.Server(
    async_mode="threading",
    cors_allowed_origins=_cors_origins(),
    logger=False,
    engineio_logger=False,
    allow_upgrades=_allow_ws,
    # Explicit: polling always; websocket only when upgrades enabled.
    transports=["polling", "websocket"] if _allow_ws else ["polling"],
    **_server_kwargs,
)

_AUTH_COOKIE = "tekeye_auth_token"


def _token_from_connect(environ: dict, auth: Any) -> str:
    """Same API token the REST calls use: auth={"token"}, ?token=, Authorization header, or the HttpOnly cookie."""
    if isinstance(auth, dict):
        token = str(auth.get("token") or "").strip()
        if token:
            return token
    query = parse_qs(environ.get("QUERY_STRING") or "")
    token = (query.get("token") or [""])[0].strip()
    if token:
        return token
    header = (environ.get("HTTP_AUTHORIZATION") or "").strip()
    if header.lower().startswith("token "):
        return header[6:].strip()
    cookie_header = environ.get("HTTP_COOKIE") or ""
    if cookie_header:
        try:
            jar = SimpleCookie()
            jar.load(cookie_header)
            if _AUTH_COOKIE in jar:
                return jar[_AUTH_COOKIE].value.strip()
        except Exception:
            pass
    return ""


def _user_for_token(token: str):
    from django.db import connection

    from users.token_auth import user_for_token_key

    try:
        return user_for_token_key(token)
    finally:
        # Socket.IO requests bypass Django's request signals, so nothing else closes this
        # thread's DB connection. (Inside a transaction, e.g. a test, leave it alone.)
        if not connection.in_atomic_block:
            connection.close()


@sio.event
def connect(sid, environ, auth=None):
    token = _token_from_connect(environ, auth)
    user = _user_for_token(token) if token else None
    if user is None:
        logger.info("[realtime] refused unauthenticated client %s", sid)
        raise socketio.exceptions.ConnectionRefusedError("authentication required")
    sio.save_session(sid, {"user_id": user.pk})
    logger.info("[realtime] client connected %s (user %s)", sid, user.pk)
    sio.emit("realtime:hello", {"ok": True}, to=sid)


@sio.event
def disconnect(sid, *args):
    logger.info("[realtime] client disconnected %s", sid)


@sio.on("subscribe")
def on_subscribe(sid, data):
    """Client may join topic rooms: { rooms: ['vms', 'cameras', ...] }."""
    rooms = []
    if isinstance(data, dict):
        raw = data.get("rooms") or data.get("topics") or []
        if isinstance(raw, str):
            rooms = [raw]
        elif isinstance(raw, (list, tuple)):
            rooms = [str(r).strip()[:64] for r in raw if str(r).strip()][:64]
    for room in rooms:
        sio.enter_room(sid, room)
    sio.emit("realtime:subscribed", {"rooms": rooms}, to=sid)


_last_emit_at: dict[str, float] = {}


def emit_invalidate(
    domains: list[str] | tuple[str, ...] | str,
    *,
    payload: dict[str, Any] | None = None,
    throttle_sec: float = 0.0,
) -> None:
    """Send a data-domain invalidation to browsers subscribed to any of these domains.

    The frontend subscribes to every domain it knows on connect, so room-only delivery
    reaches it once (a list of rooms is de-duplicated per client). The old extra
    broadcast to everyone made each subscribed browser refetch twice.
    """
    import time

    if isinstance(domains, str):
        domain_list = [domains]
    else:
        domain_list = [str(d).strip() for d in domains if str(d).strip()]
    if not domain_list:
        return
    if throttle_sec > 0:
        key = ",".join(sorted(domain_list))
        now = time.monotonic()
        last = _last_emit_at.get(key, 0.0)
        if now - last < throttle_sec:
            return
        _last_emit_at[key] = now
    body = {"domains": domain_list, **(payload or {})}
    try:
        sio.emit("realtime:invalidate", body, to=domain_list)
    except Exception:
        logger.debug("[realtime] emit failed domains=%s", domain_list, exc_info=True)


def wrap_wsgi(django_app):
    return socketio.WSGIApp(sio, django_app)
