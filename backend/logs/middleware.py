import ipaddress
import logging
import threading

from django.conf import settings
from django.contrib.auth import get_user_model

from .models import UserActivityLog

logger = logging.getLogger(__name__)

try:
    from user_agents import parse as parse_user_agent
except ImportError:
    parse_user_agent = None


# Paths we don't log (e.g. report endpoint to avoid duplicate entries, or health checks)
SKIP_LOG_PATHS = (
    "/api/activity-logs/report/",
    "/api/activity-logs/report",
    "/api/mobile-sessions/",
    "/api/mobile/heartbeat/",
    "/api/mobile/heartbeat",
    "/api/mobile/gps/",
    "/api/gps/ping/",
    "/api/gps/heartbeat/",
    "/favicon.ico",
    "/media/",
    "/api/sync/",
)

# GET / HEAD / OPTIONS are polls and list loads (every open tab refetches every few
# seconds), which filled the table with millions of rows. Page views still reach the
# log through the frontend's /api/activity-logs/report/ call; login and logout are
# logged explicitly by their views.
LOGGED_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _valid_ip(raw) -> str | None:
    value = (raw or "").strip()
    if not value:
        return None
    if value.startswith("::ffff:"):
        value = value[7:]
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return None
    return value


def get_ip(request):
    """Client IP, trusting proxy headers only from nginx on this machine (same rule as login lockout)."""
    from users.login_guard import client_ip

    return _valid_ip(client_ip(request))


def get_device_info(request):
    user_agent = request.META.get("HTTP_USER_AGENT", "")
    if parse_user_agent is None:
        return "PC", "", (user_agent or "")[:50]
    parsed = parse_user_agent(user_agent)
    device = "Mobile" if parsed.is_mobile else "PC"
    os_str = parsed.os.family or ""
    browser = parsed.browser.family or ""
    return device, os_str, browser


_geoip_lock = threading.Lock()
_geoip_reader = None
_geoip_failed = False


def _local_geoip():
    """A local MaxMind database, only when GEOIP_PATH is configured. Never a network call."""
    global _geoip_reader, _geoip_failed
    if _geoip_failed or not getattr(settings, "GEOIP_PATH", None):
        return None
    with _geoip_lock:
        if _geoip_reader is None and not _geoip_failed:
            try:
                from django.contrib.gis.geoip2 import GeoIP2

                _geoip_reader = GeoIP2()
            except Exception:
                _geoip_failed = True
                logger.warning("GEOIP_PATH is set but the GeoIP2 database could not be opened", exc_info=True)
    return _geoip_reader


def get_geo(ip):
    if not ip:
        return None, None
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return None, None
    if addr.is_loopback or addr.is_private:
        return "Pakistan", "Local"
    reader = _local_geoip()
    if reader is None:
        return None, None
    try:
        info = reader.city(ip)
        return info.get("country_name"), info.get("city")
    except Exception:
        return None, None


def create_activity_log(user, request, action, source="web"):
    ip = get_ip(request)
    device, os_str, browser = get_device_info(request)
    country, city = get_geo(ip)
    UserActivityLog.objects.create(
        user=user,
        ip_address=ip,
        country=country,
        city=city,
        device=(device or "")[:50],
        os=(os_str or "")[:50],
        browser=(browser or "")[:50],
        action=action[:255],
        source=(source or "web")[:20],
    )


class ActivityLogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        try:
            if request.method not in LOGGED_METHODS:
                return response
            user = getattr(request, "user", None)
            # Sync calls authenticate as SyncPrincipal, not a User. Logging that
            # raised ValueError and turned a successful HQ response into HTTP 500.
            if not isinstance(user, get_user_model()):
                return response
            if not user.is_authenticated:
                return response
            path = request.path.rstrip("/") or "/"
            if any(path.startswith(p.rstrip("/")) for p in SKIP_LOG_PATHS):
                return response
            action = f"{request.method} {path}"
            create_activity_log(user, request, action)
        except Exception:
            logger.exception("Activity log failed for %s", getattr(request, "path", ""))
        return response
