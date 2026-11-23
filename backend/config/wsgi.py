"""
WSGI config for config project.

Socket.IO is mounted on the same HTTP port as Django so browsers can use
same-origin /socket.io for realtime invalidations. Serve this module (gunicorn
config.wsgi:application); config/asgi.py is plain Django without Socket.IO.
"""
import logging
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from django.core.wsgi import get_wsgi_application

logger = logging.getLogger("config.wsgi")

django_app = get_wsgi_application()

try:
    from realtime.sio_app import wrap_wsgi

    application = wrap_wsgi(django_app)
except Exception:
    # API still works without sockets, but say why live updates are missing.
    logger.exception("Realtime (Socket.IO) disabled: could not load realtime.sio_app")
    application = django_app
