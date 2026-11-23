"""
Push local host metrics to Infrastructure Monitoring.

Run ON the monitored server (e.g. 192.168.190.137):

  python manage.py push_server_metrics --device-id 10 --api-base http://<ciis-host>:8000 --token <jwt>

The token can come from the INFRA_AGENT_TOKEN environment variable instead of
--token (keeps it out of the process list / shell history).

Or serve a pull endpoint for the poller. It binds 127.0.0.1 by default; to let the
CIIS poller pull it over the network pass --bind 0.0.0.0 AND set a shared secret
(--secret or INFRA_AGENT_SECRET). The poller sends the same secret (settings /
environment INFRA_AGENT_SECRET, or metadata.agent_secret on the server device) in
the X-Infra-Agent-Secret header:

  set INFRA_AGENT_SECRET=<random>
  python manage.py push_server_metrics --serve --bind 0.0.0.0 --port 9410

Serve + push loop (push errors are logged and the loop keeps running):

  python manage.py push_server_metrics --device-id 10 --api-base http://... --loop 30 --serve
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Collect local CPU/RAM/disk/GPU/services/logs and push to infra server device"

    def add_arguments(self, parser):
        parser.add_argument("--device-id", type=int, default=None)
        parser.add_argument(
            "--api-base",
            type=str,
            default="",
            help="CIIS API base, e.g. http://192.168.190.132:8000",
        )
        parser.add_argument(
            "--token",
            type=str,
            default="",
            help="JWT / Bearer token (default: INFRA_AGENT_TOKEN environment variable)",
        )
        parser.add_argument("--loop", type=int, default=0, help="Repeat every N seconds (0 = once)")
        parser.add_argument("--serve", action="store_true", help="Expose /infra-agent/metrics")
        parser.add_argument("--port", type=int, default=9410)
        parser.add_argument(
            "--bind",
            type=str,
            default="127.0.0.1",
            help="Address for --serve (default 127.0.0.1; use 0.0.0.0 for poller pulls + set --secret)",
        )
        parser.add_argument(
            "--secret",
            type=str,
            default="",
            help="Shared secret required in X-Infra-Agent-Secret for --serve "
            "(default: INFRA_AGENT_SECRET environment variable)",
        )

    def handle(self, *args, **options):
        from infrastructure_monitoring.server_health import collect_local_server_metrics

        device_id = options["device_id"]
        api_base = (options["api_base"] or "").rstrip("/")
        token = options["token"] or os.environ.get("INFRA_AGENT_TOKEN", "")
        loop_sec = int(options["loop"] or 0)
        serve = bool(options["serve"])
        port = int(options["port"])
        bind = options["bind"]
        secret = (options["secret"] or os.environ.get("INFRA_AGENT_SECRET", "")).strip()

        cache: dict = {"metrics": collect_local_server_metrics()}

        def refresh():
            cache["metrics"] = collect_local_server_metrics()
            return cache["metrics"]

        def push_once():
            if not device_id or not api_base:
                raise CommandError("--device-id and --api-base required to push")
            metrics = refresh()
            logs = metrics.get("server_logs") or []
            payload = {"metrics": {**metrics, "server_logs": []}, "logs": logs}
            url = f"{api_base}/api/infra/devices/{device_id}/update_metrics/"
            body = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"Pushed metrics to device {device_id} (HTTP {resp.status})"
                        )
                    )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
                raise CommandError(f"Push failed HTTP {exc.code}: {detail}") from exc
            except Exception as exc:
                raise CommandError(f"Push failed: {exc}") from exc

        if serve:
            if bind not in ("127.0.0.1", "localhost", "::1") and not secret:
                self.stderr.write(
                    self.style.WARNING(
                        f"Serving on {bind} WITHOUT a shared secret: anyone on the network can "
                        "read this host's metrics and logs. Set --secret / INFRA_AGENT_SECRET."
                    )
                )

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):  # noqa: N802
                    if self.path.split("?")[0] not in ("/infra-agent/metrics", "/metrics"):
                        self.send_response(404)
                        self.end_headers()
                        return
                    if secret:
                        sent = self.headers.get("X-Infra-Agent-Secret") or ""
                        if not hmac.compare_digest(sent.encode(), secret.encode()):
                            self.send_response(401)
                            self.end_headers()
                            return
                    data = json.dumps(cache.get("metrics") or {}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

                def log_message(self, fmt, *args):
                    return

            httpd = ThreadingHTTPServer((bind, port), Handler)
            t = threading.Thread(target=httpd.serve_forever, daemon=True)
            t.start()
            self.stdout.write(self.style.SUCCESS(f"Agent listening on {bind}:{port}"))

        if device_id and api_base:
            if loop_sec <= 0:
                push_once()  # single push: report failure via exit code
            else:
                while True:
                    try:
                        push_once()
                    except CommandError as exc:
                        # Keep the agent alive through CIIS restarts / network blips.
                        self.stderr.write(self.style.ERROR(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {exc}"))
                    except Exception as exc:  # noqa: BLE001
                        self.stderr.write(self.style.ERROR(f"Collect/push error: {exc}"))
                    time.sleep(loop_sec)
        elif serve:
            self.stdout.write("Serving metrics only (Ctrl+C to stop). Refreshing every 15s…")
            while True:
                time.sleep(15)
                refresh()
        else:
            metrics = refresh()
            self.stdout.write(json.dumps({k: metrics.get(k) for k in (
                "hostname", "cpu_percent", "memory_percent", "gpu_count",
                "services_total", "server_logs_count", "disks",
            )}, indent=2, default=str))
            self.stdout.write(
                self.style.WARNING(
                    "No push performed. Pass --device-id and --api-base, or --serve."
                )
            )
