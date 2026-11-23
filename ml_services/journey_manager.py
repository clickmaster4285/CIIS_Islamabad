"""Journey processing manager — one pipeline thread per camera."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any

import requests

from journey_pipeline import JourneyCameraPipeline
from live_stream import get_live_manager

logger = logging.getLogger(__name__)


def _pipeline_identity(
    *,
    rtsp_url: str,
    camera_id: int | None,
) -> tuple[str, int | None]:
    """Only RTSP/camera_id require a restart; zone/name can update in place."""
    return (rtsp_url.strip(), camera_id)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class _ObservationSender:
    """Posts observations to the backend from one background thread.

    Pipeline threads only enqueue (never block on HTTP). The queue is bounded: when the backend
    is slow or down, the OLDEST observations are dropped first and the drops are logged.
    """

    def __init__(self, post_fn, *, max_size: int):
        self._post_fn = post_fn
        self._max_size = max(1, int(max_size))
        self._queue: deque[dict[str, Any]] = deque()
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._dropped_total = 0
        self._dropped_since_log = 0
        self._last_drop_log = 0.0
        self._sent_total = 0

    def submit(self, payload: dict[str, Any]) -> None:
        with self._cond:
            if len(self._queue) >= self._max_size:
                self._queue.popleft()
                self._dropped_total += 1
                self._dropped_since_log += 1
                now = time.monotonic()
                if not self._last_drop_log or now - self._last_drop_log >= 30.0:
                    logger.warning(
                        "[journey] Ingest queue full (%d): dropped %d oldest observation(s) "
                        "since the last report (%d total) - backend ingest is too slow or down",
                        self._max_size,
                        self._dropped_since_log,
                        self._dropped_total,
                    )
                    self._dropped_since_log = 0
                    self._last_drop_log = now
            self._queue.append(payload)
            self._ensure_thread_locked()
            self._cond.notify()

    def _ensure_thread_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="journey-ingest-sender")
        self._thread.start()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                payload = self._queue.popleft()
            try:
                self._post_fn(payload)
                self._sent_total += 1
            except Exception as exc:  # never let the sender thread die
                logger.debug("[journey] Ingest send error: %s", exc)

    def stats(self) -> dict[str, int]:
        with self._cond:
            return {
                "queued": len(self._queue),
                "max_queue": self._max_size,
                "dropped_total": self._dropped_total,
                "sent_total": self._sent_total,
            }


class JourneyManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipelines: dict[str, JourneyCameraPipeline] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_flags: dict[str, threading.Event] = {}
        self._configs: dict[str, tuple[str, int | None]] = {}
        self._backend_ingest_url = ""
        self._ingest_token = ""
        self._sender = _ObservationSender(
            self._post_observation,
            max_size=_env_int("JOURNEY_INGEST_QUEUE_MAX", 500),
        )

    def configure_ingest(self, backend_ingest_url: str, ingest_token: str = ""):
        self._backend_ingest_url = (backend_ingest_url or "").strip()
        self._ingest_token = (ingest_token or "").strip()

    def register_cameras_bulk(self, payload: dict[str, Any]) -> dict[str, Any]:
        cameras = payload.get("cameras") or []
        if payload.get("backend_ingest_url"):
            self.configure_ingest(
                str(payload.get("backend_ingest_url") or ""),
                str(payload.get("ingest_token") or ""),
            )
        registered = 0
        started = 0
        reused = 0
        desired_keys: set[str] = set()
        live = get_live_manager()
        live.ensure_started()
        for item in cameras:
            key = str(item.get("key") or "").strip()
            url = str(item.get("rtsp_url") or "").strip()
            if not key or not url:
                continue
            camera_id = item.get("camera_id")
            try:
                camera_id = int(camera_id) if camera_id is not None else None
            except (TypeError, ValueError):
                camera_id = None
            desired_keys.add(key)
            live.ensure_camera(key, url)
            outcome = self._ensure_pipeline(
                key=key,
                rtsp_url=url,
                camera_id=camera_id,
                zone=str(item.get("zone") or ""),
                name=str(item.get("name") or key),
            )
            registered += 1
            if outcome == "started":
                started += 1
            else:
                reused += 1

        # Stop pipelines for cameras removed from DB (empty sync stops all).
        stopped = 0
        with self._lock:
            stale = [k for k in list(self._pipelines.keys()) if k not in desired_keys]
            retired = [self._detach_pipeline_locked(key) for key in stale]
            stopped = len(stale)
        self._finish_stopping(retired)
        for key in stale:
            try:
                live.unregister_camera(key)
            except Exception:
                pass

        return {
            "registered": registered,
            "started": started,
            "reused": reused,
            "stopped": stopped,
            "total": len(cameras),
            "running": len(self._pipelines),
            "ingest_url": self._backend_ingest_url,
        }

    def unregister_camera(self, key: str) -> bool:
        key = (key or "").strip()
        if not key:
            return False
        with self._lock:
            existed = key in self._pipelines
            retired = [self._detach_pipeline_locked(key)]
        self._finish_stopping(retired)
        try:
            get_live_manager().unregister_camera(key)
        except Exception:
            pass
        return existed

    def _ensure_pipeline(
        self,
        *,
        key: str,
        rtsp_url: str,
        camera_id: int | None,
        zone: str,
        name: str,
    ) -> str:
        """Start only when missing/dead/config-changed. Periodic sync must not thrash."""
        identity = _pipeline_identity(rtsp_url=rtsp_url, camera_id=camera_id)
        retired: list[tuple[JourneyCameraPipeline | None, threading.Thread | None]] = []
        try:
            return self._swap_pipeline(
                key=key,
                rtsp_url=rtsp_url,
                camera_id=camera_id,
                zone=zone,
                name=name,
                identity=identity,
                retired=retired,
            )
        finally:
            # Join the replaced pipeline's thread only after the manager lock is released.
            self._finish_stopping(retired)

    def _swap_pipeline(
        self,
        *,
        key: str,
        rtsp_url: str,
        camera_id: int | None,
        zone: str,
        name: str,
        identity: tuple[str, int | None],
        retired: list,
    ) -> str:
        """Reuse or replace the pipeline under the lock; a replaced one goes to ``retired``."""
        with self._lock:
            existing = self._pipelines.get(key)
            thread = self._threads.get(key)
            alive = thread is not None and thread.is_alive()
            if existing is not None and alive and self._configs.get(key) == identity:
                # Keep metadata / RTSP handle fresh without tearing down YOLO/track state.
                existing.rtsp_url = rtsp_url
                existing.camera_id = camera_id
                existing.zone = zone
                existing.name = name
                return "reused"

            retired.append(self._detach_pipeline_locked(key))
            pipeline = JourneyCameraPipeline(
                camera_key=key,
                camera_id=camera_id,
                rtsp_url=rtsp_url,
                zone=zone,
                name=name,
            )
            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run_pipeline,
                args=(key, pipeline, stop_event),
                daemon=True,
                name=f"journey-{key}",
            )
            self._pipelines[key] = pipeline
            self._configs[key] = identity
            self._stop_flags[key] = stop_event
            self._threads[key] = thread
            thread.start()
            logger.info("[journey] Started pipeline for %s (%s)", key, name)
            return "started"

    def _detach_pipeline_locked(
        self, key: str
    ) -> tuple[JourneyCameraPipeline | None, threading.Thread | None]:
        """Signal stop and forget the pipeline (caller holds the lock). Never blocks."""
        stop = self._stop_flags.pop(key, None)
        if stop:
            stop.set()
        pipeline = self._pipelines.pop(key, None)
        self._configs.pop(key, None)
        thread = self._threads.pop(key, None)
        return pipeline, thread

    @staticmethod
    def _finish_stopping(
        retired: list[tuple[JourneyCameraPipeline | None, threading.Thread | None]],
    ) -> None:
        """Join stopped threads, then reset their pipelines. Always called WITHOUT the lock."""
        for pipeline, thread in retired:
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception as exc:
                    logger.debug("[journey] Pipeline stop error: %s", exc)

    def _run_pipeline(self, key: str, pipeline: JourneyCameraPipeline, stop_event: threading.Event):
        interval = max(0.2, float(os.getenv("JOURNEY_INFER_INTERVAL_SEC", "0.5")))
        while not stop_event.is_set():
            try:
                observations = pipeline.process_once()
                if stop_event.is_set():
                    break
                for obs in observations:
                    self._sender.submit(obs)
            except Exception as exc:
                logger.warning("[journey] Pipeline error %s: %s", key, exc)
            stop_event.wait(interval)

    def _post_observation(self, payload: dict[str, Any]):
        url = self._backend_ingest_url
        if not url:
            return
        headers = {"Content-Type": "application/json"}
        if self._ingest_token:
            headers["X-Journey-Ingest-Token"] = self._ingest_token
        try:
            res = requests.post(url, json=payload, headers=headers, timeout=10)
            if res.status_code >= 400:
                logger.debug("[journey] Ingest %s: %s", res.status_code, res.text[:200])
        except Exception as exc:
            logger.debug("[journey] Ingest failed: %s", exc)

    def status(self) -> dict[str, Any]:
        with self._lock:
            out = {
                "running_pipelines": len(self._pipelines),
                "cameras": list(self._pipelines.keys()),
                "ingest_url": self._backend_ingest_url,
            }
        out["ingest_queue"] = self._sender.stats()
        return out

    def stop_all(self):
        with self._lock:
            retired = [self._detach_pipeline_locked(key) for key in list(self._pipelines.keys())]
        self._finish_stopping(retired)


_manager: JourneyManager | None = None
_manager_lock = threading.Lock()


def get_journey_manager() -> JourneyManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = JourneyManager()
    return _manager
