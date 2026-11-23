"""Local track → face (persons) → strict ReID → stable global object id."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from live_metrics import get_live_metrics, ml_node_id
from reid_extractor import extract_reid_embedding, extract_reid_embeddings  # noqa: F401

VEHICLE_CLASSES = frozenset(
    {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "vehicle",
    }
)
PERSON_CLASSES = frozenset({"person", "face"})
EXCLUDED_MODELS = frozenset({"smoke", "weapon"})
EXCLUDED_CLASS_NAMES = frozenset(
    {
        "smoke",
        "fire",
        "flame",
        "weapon",
        "gun",
        "pistol",
        "rifle",
    }
)

# Tight thresholds — prefer new IDs over wrong merges (was 0.85).
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


REID_MATCH_THRESHOLD = _env_float("ML_REID_MATCH_THRESHOLD", 0.94)
PERSON_REID_MATCH_THRESHOLD = _env_float("ML_PERSON_REID_MATCH_THRESHOLD", 0.95)
VEHICLE_REID_MATCH_THRESHOLD = _env_float("ML_VEHICLE_REID_MATCH_THRESHOLD", 0.93)
# Only blend embeddings when the match was this strong (stops polluted gallery).
REID_EMA_MIN_SCORE = _env_float("ML_REID_EMA_MIN_SCORE", 0.96)
GALLERY_TTL_SEC = 6 * 60 * 60
TRACK_EXIT_SEC = 8.0
# Gallery / bind pruning runs at most this often (it used to run on every frame of every camera).
PRUNE_EVERY_SEC = max(0.5, _env_float("ML_IDENTITY_PRUNE_SEC", 5.0))

_log = logging.getLogger("ml.identity")


def object_type_for_class(class_name: str) -> str:
    name = (class_name or "").strip().lower()
    if name in PERSON_CLASSES:
        return "person"
    if name in VEHICLE_CLASSES:
        return "vehicle"
    return "object"


def _same_identity_class(stored_class: str, new_class: str) -> bool:
    a = (stored_class or "").strip().lower()
    b = (new_class or "").strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    aliases = {"face": "person", "vehicle": "car"}
    return aliases.get(a, a) == aliases.get(b, b)


def is_identity_detection(det: dict[str, Any]) -> bool:
    model = str(det.get("model") or det.get("model_tag") or "").strip().lower()
    if model in EXCLUDED_MODELS:
        return False
    cls = str(det.get("class_name") or det.get("label") or "").strip().lower()
    if cls in EXCLUDED_CLASS_NAMES:
        return False
    if "smoke" in cls or "fire" in cls or "flame" in cls:
        return False
    return True


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(va / na, vb / nb))


def _crop(frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
    if frame is None or frame.size == 0 or len(bbox) < 4:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def _reid_threshold(object_type: str) -> float:
    if object_type == "person":
        return PERSON_REID_MATCH_THRESHOLD
    if object_type == "vehicle":
        return VEHICLE_REID_MATCH_THRESHOLD
    return REID_MATCH_THRESHOLD


def _normalize_face_key(identity: str | None) -> str:
    """Stable gallery key from face recognizer identity (staff or unknown-* cache)."""
    raw = (identity or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if low in {"unknown", "person", "face"}:
        return ""
    # Bare "unknown" rejected; "unknown-12" / staff names kept.
    if low == "unknown":
        return ""
    return f"face:{low}"


def face_bbox_key(bbox: Any) -> tuple[int, int, int, int] | None:
    """Exact-box key for reusing a face result within one frame (ByteTrack keeps boxes)."""
    try:
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    except (TypeError, ValueError, IndexError):
        return None


def face_result_from_detection(det: dict[str, Any]) -> tuple[str, float, list[float]]:
    """
    (face_key, score, face_embedding) from a detection that already went through
    KnownFaceDB.label_detection_detail in parse_yolo_result. Mirrors what
    _resolve_face_identity returns for the same crop, without recomputing it.
    """
    identity = str(det.get("label") or "")
    score = float(det.get("face_match_score") or 0.0)
    emb = det.get("face_embedding") or []
    return _normalize_face_key(identity), score, list(emb) if isinstance(emb, list) else []


def _resolve_face_identity(face_db: Any, frame: np.ndarray, bbox: list, class_name: str) -> tuple[str, float, list[float]]:
    """Return (face_key, score, face_embedding) for person/face boxes."""
    if face_db is None or frame is None or frame.size == 0 or len(bbox) < 4:
        return "", 0.0, []
    try:
        cls_id = 0 if (class_name or "").lower() == "person" else 80
        if hasattr(face_db, "label_detection_detail"):
            meta = face_db.label_detection_detail(frame, cls_id, class_name or "person", bbox)
            identity = str(meta.get("label") or "")
            score = float(meta.get("face_match_score") or 0.0)
            emb = meta.get("face_embedding") or []
            key = _normalize_face_key(identity)
            if key:
                return key, score, list(emb) if isinstance(emb, list) else []
            return "", score, list(emb) if isinstance(emb, list) else []
        if hasattr(face_db, "recognize_person"):
            identity = str(face_db.recognize_person(frame, bbox) or "")
            return _normalize_face_key(identity), 0.0, []
    except Exception as exc:
        _log_face_error(exc)
        return "", 0.0, []
    return "", 0.0, []


_face_err_at = 0.0


def _log_face_error(exc: Exception) -> None:
    """Face recognition failures used to be dropped silently; log them (at most every 30 s)."""
    global _face_err_at
    now = time.monotonic()
    if now - _face_err_at >= 30.0:
        _face_err_at = now
        _log.warning("[identity] face recognition failed: %s", exc, exc_info=True)


@dataclass
class GlobalObjectState:
    global_id: str
    object_type: str
    class_name: str
    reid_embedding: list[float] = field(default_factory=list)
    face_key: str = ""
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    entry_time: float = field(default_factory=time.time)
    exit_time: float | None = None
    camera_history: list[str] = field(default_factory=list)
    active: bool = True


@dataclass
class LocalTrackBind:
    global_id: str
    last_seen: float


_DATA_DIR = Path(__file__).resolve().parent / "data"
# Shared file used before counters were kept per node; a new per-node file starts from it.
LEGACY_COUNTER_FILE = _DATA_DIR / "identity_counters.json"


def default_counter_file() -> Path:
    """ML_IDENTITY_COUNTER_FILE, else data/identity_counters.<node>.json (one per gpu0 / gpu1).

    Two pm2 processes sharing one file overwrote each other's counters.
    """
    explicit = os.getenv("ML_IDENTITY_COUNTER_FILE", "").strip()
    if explicit:
        return Path(explicit)
    node = ml_node_id()
    return _DATA_DIR / (f"identity_counters.{node}.json" if node else "identity_counters.json")


COUNTER_FILE = default_counter_file()


# First run without a counter file: start above codes handed out before counters were kept
_FIRST_RUN_START = 1000
_COUNTER_KEYS = ("person", "vehicle", "object")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _read_counters(path: Path) -> dict[str, int]:
    """Parse a counter file; raises on any problem."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("counter file is not a JSON object")
    return {k: int(v) for k, v in raw.items() if k in _COUNTER_KEYS}


def _load_counters(path: Path, legacy: Path | None = None) -> dict[str, int]:
    """Counters to continue from; never below the first-run start or the legacy shared file.

    A corrupt / unreadable file is logged as an ERROR and moved aside — it is never silently
    reset to 0 (that handed out GP1, GP2 … again and merged different people in the backend).
    """
    floor = {k: _FIRST_RUN_START for k in _COUNTER_KEYS}
    if legacy is not None and legacy != path and legacy.exists():
        try:
            for k, v in _read_counters(legacy).items():
                floor[k] = max(floor[k], v)
        except Exception as exc:
            _log.error("[identity] Legacy ID counter file %s unreadable (%s)", legacy, exc)
    if not path.exists():
        return floor
    try:
        loaded = _read_counters(path)
    except Exception as exc:
        msg = (
            f"[identity] ERROR: ID counter file {path} is unreadable ({exc}) — continuing from "
            f"{floor}; codes handed out above that may repeat. Fix the file or raise the counters."
        )
        _log.error(msg)
        print(msg)
        try:
            path.replace(path.with_name(f"{path.name}.corrupt-{int(time.time())}"))
        except OSError:
            pass
        return floor
    return {k: max(floor[k], loaded.get(k, 0)) for k in _COUNTER_KEYS}


def _save_counters(path: Path, counters: dict[str, int]) -> None:
    """Atomic + durable: unique temp file in the same dir, fsync, then os.replace."""
    tmp_name = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(counters))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        tmp_name = ""
    except OSError as exc:
        _log.error("[identity] Could not save ID counters to %s (%s)", path, exc)
        print(f"[identity] Could not save ID counters ({exc}) — codes may repeat after restart")
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


class _TypeGallery:
    """Unit-normalised ReID vectors of one (object type, dimension) as one numpy matrix.

    Rows are updated in place, appended or swap-removed, so matching is a single mat-vec
    product instead of a Python cosine loop over every gallery entry.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.mat = np.zeros((16, dim), dtype=np.float32)
        self.gids: list[str] = []
        self.row: dict[str, int] = {}

    @staticmethod
    def _unit(vec: list[float]) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else np.zeros_like(v)

    def upsert(self, gid: str, vec: list[float]) -> None:
        v = self._unit(vec)
        i = self.row.get(gid)
        if i is None:
            i = len(self.gids)
            if i >= self.mat.shape[0]:
                grown = np.zeros((self.mat.shape[0] * 2, self.dim), dtype=np.float32)
                grown[:i] = self.mat[:i]
                self.mat = grown
            self.gids.append(gid)
            self.row[gid] = i
        self.mat[i] = v

    def remove(self, gid: str) -> None:
        i = self.row.pop(gid, None)
        if i is None:
            return
        last = len(self.gids) - 1
        if i != last:
            moved = self.gids[last]
            self.mat[i] = self.mat[last]
            self.gids[i] = moved
            self.row[moved] = i
        self.gids.pop()

    def scores(self, vec: list[float]) -> np.ndarray:
        n = len(self.gids)
        if n == 0:
            return np.zeros((0,), dtype=np.float32)
        return self.mat[:n] @ self._unit(vec)


class ObjectIdentityRegistry:
    """In-memory global IDs; PostgreSQL remains the durable source of truth on save."""

    def __init__(self, counter_file: str | Path | None = None):
        self._lock = threading.Lock()
        self._globals: dict[str, GlobalObjectState] = {}
        self._track_bind: dict[tuple[str, int], LocalTrackBind] = {}
        self._face_index: dict[str, str] = {}  # face_key → global_id
        # Stage 4: reuse a bound track's OSNet code for this many seconds (0 = every frame).
        try:
            self._reid_refresh = max(0.0, float(os.getenv("ML_REID_TRACK_REFRESH_SEC", "0") or 0))
        except (TypeError, ValueError):
            self._reid_refresh = 0.0
        self._reid_cache: dict[tuple[str, int], tuple[list[float], float]] = {}
        self._reid_cache_lock = threading.Lock()
        self._reid_cache_pruned = 0.0
        # Counters survive restarts so a code (GP12) is never handed out twice: the backend
        # links movements by this code, and a reused code would merge two different people.
        self._counter_file = Path(counter_file) if counter_file else COUNTER_FILE
        legacy = None if counter_file else LEGACY_COUNTER_FILE
        self._counters = {"person": 0, "vehicle": 0, "object": 0}
        self._counters.update(_load_counters(self._counter_file, legacy=legacy))
        # Several ML nodes hand out codes for the same backend. With STRIDE=N and a distinct
        # OFFSET (0..N-1) per node, a node only uses numbers n with n % N == OFFSET, so codes
        # keep the backend's "GP<digits>" format and never collide between nodes.
        self._code_stride = max(1, _env_int("ML_IDENTITY_CODE_STRIDE", 1))
        self._code_offset = _env_int("ML_IDENTITY_CODE_OFFSET", 0) % self._code_stride
        self._galleries: dict[tuple[str, int], _TypeGallery] = {}
        self._gallery_of: dict[str, tuple[str, int]] = {}  # global_id → gallery key
        self._last_prune = 0.0

    def _next_code(self, object_type: str) -> str:
        prefix = {"person": "GP", "vehicle": "GV", "object": "GO"}.get(object_type, "GO")
        n = self._counters.get(object_type, 0) + 1
        if self._code_stride > 1:
            n += (self._code_offset - n) % self._code_stride
        self._counters[object_type] = n
        _save_counters(self._counter_file, self._counters)
        return f"{prefix}{n}"

    def _gallery_set(self, state: GlobalObjectState) -> None:
        """Mirror state.reid_embedding into its per-type matrix."""
        emb = state.reid_embedding
        old_key = self._gallery_of.get(state.global_id)
        if not emb:
            if old_key is not None:
                self._gallery_drop(state.global_id)
            return
        key = (state.object_type, len(emb))
        if old_key is not None and old_key != key:
            self._gallery_drop(state.global_id)
        gal = self._galleries.get(key)
        if gal is None:
            gal = _TypeGallery(len(emb))
            self._galleries[key] = gal
        gal.upsert(state.global_id, emb)
        self._gallery_of[state.global_id] = key

    def _gallery_drop(self, gid: str) -> None:
        key = self._gallery_of.pop(gid, None)
        if key is not None and key in self._galleries:
            self._galleries[key].remove(gid)

    def _prune(self, now: float) -> None:
        # Timer-driven: walking the whole gallery on every frame of every camera was a hot spot.
        if now - self._last_prune < PRUNE_EVERY_SEC:
            return
        self._last_prune = now
        ttl_cut = now - GALLERY_TTL_SEC
        stale_gids = [gid for gid, st in self._globals.items() if st.last_seen < ttl_cut]
        for gid in stale_gids:
            st = self._globals.pop(gid, None)
            self._gallery_drop(gid)
            if st and st.face_key and self._face_index.get(st.face_key) == gid:
                self._face_index.pop(st.face_key, None)
        stale_tracks = [k for k, b in self._track_bind.items() if b.last_seen < now - GALLERY_TTL_SEC]
        for k in stale_tracks:
            self._track_bind.pop(k, None)

    def _match_face(self, face_key: str) -> GlobalObjectState | None:
        if not face_key:
            return None
        gid = self._face_index.get(face_key)
        if not gid:
            return None
        return self._globals.get(gid)

    def _match_reid(
        self,
        object_type: str,
        class_name: str,
        embedding: list[float],
    ) -> tuple[GlobalObjectState | None, float]:
        if not embedding:
            return None, 0.0
        threshold = _reid_threshold(object_type)
        gal = self._galleries.get((object_type, len(embedding)))
        if gal is None or not gal.gids:
            return None, 0.0
        scores = gal.scores(embedding)
        # Best-scoring entry of a compatible class at or above the threshold.
        # (Conflicting face keys are blocked by the caller / at bind time.)
        for i in np.argsort(-scores):
            score = float(scores[i])
            if score < threshold:
                break
            state = self._globals.get(gal.gids[int(i)])
            if state is None or not _same_identity_class(state.class_name, class_name):
                continue
            return state, score
        return None, 0.0

    def _bind_face(self, state: GlobalObjectState, face_key: str) -> None:
        if not face_key:
            return
        # Face key already owned by another global → keep existing stronger identity
        owner = self._face_index.get(face_key)
        if owner and owner != state.global_id:
            return
        state.face_key = face_key
        self._face_index[face_key] = state.global_id

    def forget_camera(self, camera_key: str) -> None:
        """Drop the local track binds / cached codes of a camera (unregistered or reopened).

        A new tracker restarts its track ids; old binds would glue new objects to old IDs.
        """
        cam = (camera_key or "").strip()
        with self._lock:
            for k in [k for k in self._track_bind if k[0] == cam]:
                self._track_bind.pop(k, None)
        with self._reid_cache_lock:
            for k in [k for k in self._reid_cache if k[0] == cam]:
                self._reid_cache.pop(k, None)

    def _close_inactive_tracks(self, camera_key: str, active_ids: set[int], now: float) -> None:
        prefix = (camera_key or "").strip()
        for (cam, tid), bind in list(self._track_bind.items()):
            if cam != prefix:
                continue
            if tid in active_ids:
                continue
            if now - bind.last_seen < TRACK_EXIT_SEC:
                continue
            state = self._globals.get(bind.global_id)
            if state is not None:
                state.active = False
                state.exit_time = bind.last_seen
            self._track_bind.pop((cam, tid), None)

    def _update_reid(self, state: GlobalObjectState, embedding: list[float], match_score: float) -> None:
        if not embedding:
            return
        # Only EMA when confident — avoids blending two different people into one vector.
        if state.reid_embedding and len(state.reid_embedding) == len(embedding):
            if match_score >= REID_EMA_MIN_SCORE:
                merged = [
                    0.85 * float(a) + 0.15 * float(b)
                    for a, b in zip(state.reid_embedding, embedding)
                ]
                norm = float(np.linalg.norm(np.asarray(merged, dtype=np.float32)))
                state.reid_embedding = (
                    [float(v) / norm for v in merged] if norm > 0 else list(embedding)
                )
                self._gallery_set(state)
            # Weak match: keep prior gallery vector
        else:
            state.reid_embedding = list(embedding)
            self._gallery_set(state)

    def enrich_detections(
        self,
        camera_key: str,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        face_db: Any = None,
        face_results: dict[tuple[int, int, int, int], tuple[str, float, list[float]]] | None = None,
    ) -> list[dict[str, Any]]:
        """Add reid_embedding, face keys, object_type, global_object_id.

        face_results: {bbox: (face_key, score, embedding)} already computed for this frame
        (face pass 1 in parse_yolo_result). Those boxes skip face recognition here.
        """
        now = time.time()
        cam = (camera_key or "").strip()
        out: list[dict[str, Any]] = []
        active_ids: set[int] = set()

        # Appearance codes for every object in the frame in ONE batch, before taking the
        # registry lock (the lock is shared by all camera workers; OSNet per object inside
        # it made every camera wait).
        ident = [d for d in (detections or []) if is_identity_detection(d)]
        embedding_for: dict[int, list[float]] = {}
        reused_ids: set[int] = set()
        if self._reid_refresh > 0:
            # Already-bound tracks keep their recent code: the bind decides identity, the
            # code is only carried to the backend and folded into the gallery.
            todo = []
            with self._reid_cache_lock:
                for d in ident:
                    tid = d.get("track_id")
                    try:
                        key = (cam, int(tid)) if tid is not None else None
                    except (TypeError, ValueError):
                        key = None
                    hit = self._reid_cache.get(key) if key is not None else None
                    if hit is not None and (now - hit[1]) < self._reid_refresh and key in self._track_bind:
                        embedding_for[id(d)] = hit[0]
                        reused_ids.add(id(d))
                    else:
                        todo.append(d)
        else:
            todo = ident
        crops = [_crop(frame, list(d.get("bbox") or [])) for d in todo]
        t_reid = time.perf_counter()
        codes = extract_reid_embeddings(crops) if crops else []
        get_live_metrics().record_stage("track.reid_osnet_batch", time.perf_counter() - t_reid)
        for d, e in zip(todo, codes):
            embedding_for[id(d)] = e
        if self._reid_refresh > 0:
            if reused_ids:
                get_live_metrics().record_stage("track.reid_reused", 0.0)
            with self._reid_cache_lock:
                for d, e in zip(todo, codes):
                    tid = d.get("track_id")
                    if e and tid is not None:
                        try:
                            self._reid_cache[(cam, int(tid))] = (e, now)
                        except (TypeError, ValueError):
                            pass
                if now - self._reid_cache_pruned > 10.0:
                    self._reid_cache_pruned = now
                    cutoff = now - max(60.0, self._reid_refresh * 2)
                    for k in [k for k, v in self._reid_cache.items() if v[1] < cutoff]:
                        del self._reid_cache[k]

        t_wait = time.perf_counter()
        with self._lock:
            t_hold = time.perf_counter()
            self._prune(now)

            for det in detections or []:
                if not is_identity_detection(det):
                    # Smoke / fire / weapon: no ReID or global id, but keep them so the
                    # overlay shows them and the backend raises its fire/weapon alerts.
                    out.append(dict(det))
                    continue

                enriched = dict(det)
                class_name = str(enriched.get("class_name") or enriched.get("label") or "object")
                object_type = object_type_for_class(class_name)
                enriched["object_type"] = object_type

                bbox = enriched.get("bbox") or []
                bbox_list = bbox if isinstance(bbox, list) else list(bbox)
                embedding = embedding_for.get(id(det)) or []
                if embedding:
                    enriched["reid_embedding"] = embedding

                face_key = ""
                face_score = 0.0
                if object_type == "person":
                    reused = None
                    if face_results:
                        bkey = face_bbox_key(bbox_list)
                        reused = face_results.get(bkey) if bkey is not None else None
                    if reused is not None:
                        face_key, face_score, face_emb = reused
                        get_live_metrics().record_stage("face.reused_from_pass1", 0.0)
                    else:
                        t_face = time.perf_counter()
                        face_key, face_score, face_emb = _resolve_face_identity(
                            face_db, frame, bbox_list, class_name
                        )
                        get_live_metrics().record_stage(
                            "face.call_from_identity_lock", time.perf_counter() - t_face
                        )
                    if face_key:
                        enriched["face_identity_key"] = face_key
                        enriched["face_match_score"] = face_score
                    if face_emb:
                        enriched["face_embedding"] = face_emb

                track_id = enriched.get("track_id")
                try:
                    track_id_i = int(track_id) if track_id is not None else None
                except (TypeError, ValueError):
                    track_id_i = None

                state: GlobalObjectState | None = None
                match_score = 0.0

                # 1) Active local ByteTrack bind
                if track_id_i is not None:
                    active_ids.add(track_id_i)
                    bind = self._track_bind.get((cam, track_id_i))
                    if bind is not None:
                        state = self._globals.get(bind.global_id)
                        if state is not None and (
                            state.object_type != object_type
                            or not _same_identity_class(state.class_name, class_name)
                        ):
                            self._track_bind.pop((cam, track_id_i), None)
                            state = None
                        elif state is not None and face_key and state.face_key and state.face_key != face_key:
                            # Face proves this track flipped to another person — split.
                            self._track_bind.pop((cam, track_id_i), None)
                            state = None
                        else:
                            match_score = 1.0

                # 2) Face identity (strongest for persons)
                if state is None and face_key:
                    state = self._match_face(face_key)
                    if state is not None:
                        match_score = max(face_score, 0.99)

                # 3) Strict appearance ReID (leave/return without clear face)
                if state is None and embedding:
                    state, match_score = self._match_reid(object_type, class_name, embedding)
                    # If gallery entry is face-locked to someone else, don't steal via clothes.
                    if state is not None and state.face_key and face_key and state.face_key != face_key:
                        state = None
                        match_score = 0.0

                if state is None:
                    gid = self._next_code(object_type)
                    state = GlobalObjectState(
                        global_id=gid,
                        object_type=object_type,
                        class_name=class_name,
                        reid_embedding=list(embedding),
                        face_key="",
                        first_seen=now,
                        last_seen=now,
                        entry_time=now,
                        camera_history=[cam] if cam else [],
                        active=True,
                    )
                    self._globals[gid] = state
                    self._gallery_set(state)
                    match_score = 1.0
                else:
                    reentered = (not state.active) or state.exit_time is not None
                    state.last_seen = now
                    state.active = True
                    state.exit_time = None
                    if reentered:
                        state.entry_time = now
                    if id(det) not in reused_ids:
                        # A reused code is already in the gallery; folding it in again would over-weight it.
                        self._update_reid(state, embedding, match_score)
                    if cam and (not state.camera_history or state.camera_history[-1] != cam):
                        state.camera_history.append(cam)
                        state.camera_history = state.camera_history[-40:]

                if face_key:
                    self._bind_face(state, face_key)

                if track_id_i is not None:
                    self._track_bind[(cam, track_id_i)] = LocalTrackBind(
                        global_id=state.global_id,
                        last_seen=now,
                    )

                enriched["global_object_id"] = state.global_id
                enriched["first_seen"] = state.first_seen
                enriched["last_seen"] = state.last_seen
                enriched["entry_time"] = state.entry_time
                if state.face_key:
                    enriched["face_identity_key"] = state.face_key
                out.append(enriched)

            self._close_inactive_tracks(cam, active_ids, now)
            held = time.perf_counter() - t_hold

        get_live_metrics().record_model(
            "identity_registry", images=len(out), latency=held, lock_wait=t_hold - t_wait, kind="cpu"
        )
        return out


_registry: ObjectIdentityRegistry | None = None
_registry_lock = threading.Lock()


def get_object_identity_registry() -> ObjectIdentityRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ObjectIdentityRegistry()
    return _registry
