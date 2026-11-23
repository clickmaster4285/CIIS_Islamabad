"""Appearance ReID feature extraction.

Default: OSNet-inspired multi-part (stripe) embedding — no external weights.
Optional: Torch OSNet when ML_REID_BACKEND=osnet (strict: [] if it cannot load, unless
ML_REID_ALLOW_HANDCRAFTED_FALLBACK=1) or ML_REID_BACKEND=auto (OSNet if it loads).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REID_DIM = 512
BASE_DIR = Path(__file__).resolve().parent

_osnet_model: Any = None
_osnet_device: str | None = None
_osnet_failed = False
_osnet_error = ""
_osnet_load_lock = threading.Lock()


def _env_backend() -> str:
    return (os.getenv("ML_REID_BACKEND", "parts") or "parts").strip().lower()


def _part_embedding(person_crop: np.ndarray) -> list[float]:
    """
    Stronger handcrafted ReID: 6 horizontal body stripes × (HSV hist + gradients).
    More discriminative than a single global histogram (reduces wrong person merges).
    """
    if person_crop is None or person_crop.size == 0:
        return []

    img = person_crop
    if img.shape[0] < 32 or img.shape[1] < 16:
        img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_LINEAR)

    h, w = img.shape[:2]
    resized = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ori = np.arctan2(gy, gx)

    parts: list[np.ndarray] = []
    stripes = 6
    stripe_h = resized.shape[0] // stripes
    for i in range(stripes):
        y0 = i * stripe_h
        y1 = resized.shape[0] if i == stripes - 1 else (i + 1) * stripe_h
        hsv_s = hsv[y0:y1, :]
        mag_s = mag[y0:y1, :]
        ori_s = ori[y0:y1, :]
        hist_h = cv2.calcHist([hsv_s], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv_s], [1], None, [12], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv_s], [2], None, [12], [0, 256]).flatten()
        ori_hist, _ = np.histogram(ori_s, bins=8, range=(-np.pi, np.pi), weights=mag_s)
        parts.extend(
            [
                hist_h.astype(np.float32),
                hist_s.astype(np.float32),
                hist_v.astype(np.float32),
                ori_hist.astype(np.float32),
                np.asarray([float(mag_s.mean()), float(mag_s.std())], dtype=np.float32),
            ]
        )

    # Global cues
    aspect = np.array([w / max(h, 1), h / max(w, 1)], dtype=np.float32)
    upper = resized[: resized.shape[0] // 2, :]
    lower = resized[resized.shape[0] // 2 :, :]
    parts.append(aspect)
    parts.append(upper.mean(axis=(0, 1)).astype(np.float32) if upper.size else np.zeros(3, np.float32))
    parts.append(lower.mean(axis=(0, 1)).astype(np.float32) if lower.size else np.zeros(3, np.float32))

    vec = np.concatenate([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts])
    if vec.size < REID_DIM:
        vec = np.pad(vec, (0, REID_DIM - vec.size))
    else:
        vec = vec[:REID_DIM]
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return [float(v) for v in vec]


def _resolve_osnet_weights() -> Path | None:
    env = (os.getenv("ML_REID_WEIGHTS", "") or "").strip()
    candidates: list[Path] = []
    if env:
        p = Path(env)
        candidates.append(p if p.is_absolute() else BASE_DIR / p)
    candidates.extend(
        [
            BASE_DIR / "runs" / "reid" / "osnet_x1_0_market1501.pth",
            BASE_DIR / "runs" / "reid" / "osnet_x1_0.pth",
            BASE_DIR / "weights" / "osnet_x1_0.pth",
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _torchreid_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("torchreid") is not None


def _allow_handcrafted_fallback() -> bool:
    """Explicit opt-in: may a requested-but-broken OSNet silently degrade to handcrafted vectors?"""
    return (os.getenv("ML_REID_ALLOW_HANDCRAFTED_FALLBACK", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _osnet_requested() -> bool:
    return _env_backend() in ("osnet", "torchreid", "auto")


def _strict_osnet() -> bool:
    """``osnet`` / ``torchreid`` = OSNet explicitly required; ``auto`` = OSNet if possible."""
    return _env_backend() in ("osnet", "torchreid")


def _report_osnet_failure(reason: str) -> None:
    global _osnet_error
    _osnet_error = reason
    policy = (
        "falling back to handcrafted part-based ReID (ML_REID_ALLOW_HANDCRAFTED_FALLBACK=1)"
        if (_allow_handcrafted_fallback() or not _strict_osnet())
        else "ReID embeddings DISABLED (returning []). Fix the weights/torch install, or set "
        "ML_REID_ALLOW_HANDCRAFTED_FALLBACK=1 to accept handcrafted vectors (different vector "
        "space: never mix them with OSNet galleries)"
    )
    msg = f"[reid] ERROR: OSNet requested (ML_REID_BACKEND={_env_backend()}) but unavailable: {reason} — {policy}"
    logger.error(msg)
    print(msg, flush=True)


def _load_osnet() -> bool:
    """Load OSNet once (double-checked under a lock). True when the model is ready."""
    global _osnet_model, _osnet_device, _osnet_failed
    if _osnet_model is not None:
        return True
    if _osnet_failed:
        return False
    with _osnet_load_lock:
        if _osnet_model is not None:
            return True
        if _osnet_failed:
            return False

        weights = _resolve_osnet_weights()
        if weights is None:
            _osnet_failed = True
            _report_osnet_failure("no OSNet weights found (ML_REID_WEIGHTS / runs/reid/*.pth)")
            return False

        try:
            import torch
        except Exception as exc:
            _osnet_failed = True
            _report_osnet_failure(f"torch import failed ({exc})")
            return False

        try:
            from inference_engine import resolve_ml_device

            device = "cpu" if resolve_ml_device() == "cpu" else "cuda"
        except Exception:
            device = "cpu"

        if not _torchreid_available():
            # torchreid is not installed (needs a C compiler on Windows): same network, bundled
            try:
                from osnet import load_osnet_x1_0

                model = load_osnet_x1_0(str(weights), device)
                _osnet_device = device
                _osnet_model = model
                print(f"[reid] OSNet loaded from {weights} ({device}, bundled osnet.py, all tensors matched)")
                return True
            except Exception as exc:
                _osnet_failed = True
                _report_osnet_failure(f"bundled OSNet load failed ({exc})")
                return False

        try:
            import torchreid  # type: ignore

            from osnet import load_checkpoint_state

            model = torchreid.models.build_model(
                name="osnet_x1_0",
                num_classes=1,
                pretrained=False,
                use_gpu=device != "cpu",
            )
            state = load_checkpoint_state(str(weights))
            # strict=False only skips missing/unexpected keys — it still raises on a
            # shape mismatch for a key present in both. The checkpoint's classifier
            # head is sized for its own training classes (e.g. 751 for Market1501),
            # which never matches num_classes=1 here; we only need the 512-d backbone
            # embedding anyway, so drop any tensor whose shape doesn't match ours.
            state = {k.removeprefix("module."): v for k, v in state.items()}
            model_state = model.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
            skipped = [k for k in state if k not in filtered]
            model.load_state_dict(filtered, strict=False)
            model.eval()
            model = model.to(device)
            _osnet_device = device
            _osnet_model = model
            print(
                f"[reid] OSNet loaded from {weights} ({device}); "
                f"matched {len(filtered)}/{len(model_state)} tensors (skipped {skipped or 'none'})"
            )
            return True
        except Exception as exc:
            _osnet_failed = True
            _report_osnet_failure(f"torchreid OSNet load failed ({exc})")
            return False


def _try_osnet_embedding(person_crop: np.ndarray) -> list[float]:
    """OSNet vector for one crop, or [] (model unavailable or this crop failed)."""
    if person_crop is None or person_crop.size == 0:
        return []
    if not _load_osnet():
        return []
    try:
        import torch

        tensor = torch.from_numpy(_osnet_preprocess(person_crop)).unsqueeze(0).to(_osnet_device)
        with torch.no_grad():
            feat = _osnet_model(tensor)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = torch.nn.functional.normalize(feat, dim=1)
        vec = feat.squeeze(0).detach().cpu().numpy().astype(np.float32).reshape(-1)
        if vec.size < REID_DIM:
            vec = np.pad(vec, (0, REID_DIM - vec.size))
        else:
            vec = vec[:REID_DIM]
        return [float(v) for v in vec]
    except Exception as exc:
        logger.warning("[reid] OSNet inference failed for one crop: %s", exc)
        return []


def reid_model_name() -> str:
    """Which vector space ``extract_reid_embedding`` currently produces.

    "osnet" | "handcrafted" | "" (no embeddings: OSNet required but unavailable). Vectors from
    different spaces must never be compared — tag stored galleries with this.
    """
    if not _osnet_requested():
        return "handcrafted"
    if _osnet_model is not None or _load_osnet():
        return "osnet"
    if _allow_handcrafted_fallback() or not _strict_osnet():
        return "handcrafted"
    return ""


def extract_reid_embedding_tagged(person_crop: np.ndarray) -> tuple[list[float], str]:
    """``(vector, model)`` — same vector as ``extract_reid_embedding`` plus its backend tag."""
    vec = extract_reid_embedding(person_crop)
    return vec, (reid_model_name() if vec else "")


def extract_reid_embedding(person_crop: np.ndarray) -> list[float]:
    """
    Extract a normalized appearance embedding from a person/vehicle crop.

    Backend (ML_REID_BACKEND):
      - parts (default): multi-stripe OSNet-inspired handcrafted features
      - osnet / torchreid: Torch OSNet. If OSNet cannot load, returns [] (logged as ERROR) —
        unless ML_REID_ALLOW_HANDCRAFTED_FALLBACK=1, then handcrafted vectors are returned.
      - auto: OSNet when it loads, otherwise handcrafted.
    Never mixes the two spaces silently: once OSNet is loaded a failed crop gives [].
    """
    if _osnet_requested():
        if _load_osnet():
            return _try_osnet_embedding(person_crop)
        if _strict_osnet() and not _allow_handcrafted_fallback():
            return []
    return _part_embedding(person_crop)


def _osnet_preprocess(crop: np.ndarray) -> np.ndarray:
    img = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return rgb.transpose(2, 0, 1)


def extract_reid_embeddings(crops: list[np.ndarray | None]) -> list[list[float]]:
    """Appearance codes for many crops at once (one GPU batch with OSNet). [] for a bad crop."""
    out: list[list[float]] = [[] for _ in crops]
    valid = [i for i, c in enumerate(crops) if c is not None and getattr(c, "size", 0)]
    if not valid:
        return out
    if _osnet_requested():
        # Load first (without consuming a crop), so every crop goes through the same path.
        if not _load_osnet():
            if _strict_osnet() and not _allow_handcrafted_fallback():
                return out
            for i in valid:
                out[i] = _part_embedding(crops[i])
            return out
        if _osnet_model is not None:
            try:
                import torch

                for start in range(0, len(valid), 32):
                    idx = valid[start : start + 32]
                    batch = np.stack([_osnet_preprocess(crops[i]) for i in idx])
                    tensor = torch.from_numpy(batch).to(_osnet_device)
                    with torch.no_grad():
                        feat = _osnet_model(tensor)
                        if isinstance(feat, (tuple, list)):
                            feat = feat[0]
                        feat = torch.nn.functional.normalize(feat, dim=1)
                    for i, vec in zip(idx, feat.detach().cpu().numpy().astype(np.float32)):
                        vec = vec.reshape(-1)
                        vec = np.pad(vec, (0, REID_DIM - vec.size)) if vec.size < REID_DIM else vec[:REID_DIM]
                        out[i] = [float(v) for v in vec]
                return out
            except Exception as exc:
                print(f"[reid] batch failed ({exc}) — per-crop fallback")
                return [extract_reid_embedding(c) if c is not None else [] for c in crops]
    for i in valid:
        out[i] = extract_reid_embedding(crops[i])
    return out
