"""Offline video AI test: person / vehicle / weapon / fire / licence plate with staff-name tags."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Callable

import cv2

from face_recognizer import is_unknown_label
from inference_engine import (
    ALLOWED_COCO_CLASS_IDS,
    SMOKE_FIRE_MIN_CONF,
    WEAPON_MIN_CONF,
    get_face_db,
    get_yolo_coco_model,
    get_yolo_smoke_model,
    get_yolo_weapon_model,
    is_fire_smoke_class,
    parse_yolo_result,
    use_gpu_half,
    _predict_model,
)
from live_stream import draw_detections

PERSON_CLASSES = frozenset({"person", "face"})
VEHICLE_CLASSES = frozenset(
    {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "vehicle",
        "microbus",
        "pickup-van",
        "pickup_van",
    }
)
WEAPON_CLASSES = frozenset(
    {
        "weapon",
        "gun",
        "handgun",
        "pistol",
        "rifle",
        "firearm",
        "knife",
        "knife_weapon",
        "sword",
        "machete",
        "heavy-weapon",
        "heavy_weapon",
        "heavyweapon",
    }
)
GENERIC_PERSON = frozenset({"person", "face", "unknown", ""})

MAX_DURATION_SEC = float(os.getenv("ML_VIDEO_ANALYZE_MAX_DURATION", "7200"))
DETECT_IMG_SIZE = int(os.getenv("ML_VIDEO_ANALYZE_IMG_SIZE", "416"))
OUTPUT_MAX_WIDTH = int(os.getenv("ML_VIDEO_ANALYZE_MAX_WIDTH", "960"))
# Offline work shares the GPU with live cameras (live inference is NOT paused): after every
# sampled frame yield briefly, and after each batch of BATCH_FRAMES samples pause longer.
YIELD_SEC = max(0.0, float(os.getenv("ML_VIDEO_ANALYZE_YIELD_SEC", "0.02")))
BATCH_FRAMES = max(1, int(os.getenv("ML_VIDEO_ANALYZE_BATCH_FRAMES", "8")))
BATCH_PAUSE_SEC = max(0.0, float(os.getenv("ML_VIDEO_ANALYZE_BATCH_PAUSE_SEC", "0.25")))
_PERSON_COCO_IDS = frozenset({0})
_VEHICLE_COCO_IDS = frozenset({2, 3, 5, 7, 1})


def _norm(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def _is_weapon(det: dict[str, Any]) -> bool:
    tag = str(det.get("model") or det.get("model_tag") or "").strip().lower()
    if tag == "weapon":
        return True
    cls = _norm(det.get("class_name") or "")
    label = _norm(det.get("label") or "")
    if cls in WEAPON_CLASSES or label in WEAPON_CLASSES:
        return True
    return any(token in cls or token in label for token in ("gun", "rifle", "pistol", "firearm", "weapon"))


def _is_fire(det: dict[str, Any]) -> bool:
    tag = str(det.get("model") or det.get("model_tag") or "").strip().lower()
    if tag == "smoke":
        return True
    cls = str(det.get("class_name") or det.get("label") or "")
    return is_fire_smoke_class(cls)


def _is_person(det: dict[str, Any]) -> bool:
    cls = _norm(det.get("class_name") or "")
    label = _norm(det.get("label") or "")
    return cls in PERSON_CLASSES or label in PERSON_CLASSES


def _is_vehicle(det: dict[str, Any]) -> bool:
    cls = _norm(det.get("class_name") or "")
    label = _norm(det.get("label") or "")
    return cls in VEHICLE_CLASSES or label in VEHICLE_CLASSES


def _is_known_staff(det: dict[str, Any]) -> bool:
    label = str(det.get("label") or "").strip()
    cls = _norm(det.get("class_name") or "")
    if not label or is_unknown_label(label):  # "unknown", "unknown:U0007", "person", "face"
        return False
    if _norm(label) in GENERIC_PERSON:
        return False
    return cls in PERSON_CLASSES or bool(label)


class OfflineThrottle:
    """Cooperative GPU sharing for offline jobs: short sleep per sample, longer per batch."""

    def __init__(
        self,
        *,
        yield_sec: float = YIELD_SEC,
        batch_frames: int = BATCH_FRAMES,
        batch_pause_sec: float = BATCH_PAUSE_SEC,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.yield_sec = max(0.0, float(yield_sec))
        self.batch_frames = max(1, int(batch_frames))
        self.batch_pause_sec = max(0.0, float(batch_pause_sec))
        self._sleep = sleep
        self._count = 0

    def step(self) -> None:
        self._count += 1
        if self._count % self.batch_frames == 0:
            if self.batch_pause_sec > 0:
                self._sleep(self.batch_pause_sec)
        elif self.yield_sec > 0:
            self._sleep(self.yield_sec)


class _OfflineFaceDB:
    """Shared gallery, but no live temporary-unknown IDs: offline video must not mint or touch
    the live ``unknown:Uxxxx`` cache (it would leak foreign faces into live tracking)."""

    def __init__(self, db):
        self._db = db

    def label_detection_detail(self, frame, cls_id, yolo_name, bbox):
        return self._db.label_detection_detail(frame, cls_id, yolo_name, bbox, use_unknown_cache=False)


def _shrink(frame, max_w: int = OUTPUT_MAX_WIDTH):
    h, w = frame.shape[:2]
    if w <= max_w or max_w <= 0:
        return frame
    scale = max_w / float(w)
    nw = max(2, int(w * scale) // 2 * 2)
    nh = max(2, int(h * scale) // 2 * 2)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def detect_selected(
    frame,
    *,
    person: bool,
    vehicle: bool,
    weapon: bool,
    fire: bool,
    match_staff: bool,
    img_size: int = DETECT_IMG_SIZE,
) -> list[dict[str, Any]]:
    """Run only the models the user selected (much faster than the full live stack)."""
    half = use_gpu_half()
    detections: list[dict[str, Any]] = []
    if person or vehicle:
        coco = get_yolo_coco_model()
        if coco is not None:
            classes: set[int] = set()
            if person:
                classes |= _PERSON_COCO_IDS
            if vehicle:
                classes |= _VEHICLE_COCO_IDS
            class_ids = sorted(cid for cid in classes if cid in ALLOWED_COCO_CLASS_IDS)
            face_db = _OfflineFaceDB(get_face_db()) if person and match_staff else None
            for result in _predict_model(
                coco,
                frame,
                conf=0.25,
                iou=0.45,
                img_size=img_size,
                half=half,
                max_det=50,
                classes=class_ids or None,
            ):
                detections.extend(
                    parse_yolo_result(
                        frame,
                        result,
                        recognize_faces=bool(person and match_staff),
                        smoke_model=False,
                        model_tag="coco",
                        face_db=face_db,
                    )
                )
    if weapon:
        weapon_model = get_yolo_weapon_model()
        if weapon_model is not None:
            for result in _predict_model(
                weapon_model,
                frame,
                conf=WEAPON_MIN_CONF,
                iou=0.45,
                img_size=img_size,
                half=half,
                max_det=50,
            ):
                detections.extend(
                    parse_yolo_result(
                        frame,
                        result,
                        recognize_faces=False,
                        weapon_model=True,
                        model_tag="weapon",
                        face_db=None,
                    )
                )
    if fire:
        smoke_model = get_yolo_smoke_model()
        if smoke_model is not None:
            for result in _predict_model(
                smoke_model,
                frame,
                conf=SMOKE_FIRE_MIN_CONF,
                iou=0.45,
                img_size=img_size,
                half=half,
                max_det=50,
            ):
                detections.extend(
                    parse_yolo_result(
                        frame,
                        result,
                        recognize_faces=False,
                        smoke_model=True,
                        model_tag="smoke",
                        face_db=None,
                    )
                )
    return detections


def filter_detections(
    detections: list[dict[str, Any]],
    *,
    person: bool,
    vehicle: bool,
    weapon: bool,
    fire: bool,
    match_staff: bool,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for det in detections or []:
        work = dict(det)
        keep = False
        if person and _is_person(work):
            keep = True
            if not match_staff:
                work["label"] = str(work.get("class_name") or "person")
        if vehicle and _is_vehicle(work):
            keep = True
        if weapon and _is_weapon(work):
            keep = True
            work["alert"] = True
        if fire and _is_fire(work):
            keep = True
            work["alert"] = True
        if keep:
            kept.append(work)
    return kept


def _hit_kind(det: dict[str, Any]) -> str:
    if _is_weapon(det):
        return "weapon"
    if _is_fire(det):
        return "fire"
    if _is_vehicle(det):
        return "vehicle"
    if _is_known_staff(det):
        return "known"
    if _is_person(det):
        return "unknown"
    return "other"


def _crop_data_url(crop, height: int = 64) -> str:
    """Small JPEG thumbnail of a plate crop, inlined so the UI can show what was read."""
    try:
        if crop is None or crop.size == 0:
            return ""
        h, w = crop.shape[:2]
        if h > height:
            scale = height / float(h)
            crop = cv2.resize(crop, (max(1, int(w * scale)), height), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return ""
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


def _read_plates(
    full_frame,
    small_frame,
    raw_small_detections: list[dict[str, Any]],
    engine,
) -> list[dict[str, Any]]:
    """Detect + OCR plates on the full-size frame; return boxes scaled to the small frame.

    Nothing is saved to the live licence-plate captures (save=False).
    """
    fh, fw = full_frame.shape[:2]
    sh, sw = small_frame.shape[:2]
    sx = fw / float(sw or 1)
    sy = fh / float(sh or 1)
    vehicle_boxes: list[list[float]] = []
    for det in raw_small_detections:
        if not _is_vehicle(det):
            continue
        bbox = det.get("bbox") or []
        if len(bbox) >= 4:
            vehicle_boxes.append(
                [float(bbox[0]) * sx, float(bbox[1]) * sy, float(bbox[2]) * sx, float(bbox[3]) * sy]
            )
    try:
        plates = engine.detect_and_read(full_frame, save=False, vehicle_boxes=vehicle_boxes)
    except Exception as exc:
        print(f"[video-ai] plate read failed: {exc}", flush=True)
        return []
    out: list[dict[str, Any]] = []
    for det in plates:
        text = str(det.get("plate_number") or "").strip()
        bbox = det.get("bbox") or []
        if not text or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        crop = full_frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        work = dict(det)
        work["label"] = text
        work["bbox"] = [int(x1 / sx), int(y1 / sy), int(x2 / sx), int(y2 / sy)]
        work["_crop"] = crop
        out.append(work)
    return out


def _open_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter | None:
    fps = max(1.0, float(fps) or 1.0)
    for fourcc_name in ("mp4v", "avc1", "XVID"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_name), fps, (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    return None


def analyze_video_path(
    video_path: str,
    output_path: str,
    *,
    person: bool = True,
    vehicle: bool = True,
    weapon: bool = True,
    fire: bool = True,
    match_staff: bool = True,
    plate: bool = False,
    sample_fps: float = 0.5,
    progress_cb: Callable[[int, str], None] | None = None,
    throttle: OfflineThrottle | None = None,
) -> dict[str, Any]:
    if not any([person, vehicle, weapon, fire, plate]):
        raise ValueError("Select at least one detection type.")
    plate_engine = None
    if plate:
        from plate_recognizer import (
            canonicalize_plate,
            get_plate_engine,
            plate_key,
            plates_are_same_vehicle,
        )

        plate_engine = get_plate_engine()
        if not plate_engine.available:
            print("[video-ai] plate engine not available — plates skipped", flush=True)
            plate_engine = None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (frame_count / src_fps) if frame_count > 0 else 0.0
    if duration > MAX_DURATION_SEC:
        cap.release()
        raise ValueError(
            f"Video is too long ({duration / 60:.1f} min). Maximum is {MAX_DURATION_SEC / 60:.0f} minutes."
        )
    if width < 32 or height < 32:
        cap.release()
        raise ValueError("Video has an invalid frame size.")

    out_fps = max(0.5, min(float(sample_fps), 4.0))
    detect_every = max(1, int(round(src_fps / out_fps)))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report_frames = max(1, int(frame_count / detect_every) if frame_count > 0 else 1)

    hits: list[dict[str, Any]] = []
    frames_written = 0
    index = 0
    scanned = 0
    writer: cv2.VideoWriter | None = None
    known_staff: set[str] = set()
    counts = {"known": 0, "unknown": 0, "vehicle": 0, "weapon": 0, "fire": 0}
    # Unique plates: canonical key → best read across the whole video.
    unique_plates: dict[str, dict[str, Any]] = {}

    def remember_plate(det: dict[str, Any], t: float) -> None:
        text = str(det.get("plate_number") or det.get("label") or "")
        key = canonicalize_plate(text) or plate_key(text)
        if not key:
            return
        ocr = float(det.get("ocr_confidence") or 0)
        slot = None
        for existing_key, row in unique_plates.items():
            if plates_are_same_vehicle(key, existing_key, ocr, float(row.get("ocr_confidence") or 0)):
                slot = existing_key
                break
        if slot is None:
            unique_plates[key] = {
                "plate_number": text,
                "first_t": round(t, 2),
                "last_t": round(t, 2),
                "reads": 1,
                "ocr_confidence": round(ocr, 4),
                "confidence": det.get("confidence"),
                "image": _crop_data_url(det.get("_crop")),
            }
            return
        row = unique_plates[slot]
        row["reads"] += 1
        row["last_t"] = round(t, 2)
        if ocr > float(row.get("ocr_confidence") or 0):
            row["plate_number"] = text
            row["ocr_confidence"] = round(ocr, 4)
            row["confidence"] = det.get("confidence")
            row["image"] = _crop_data_url(det.get("_crop")) or row.get("image", "")

    def report(pct: int, message: str) -> None:
        if progress_cb:
            progress_cb(max(1, min(99, pct)), message)

    if throttle is None:
        throttle = OfflineThrottle()
    report(4, "Opening video")
    try:
        while True:
            if index % detect_every != 0:
                if not cap.grab():
                    break
                index += 1
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            small = _shrink(frame)
            if writer is None:
                oh, ow = small.shape[:2]
                writer = _open_writer(output_path, out_fps, ow, oh)
                if writer is None:
                    raise ValueError("Could not create the tagged video file.")
            report(5 + int(90 * scanned / report_frames), f"Detecting sample {scanned + 1}…")
            do_faces = bool(match_staff and person and (scanned % 2 == 0))
            # Plates are only kept when they sit on a vehicle, so vehicles are always
            # detected when plates are on (but only tagged if Vehicle was ticked).
            raw = detect_selected(
                small,
                person=person,
                vehicle=vehicle or plate_engine is not None,
                weapon=weapon,
                fire=fire,
                match_staff=do_faces,
            )
            boxes = filter_detections(
                raw,
                person=person,
                vehicle=vehicle,
                weapon=weapon,
                fire=fire,
                match_staff=match_staff,
            )
            t = index / src_fps
            for det in boxes:
                kind = _hit_kind(det)
                label = str(det.get("label") or det.get("class_name") or kind)
                if kind == "known":
                    known_staff.add(label)
                if kind in counts:
                    counts[kind] += 1
                hits.append(
                    {
                        "t": round(t, 2),
                        "kind": kind,
                        "class_name": det.get("class_name"),
                        "label": label,
                        "model": det.get("model"),
                        "confidence": det.get("confidence"),
                        "bbox": det.get("bbox") or [],
                    }
                )
            plate_boxes: list[dict[str, Any]] = []
            if plate_engine is not None:
                plate_boxes = _read_plates(frame, small, raw, plate_engine)
                for det in plate_boxes:
                    remember_plate(det, t)
                    hits.append(
                        {
                            "t": round(t, 2),
                            "kind": "plate",
                            "class_name": "license_plate",
                            "label": str(det.get("plate_number") or ""),
                            "model": "plate",
                            "confidence": det.get("ocr_confidence"),
                            "bbox": det.get("bbox") or [],
                        }
                    )
            drawn = boxes + [{k: v for k, v in d.items() if k != "_crop"} for d in plate_boxes]
            tagged = draw_detections(small, drawn) if drawn else small
            writer.write(tagged)
            frames_written += 1
            scanned += 1
            index += 1
            throttle.step()  # let live cameras use the GPU between samples
            report(
                5 + int(90 * scanned / report_frames),
                f"Tagging {scanned} samples ({t:.0f}s / {duration:.0f}s)",
            )
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames_written <= 0:
        raise ValueError("No frames could be read from the video.")

    report(99, "Finishing tagged video")
    return {
        "duration_sec": round(duration or (frames_written / out_fps), 2),
        "fps": round(out_fps, 2),
        "frames_written": frames_written,
        "frames_scanned": scanned,
        "sample_fps": round(out_fps, 2),
        "hit_count": len(hits),
        "known_staff": sorted(known_staff),
        "unknown_people": counts["unknown"],
        "vehicles": counts["vehicle"],
        "weapons": counts["weapon"],
        "fires": counts["fire"],
        "plate_requested": bool(plate),
        "plate_engine_available": plate_engine is not None,
        "plates": sorted(unique_plates.values(), key=lambda r: float(r["first_t"]))[:200],
        "hits": hits[:2000],
        "output_name": Path(output_path).name,
    }
