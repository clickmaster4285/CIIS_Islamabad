# CIIS Peshawar — AI / ML Models Reference

_Last updated: 2026-11-19. Written from a full read of `ml_services/` and `backend/`._

This document lists **every AI/ML model and AI-related algorithm in the system**: what it is, where its
weights live, what it outputs, where it is used, its settings, and how to use or tune it.

> Paths are relative to the repo root (`G:\tekeye\Peshawar`) unless stated.
> "`.env`" = `ml_services/.env` for ML-service settings and `backend/.env` for Django settings.
> Where `.env` and the code default differ, both are shown.

---

## Contents

1. [System overview](#1-system-overview)
2. [Model inventory (quick table)](#2-model-inventory-quick-table)
3. [Camera purposes — which models run on which camera](#3-camera-purposes--which-models-run-on-which-camera)
4. [Object detection models (YOLO)](#4-object-detection-models-yolo)
5. [Number-plate recognition (ANPR)](#5-number-plate-recognition-anpr)
6. [Face recognition — ML service (YuNet + SFace)](#6-face-recognition--ml-service-yunet--sface)
7. [Face recognition — attendance (InsightFace buffalo_l)](#7-face-recognition--attendance-insightface-buffalo_l)
8. [Person / object re-identification (ReID)](#8-person--object-re-identification-reid)
9. [Tracking, identity and motion](#9-tracking-identity-and-motion)
10. [Person journey (cross-camera)](#10-person-journey-cross-camera)
11. [Identity app — galleries, identity engine, clustering](#11-identity-app--galleries-identity-engine-clustering)
12. [Offline video features (Video AI test, Find-in-Video)](#12-offline-video-features)
13. [Voice agent (LLM + Whisper speech-to-text)](#13-voice-agent-llm--whisper-speech-to-text)
14. [Camera health analyzer (rule-based)](#14-camera-health-analyzer-rule-based)
15. [Other: video recovery](#15-other-video-recovery)
16. [GPU / CPU / deployment](#16-gpu--cpu--deployment)
17. [ML service API reference](#17-ml-service-api-reference)
18. [Known issues found during the review](#18-known-issues-found-during-the-review)

---

## 1. System overview

There are two processes that run AI:

| Process | Where | Runs |
|---|---|---|
| **ML service** (FastAPI, port 8100; second GPU node on 8101) | `ml_services/` | YOLO detectors, plate OCR, SFace faces, ReID, ByteTrack, motion clips, person journey, Whisper STT, camera health |
| **Django backend** | `backend/` | InsightFace attendance (CCTV + webcam/kiosk), anti-spoofing, identity galleries & clustering, journey matching, voice-agent LLM |

The backend talks to the ML service over HTTP via `backend/ml/client.py` (node URL = `ML_SERVICE_URL` or the
camera's assigned ML server).

**Live frame pipeline (ML service, per camera):**

```
RTSP ──FFmpeg (NVDEC, scale 1280x720)──► latest-frame buffer
   │
   ├─► every ML_LIVE_INFER_INTERVAL (0.5 s):
   │     1. YOLO26l COCO (classes limited by camera purpose) + SFace on person boxes
   │     2. best.pt custom      (if purpose custom_objects)
   │     3. smoke/fire model    (if purpose smoke_fire)
   │     4. weapon model        (if purpose weapon)
   │     5. plate YOLO + PaddleOCR (if purpose anpr, every 3rd cycle)
   │     6. OSD band filter → purpose hard-gate
   │     7. ByteTrack  → track_id
   │     8. Identity / ReID → global id GP#/GV#/GO#
   │     9. Motion state (moving/stationary) → motion clip recorder
   │    10. publish → /live/cam/{key}/detections  (+ evidence JPEG)
   │
   └─► render thread (12 fps): draw boxes → /live/cam/{key}/mjpeg
```

The backend `cameras/detection_worker.py` polls `/detections` every `DETECTION_WORKER_INTERVAL_SEC` (4 s) to store
detections and alerts.

---

## 2. Model inventory (quick table)

| # | Model | Type | Weights | Runs in | Purpose |
|---|---|---|---|---|---|
| 1 | **YOLO26l COCO** | Object detector (Ultralytics) | `ml_services/runs/train/stage3_finetune3/weights/yolo26l.pt` | ML | People, vehicles, bags, phones, laptops… |
| 2 | **best.pt (custom)** | YOLO26l fine-tuned, 89 classes | `.../weights/best.pt` | ML | face, gun, heavy-weapon, knife_weapon, microbus, pickup-van, License_Plate |
| 3 | **Smoke/Fire** | YOLO26s, 3 classes | `.../weights/best_Smoke_Detection.pt` | ML | fire, smoke |
| 4 | **Weapon** | YOLO26n, 2 classes | `.../weights/best_weapon_detection.pt` | ML | Handgun, Rifle |
| 5 | **Plate detector** | YOLO11s, 1 class | `.../weights/best_number_plate_detection.pt` | ML | Locate number plates |
| 6 | **PaddleOCR PP-OCRv5** | Text det (server) + rec (en mobile) | auto-downloaded by PaddleOCR | ML | Read plate text |
| 7 | **YuNet** | Face detector (OpenCV ONNX) | `ml_services/models/face/face_detection_yunet_2023mar.onnx` | ML | Find faces |
| 8 | **SFace** | Face embedding 128-d (OpenCV ONNX) | `ml_services/models/face/face_recognition_sface_2021dec.onnx` | ML | Recognise staff on live cameras, journey, video search |
| 9 | **OSNet x1_0** | Person ReID 512-d (PyTorch/torchreid) | `ml_services/runs/reid/osnet_x1_0_market1501.pth` | ML | Re-identify people/objects (falls back to handcrafted "parts" vector) |
| 10 | **ByteTrack** | Multi-object tracker (Ultralytics) | — | ML | Stable track IDs per camera |
| 11 | **Motion state + clip recorder** | Rule-based | — | ML | Moving/stationary, motion video clips |
| 12 | **faster-whisper** `small` | Speech-to-text | auto-downloaded (Hugging Face cache) | ML | Voice agent transcription (English/Urdu) |
| 13 | **Camera health analyzer** | OpenCV rules | — | ML | Blur, dark, frozen, glare, plate/face visibility |
| 14 | **InsightFace buffalo_l** | SCRFD detector + ArcFace R50 512-d + 3D landmarks | `~/.insightface/models/buffalo_l/` (auto-download) | Backend | Attendance (CCTV, webcam, kiosk, selfie), visitors |
| 15 | **MiniFASNet V2 + V1SE** | Anti-spoofing (liveness) ONNX | `backend/recognition/weights/` | Backend | Reject photo/screen spoofs — **currently not loading, see §18** |
| 16 | **FAISS IndexFlatIP** | Vector search (optional) | — | ML + Backend | Fast gallery search (not installed → numpy/linear fallback) |
| 17 | **Face BYTE tracker** | Kalman + IoU (hand-written) | — | Backend | Track faces in CCTV attendance |
| 18 | **Identity engine + clustering** | Scoring + union-find clustering | — | Backend | Unknown-person grouping, identity decisions |
| 19 | **LLM** — Claude (`claude-opus-5`) or Ollama (`qwen2.5:14b`) | Large language model | API / local Ollama | Backend | Voice/chat agent ("Hello Customs") |
| 20 | **YOLO26s-pose** | YOLO pose (17 COCO keypoints) | `ml_services/models/pose/yolo26s-pose.pt` | ML | Body points for each detected person (live), runs only when a frame has people |

Weight file sizes (stage3_finetune3): `yolo26l.pt` 53 MB, `best.pt` 106 MB, `best_Smoke_Detection.pt` 20 MB,
`best_weapon_detection.pt` 5.4 MB, `best_number_plate_detection.pt` 19 MB, OSNet 10.4 MB, SFace 38.7 MB, YuNet 232 KB.

---

## 3. Camera purposes — which models run on which camera

Each camera in Django (`backend/cameras/models.py`) has one or more **purposes**. They are pushed to the ML node
via `POST /live/register/bulk`. Only the models for the selected purposes run (`live_stream.py: _purpose_pipeline`).
A final hard-gate (`filter_detections_for_purposes`) drops anything outside the purposes.

| Purpose | Models run | What you get |
|---|---|---|
| `general_objects` (aliases: `object_detection`, `surveillance`, `zone_monitoring`) | YOLO26l COCO | person, bicycle, backpack, handbag, suitcase, bottle, knife, dining table, laptop, cell phone (**no vehicles**) |
| `custom_objects` | best.pt | face, gun, heavy-weapon, knife_weapon, License_Plate, microbus, pickup-van |
| `smoke_fire` (alias `thermal`) | Smoke/Fire model | fire, smoke (alerts) |
| `weapon` | Weapon model | Handgun, Rifle (alerts) |
| `face_recognition` / `attendance` | YOLO26l (person only) + YuNet/SFace | person boxes labelled with staff name or `unknown:U####` |
| `anpr` | YOLO26l (car, motorcycle, bus, truck) + plate YOLO + PaddleOCR | vehicles + plate numbers |

- A camera with **no purpose runs nothing**.
- `ML_PLATE_ON_ALL=true` runs plate reading on every camera (currently `false`).
- Attendance cameras are additionally processed by the **backend** InsightFace worker (§7).

**How to use:** set purposes on the camera in the UI (Cameras → edit). The backend syncs them
(`backend/ml/camera_sync.py`). Ad-hoc override: `GET /live/cam/{key}/detections?purposes=anpr,weapon`.

---

## 4. Object detection models (YOLO)

All YOLO models use **Ultralytics**, load at startup (`warmup_all_models`), run in **FP16 on GPU**, and share one
GPU lock (`_PREDICT_LOCK`) so predictions are serialised. Loader: `ml_services/inference_engine.py`.

### Common live settings (`ml_services/.env`)

| Setting | `.env` | Code default | Meaning |
|---|---|---|---|
| `ML_YOLO_CONF` | 0.30 | 0.12 | Min confidence (also the floor for smoke/weapon in live) |
| `ML_YOLO_IOU` | — | 0.45 | NMS IoU |
| `ML_YOLO_IMGSZ` | 960 | 1280 | Inference size (min 320) |
| `ML_YOLO_MAX_DET` | 300 | 300 | Max boxes per frame |
| `ML_LIVE_INFER_INTERVAL` | 0.50 | 0.15 | Seconds between inferences per camera |
| `ML_LIVE_INFER_WORKERS` | 4 | 5 | Inference threads (cameras are sharded across them) |

### 4.1 YOLO26l COCO — `yolo26l.pt`

- **Override:** `ML_YOLO_WEIGHTS`
- **Classes kept (allowlist, `inference_engine.py:72-89`):** 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus,
  7 truck, 24 backpack, 26 handbag, 28 suitcase, 39 bottle, 43 knife, 60 dining table, 63 laptop, 67 cell phone.
  All other COCO classes are dropped.
- **Priority:** "high" = person, vehicles, bags, phone, laptop, knife; "medium" = bicycle, table, bottle.
- COCO `knife` is **not** an alert.
- **Used by:** live cameras, `/detect/image`, Video AI test, Find-in-Video, person journey (person only), and
  vehicle boxes for ANPR.

### 4.2 Custom model — `best.pt`

- YOLO26l fine-tuned 35 epochs on `data/combined_dataset`, 89 classes (COCO 0–79 + 9 custom).
- **Override:** `ML_YOLO_CUSTOM_WEIGHTS`
- Only custom classes are kept (COCO ids ≤ `ML_COCO_MAX_CLASS_ID`=79 and fire/smoke 84/85 are dropped):
  **80 face, 81 gun, 82 heavy-weapon, 83 knife_weapon, 86 License_Plate, 87 microbus, 88 pickup-van**.
- **Alert classes:** 80, 81, 82, 83 (red box).
- **Used by:** live (purpose `custom_objects`), `/detect/image`.

### 4.3 Smoke / Fire — `best_Smoke_Detection.pt`

- YOLO26s, classes `fire`, `other`, `smoke` (`other` is discarded). All hits are alerts, `model:"smoke"`.
- **Override:** `ML_YOLO_SMOKE_WEIGHTS`; **threshold** `ML_SMOKE_FIRE_CONF` (`.env` 0.25, default 0.42).
  Live effective = max(`ML_YOLO_CONF`, this) = **0.30**.
- Replaces an overlapping (IoU ≥ 0.45) fire/alert box from other models when more confident.
- **Used by:** live (purpose `smoke_fire`), `/detect/image`, Video AI test (`fire=true`).

### 4.4 Weapon — `best_weapon_detection.pt`

- YOLO26n, classes `Handgun`, `Rifle`. All hits are alerts, priority high, `model:"weapon"`.
- **Override:** `ML_YOLO_WEAPON_WEIGHTS`; **threshold** `ML_WEAPON_CONF` (`.env` 0.25, default 0.40). Live effective **0.30**.
- **Used by:** live (purpose `weapon`), `/detect/image`, Video AI test (`weapon=true`), person journey (weapon
  attached to the person carrying it).

### How to use / tune

- Change a model: put the new `.pt` in `runs/train/.../weights/` and point the env var at it; restart the ML service (PM2).
- Fewer false positives: raise `ML_YOLO_CONF` (affects all live models) or the specialist thresholds.
- Faster: lower `ML_YOLO_IMGSZ` (e.g. 640) or raise `ML_LIVE_INFER_INTERVAL`.
- Test one image: `POST /detect/image?conf=0.25&iou=0.45&recognize_faces=true` (multipart `image`) — runs all four models, no purpose gating.

---

## 5. Number-plate recognition (ANPR)

File: `ml_services/plate_recognizer.py` (`PlateEngine`).

### Models

| Stage | Model | Setting |
|---|---|---|
| Detect plate | YOLO11s `best_number_plate_detection.pt` (class `License_Plate`) | `ML_PLATE_WEIGHTS`, `ML_PLATE_IMGSZ` 1280, `ML_PLATE_CONF` 0.30 |
| Read text | PaddleOCR 3.x: `PP-OCRv5_server_det` + `en_PP-OCRv5_mobile_rec` | `ML_PADDLE_DET_MODEL`, `ML_PADDLE_REC_MODEL` |

### Pipeline (`detect_and_read`)

1. Plate YOLO on the full frame.
2. Shape check: width ≤ 28 % of frame, height ≤ 20 %, area ≤ 5 %, aspect 1.2–6.5.
3. Drop plates inside the camera's on-screen-text band (`ML_OSD_*`, top 22 %).
4. Must be near a vehicle box if `ML_PLATE_REQUIRE_VEHICLE=true` (`ML_PLATE_VEHICLE_IOU` 0.02, `ML_PLATE_VEHICLE_EXPAND` 0.35).
5. OCR the crop; if unreadable, retry with CLAHE / sharpen / blur / Otsu variants.
6. Clean text: A–Z/0–9 only, strip region words (PESHAWAR, PUNJAB, ISB…), reject date/time/CAM text,
   length ≥ `ML_PLATE_MIN_LEN` (5).
7. Accept when OCR conf ≥ `ML_PLATE_MIN_OCR_CONF` (0.30) **and** det conf ≥ `ML_PLATE_MIN_DET_CONF` (0.30).

### Output & storage

- Detection fields: `class_name:"license_plate"`, `plate_number`, `ocr_confidence`, `accepted`, `near_vehicle`.
- Saved to `backend/media/licence plates/{plates,frames}/` + `captures.csv` + `numbers.txt`
  (`ML_MEDIA_ROOT` / `ML_PLATE_MEDIA_DIR`).
- **One record per visit:** the same plate on the same camera only updates if the read improves; a new record
  after `ML_PLATE_VISIT_TIMEOUT` (600 s) unseen.

### Usage

- Set camera purpose **`anpr`**. Plates run every `ML_PLATE_OCR_EVERY` (3) inference cycles (~1.5 s).
- One image: `POST /plates/detect` (multipart `image`, params `conf`, `save`, `camera_key`).
- Video: Video AI test with `plate=true`.
- Backend reads plate captures in `backend/cameras/plate_captures.py`.

---

## 6. Face recognition — ML service (YuNet + SFace)

File: `ml_services/face_recognizer.py` (`KnownFaceDB`, shared singleton).

### Models

| Model | Output | Notes |
|---|---|---|
| **YuNet** 2023mar | face box + 5 landmarks + score | score 0.45, NMS 0.3 |
| **SFace** 2021dec | **128-d** embedding, cosine similarity | face aligned with `alignCrop` first |

Both run on **CPU** unless `ML_FACE_USE_CUDA=1` (not set). If the weights are missing they are downloaded from opencv_zoo.

### Matching

- A face matches a staff member when cosine similarity is **above the threshold**.
  `ML_FACE_THRESHOLD=0.32` in `.env`, **but the live engine overrides it to 0.28** (`live_stream.py:1090`) — see §18.
- **FAISS** (`IndexFlatIP`) is used automatically when `faiss` is installed **and** ≥ `ML_FACE_FAISS_MIN_EMBEDDINGS`
  (500) faces are enrolled; top-K `ML_FACE_FAISS_TOP_K` (5). Currently linear search (faiss not installed).
- **Unknown faces** get temporary IDs `unknown:U0001…`; a new unknown matching a recent one above
  `ML_FACE_UNKNOWN_MATCH_THRESHOLD` (0.38) reuses its ID. Kept `ML_FACE_UNKNOWN_TTL_SEC` (600 s), in memory only.
- **Track cache:** identity per (camera, track) re-verified every `ML_FACE_TRACK_REVERIFY_SEC` (30 s) — used by journey.

### Quality gates (code defaults, none set in `.env`)

| | Live / recognition | Enrolment photos |
|---|---|---|
| Min confidence | `ML_FACE_MIN_CONFIDENCE` 0.50 | `ML_FACE_ENROLL_MIN_CONFIDENCE` 0.40 |
| Min face size | `ML_FACE_MIN_SIZE` 100 px | `ML_FACE_ENROLL_MIN_SIZE` 60 px |
| Sharpness (Laplacian) | 50 | 25 |
| Brightness | 35–225 | 25–235 |
| Max yaw / pitch ratio | 0.35 / 0.55 | 0.60 / 0.80 |

### Enrolment flow (staff photos → ML)

1. Staff photos saved in Django → `backend/ml/face_sync.py: enroll_staff_faces`.
2. Each photo → `POST /faces/extract` → 128-d embedding stored on `Staff.face_embeddings`.
3. `push_face_embeddings_to_ml` sends all staff vectors to `POST /reload/faces` on **every** ML node.

### Where used

Live cameras with purpose `face_recognition`/`attendance` (person box labels), person journey, object identity,
Find-in-Video (own CPU copy in `video_search_face.py`), Video AI test (`match_staff=true`).

### Tuning

- Calibrate: `POST /faces/calibrate` with `{genuine_scores:[…], impostor_scores:[…], target_far:0.01}` → suggests
  a threshold (EER and target-FAR). Then change `ML_FACE_THRESHOLD` **and** the hard-coded 0.28 in `live_stream.py`.
- Check what's live: `POST /recognize/face` returns the `threshold` in use; `GET /health` → `face_matching`.

---

## 7. Face recognition — attendance (InsightFace buffalo_l)

The **attendance** system uses a different, stronger face model in the backend. Code: `backend/recognition/services/`.

### Models

| Sub-model (buffalo_l pack) | Role |
|---|---|
| `det_10g.onnx` (SCRFD-10G) | face detection |
| `w600k_r50.onnx` (ArcFace R50) | **512-d** embedding |
| `1k3d68.onnx` | 3D landmarks → head pose (yaw/pitch/roll) |
| `2d106det.onnx`, `genderage.onnx` | loaded but not used for decisions |

- **Setting:** `ATTENDANCE_INSIGHTFACE_MODEL=buffalo_l`. Downloaded automatically to `~/.insightface/models/buffalo_l/`.
- **Runtime:** ONNX Runtime; provider order from `ATTENDANCE_ONNX_PROVIDERS`
  (`.env`: DirectML → CUDA → CPU). Detector input 640×640.
- Singleton: `get_face_engine()` in `face_engine.py`.

### Supporting components

| Component | File | What it does |
|---|---|---|
| **Head pose** | `pose.py` | yaw/pitch from 3D landmarks (fallback: 5 keypoints). Enrol limit yaw 40° / pitch 30°; CCTV `ATTENDANCE_CCTV_MAX_YAW` 60 / `_MAX_PITCH` 55 |
| **Anti-spoof (liveness)** | `antispoof.py` | MiniFASNetV2 + V1SE (80×80, CPU) + FFT moiré check. Webcam ≥ 0.50, CCTV ≥ 0.40, moiré < 0.62. **Not loading — §18** |
| **Quality checker** | `quality_checker.py` | det ≥ 0.5, face area ≥ 2 %, blur ≥ 40, exposure 40–220 |
| **Lighting** | `lighting.py`, `preprocess.py` | gamma/CLAHE variants for training; exposure correction; relight retries |
| **Gallery index** | `gallery_index.py` | FAISS `IndexFlatIP` or numpy; score = mean of top-3 hits per person; ambiguous if runner-up within 0.06; soft-miss ≥ 0.28 goes to HR review |
| **Face tracker** | `byte_tracker.py` | Kalman + IoU; high 0.55, low 0.30, IoU 0.30, lost after 3 scans |
| **Match confirmation** | `match_confirm.py` | webcam: 2 hits in 12 s; CCTV: 3 agreeing votes |

### Thresholds (`backend/.env`)

| Setting | Value |
|---|---|
| `ATTENDANCE_WEBCAM_SIMILARITY_THRESHOLD` | 0.45 |
| `ATTENDANCE_CCTV_SIMILARITY_THRESHOLD` | 0.38 |
| `ATTENDANCE_MIN_ENROLLMENT_IMAGES` | 5 |
| `ATTENDANCE_CCTV_MIN_RECOG_FACE_PX` | 80 |
| `ATTENDANCE_CCTV_MIN_BLUR` | 20 |
| `ATTENDANCE_CAMERA_MARK_COOLDOWN_SECONDS` | 120 |
| `VISITOR_FACE_SIMILARITY_THRESHOLD` | 0.42 |

### Flows

- **Enrol:** Face Enrollment page → capture (`/api/recognition/enroll/<id>/capture/`, quality + pose + liveness) →
  **Train** (`/enroll/<id>/train/`). Each photo is embedded 4× (original, dark gamma, bright gamma, CLAHE); stored as
  `FaceEnrollment.embedding` (mean) + `.embeddings` (all).
- **Kiosk / selfie:** `/api/recognition/identify/`, `/selfie/` → detect → liveness → relight → gallery match.
- **CCTV attendance worker** (`cctv_worker.py`, autostart `ATTENDANCE_CCTV_AUTOSTART=True`, every 0.5 s):
  detect faces → track → quality & pose gate → liveness → upscale small faces → vote over best 5 crops →
  mark attendance when ≥ 3 votes and ≥ 60 % agree, outside the 120 s cooldown. Saves `FaceSample` crops for HR review.

### Management commands (`backend/`)

```bash
python manage.py start_attendance_cctv [--stop]
python manage.py calibrate_attendance_threshold [--target-far 0.0] [--safety 0.02] [--min-probes 30]
python manage.py build_face_gallery [--staff ID] [--keep-mode]
python manage.py retrain_faces [--staff IDS]      # currently broken — use the web Train button (§18)
```

`calibrate_attendance_threshold` uses HR-approved/rejected samples to suggest a threshold; it changes nothing itself.

---

## 8. Person / object re-identification (ReID)

File: `ml_services/reid_extractor.py`. Output is always a **512-d** L2-normalised vector.

| Backend | When | How |
|---|---|---|
| **OSNet x1_0** (Market1501) | `ML_REID_BACKEND=osnet` (set) **and** `torchreid` installed | PyTorch, 128×256 crop, GPU per `ML_DEVICE` |
| **"parts"** (handcrafted) | fallback / default | 6 horizontal stripes × HSV histograms + gradient histograms, padded to 512 |

> `torchreid` is not in `requirements.txt` and not in the local venv → the **parts** vector is what actually runs
> unless the production venv has torchreid installed. Install `torchreid` to activate OSNet.

**Thresholds (object identity):** `ML_REID_MATCH_THRESHOLD` 0.94, `ML_PERSON_REID_MATCH_THRESHOLD` 0.95,
`ML_VEHICLE_REID_MATCH_THRESHOLD` 0.93, EMA update only above `ML_REID_EMA_MIN_SCORE` 0.96.

**Used by:** object identity (global IDs), person journey, Find-in-Video, backend object tracking
(`backend/object_tracking/services.py`, same thresholds hard-coded).

**Endpoint:** `POST /reid/extract` (multipart `image`) → `{embedding, dim}`.

The backend also has its own handcrafted 256-d ReID in `backend/person_journey/reid_utils.py` (unknown-person resolution).

---

## 9. Tracking, identity and motion

### 9.1 ByteTrack (`ml_services/camera_byte_tracker.py`)

- Ultralytics `BYTETracker`, stock `bytetrack.yaml`, `track_buffer=90`.
- One tracker per (camera, class). Every live detection gets a `track_id` (IDs are not stable across restarts).

### 9.2 Object identity (`ml_services/object_identity.py`)

- Assigns global IDs: **`GP#`** person, **`GV#`** vehicle, **`GO#`** other.
- Match order: existing track binding → face identity → ReID cosine against an in-memory gallery (6 h TTL).
- Smoke and weapon detections are excluded from identity — and currently dropped from live output (§18).

### 9.3 Motion state (`ml_services/motion_state.py`) — rule-based

| Setting | Default | Meaning |
|---|---|---|
| `ML_MOTION_ENABLED` | true | on/off |
| `ML_MOTION_MOVE_FRAC` | 0.15 | centre shift (× box size) to leave the resting spot |
| `ML_MOTION_SCALE_FRAC` | 0.30 | or box size change |
| `ML_MOTION_START_SEC` | 1.0 | must stay away this long → "moving" |
| `ML_MOTION_STILL_FRAC` | 0.05 | still-zone while moving |
| `ML_MOTION_STOP_SEC` | 3.0 | still this long → "stationary" |
| `ML_MOTION_TRACK_TTL_SEC` | 10 | forget unseen track ("lost") |
| `ML_MOTION_EXCLUDE_CLASSES` | face,license_plate,smoke,fire,flame | never trigger motion |

### 9.4 Motion clips (`ml_services/motion_clips.py`)

| Setting | Default | Meaning |
|---|---|---|
| `ML_MOTION_CLIP_FPS` | 8 | frames/sec sampled from camera |
| `ML_MOTION_PREROLL_SEC` | 5 | seconds before motion |
| `ML_MOTION_POSTROLL_SEC` | 2 | seconds after last motion |
| `ML_MOTION_CLIP_MIN_SEC` / `_MAX_SEC` | 10 / 30 | clip length (longer motion → part 2, 3…) |
| `ML_MOTION_CLIP_COOLDOWN_SEC` | 0 | gap before next clip |
| `ML_MOTION_CLIP_WIDTH` | 1280 | max width |
| `ML_MOTION_CLIP_CRF` | 28 | H.264 quality (higher = smaller) |
| `ML_MOTION_CLIP_MAX_PENDING_MB` | 2048 | oldest uncollected clips pruned |
| `ML_MOTION_CLIP_DIR` | `ml_services/motion_clips/` | output |
| `ML_MOTION_KEEP_MIN_PCT` | 5 | keep a clip only if an object moved this far A → B (% of frame diagonal); 0 = keep all |
| `ML_MOTION_KEEP_PERSON_PCT` | 3 | same, for people |
| `ML_MOTION_KEEP_SCALE` | 1.5 | or its box grew/shrank this much (walked toward/away from camera) |
| `ML_MOTION_KEEP_ALWAYS_CLASSES` | weapon,gun,pistol,rifle,firearm,knife_weapon,heavy-weapon | always kept, any distance |
| `ML_CLIP_VEHICLE_TRACK_ENABLED` | true | after a clip ends, YOLO vehicle detection on every clip frame: box per frame + dense wheel (ground-point) path |
| `ML_CLIP_POSE_ENABLED` | true | after a clip ends, YOLO pose on every clip frame: skeleton per frame + dense hip path (seeded at end box B, tracked back/forward) |
| `ML_SPECIALIST_EVERY_N` | 3 | custom / smoke-fire / weapon models run every Nth frame per camera (last result reused between); object detection + pose every frame |
| `ML_POSE_ENABLED` | true | YOLO pose for people on/off |
| `ML_POSE_WEIGHTS` | `models/pose/yolo26s-pose.pt` | pose weights |
| `ML_POSE_CONF` | 0.30 | min pose confidence |
| `ML_POSE_MATCH_IOU` | 0.45 | pose box ↔ person box overlap to attach keypoints |
| `ML_POSE_IMGSZ` | 640 | pose inference size (smaller than detection = less GPU) |
| `ML_POSE_DRAW` | true | draw skeleton + posture on live view |
| `ML_MOTION_POSE_MOVE_FRAC` | 0.5 | person "moving" when hips leave rest spot by this × torso length |
| `ML_MOTION_POSE_STILL_FRAC` | 0.15 | still-zone while moving (× torso) |
| `ML_MOTION_TRACK_TTL_FRAMES` | 3 | forget an object only after this many real frame gaps (slow cameras) |
| `ML_MOTION_KEEP_BODY_SHIFT` | 1.0 | keep a person clip when hips moved ≥ this × torso A → B (replaces % rule when pose exists) |
| `ML_MOTION_KEEP_POSTURE_CHANGE` | true | keep a person clip when posture changed (sat down / stood up) |
| `ML_IDENTITY_COUNTER_FILE` | `data/identity_counters.json` | GP/GV/GO counters kept across restarts (codes never repeat) |

Endpoints: `GET /live/motion/status`, `GET /live/motion/clips?limit=`, `GET /live/motion/clips/{id}/{video|poster|end}`,
`DELETE /live/motion/clips/{id}`. The backend (`backend/object_tracking/motion_clips.py`) pulls them into
`MotionClip` / `ObjectMovement` tables → **Movement Clips** page.

---

## 10. Person journey (cross-camera)

Files: `ml_services/journey_manager.py`, `journey_pipeline.py` (detection side); `backend/person_journey/` (matching side).

**ML side, per camera thread:**
1. YOLO26l person detection (class 0) + own ByteTrack.
2. Weapon model — weapons overlapping a person are attached to them.
3. SFace face (`recognize_track`); retry on upscaled head crop.
4. ReID on good crops (height ≥ 96 px, upright, not cut at edges); mean of last 10.
5. POST observation (`face_embedding` 128-d, `reid_embedding` 512-d, label, bbox, weapons) to Django with header
   `X-Journey-Ingest-Token`. First post after 3 ReID samples or 4 s; then every 1.5 s; `finished` after 3 s unseen.

ML settings (defaults): `JOURNEY_CONF` 0.35, `JOURNEY_IOU` 0.5, `JOURNEY_IMGSZ` 640, `JOURNEY_INFER_INTERVAL_SEC` 0.5,
`JOURNEY_REID_MIN_HEIGHT` 96, `JOURNEY_MIN_REID_SAMPLES` 3, `JOURNEY_POST_INTERVAL_SEC` 1.5, `JOURNEY_TRACK_TTL_SEC` 3.

**Backend matching** (`backend/person_journey/matching.py`): score = face 0.5 + ReID 0.3 + travel time 0.2.

| Setting (`backend/.env`) | `.env` | default |
|---|---|---|
| `JOURNEY_FACE_MATCH_THRESHOLD` | 0.72 | 0.72 |
| `JOURNEY_REID_MATCH_THRESHOLD` | 0.68 | 0.80 |
| `JOURNEY_COMBINED_MATCH_THRESHOLD` | 0.75 | 0.70 |

Endpoints: `POST /journey/register/bulk`, `GET /journey/status`, `DELETE /journey/cam/{key}`, `POST /journey/stop-all`
(called by `backend/person_journey/journey_worker.py`).

---

## 11. Identity app — galleries, identity engine, clustering

Code: `backend/identity/`.

- **Two face galleries** (`FaceGalleryEntry`), not comparable with each other:
  `insightface_buffalo_l` (512-d, attendance) and `sface_2021dec` (128-d, live/journey).
  Score = 0.7 × best entry + 0.3 × centroid. Defaults: min quality 0.55, max 60 entries, duplicate ≥ 0.93.
- **Identity engine** (`identity/services/engine.py`): combines face score (0.55), margin (0.15), quality (0.15),
  track-vote consistency (0.15) → ACCEPTED / REVIEW / UNKNOWN. SFace accept 0.36, review band 0.06–0.08.
- **Face quality** (`identity/services/quality.py`): weighted mix of detection score, size, blur, pose, light.
- **Unknown-person clustering** (`identity/services/clustering.py`): union-find; links two unknown observations when
  SFace face cosine ≥ 0.48 **or** ReID cosine ≥ 0.85 (same day, ≤ 1 h apart); never links two people seen side-by-side.
  Clusters with ≥ 5 observations get a suggested employee for a human to approve.

Usage:
```bash
python manage.py build_face_gallery        # backfill both galleries
python manage.py cluster_unknowns          # or POST /api/identity/clusters/run/
```
(The automatic clustering thread exists but is never started.)

---

## 12. Offline video features

### Video AI test (`/video-ai-test`)

- ML: `POST /analyze/video` (form: `video` or `video_path`, `person`, `vehicle`, `weapon`, `fire`, `match_staff`,
  `plate`, `sample_fps` 0.5–4) → poll `GET /analyze/video/{job_id}` → `GET /analyze/video/{job_id}/file` (tagged MP4).
- Runs only the selected models (COCO / weapon / smoke + SFace + plates) at imgsz 416, frames ≤ 960 px.
- One job at a time; **pauses live inference** while running. Max 2 h video.
- Backend: `backend/ml/video_analyze_jobs.py`.

### Find-in-Video (search a video by photo)

- ML: `POST /search/video` (multipart `image` + `video`/`video_path`, `face_threshold` 0.45, `reid_threshold` 0.88,
  `sample_fps` 0=auto, `clip_seconds` 4) → `GET /search/video/{job_id}`.
- The query photo gives a face embedding (SFace) + ReID + object class; people are matched by face then appearance,
  objects by ReID; hits are verified and grouped into clips.
- Settings: `ML_VIDEO_SEARCH_MIN_FACE_SCORE` 0.44, `_MIN_REID_SCORE` 0.86, `_MIN_HITS` 2, `_MAX_DURATION` 3600 s.
- Backend: `backend/ml/video_search_jobs.py`.

---

## 13. Voice agent (LLM + Whisper speech-to-text)

### Speech-to-text — faster-whisper (ML service)

- File: `ml_services/speech_to_text.py`. Model `VOICE_STT_MODEL` (default **`small`**; `medium` = better Urdu).
- `VOICE_STT_DEVICE` auto (CUDA if available, float16) else CPU int8. Beam 5 (GPU) / 1 (CPU). VAD on.
- If detected language is a South-Asian neighbour (hi, pa, …) it re-decodes as **Urdu**.
- Endpoint: `POST /voice/transcribe` (multipart `audio` WAV, optional `language`, `prompt`) →
  `{text, language, confidence, duration}`.
- Backend forwards browser audio from `/api/voice-agent/transcribe` (max 5 MB) with a bias prompt
  "Hello Customs. CIIS, camera, main gate, ANPR, incident." Wake phrase "Hello Customs" (fuzzy ≥ 0.8, English/Urdu).

### LLM (backend `backend/voice_agent/`)

| Provider (`VOICE_AGENT_PROVIDER`) | Model | Notes |
|---|---|---|
| `anthropic` (default) | `VOICE_AGENT_MODEL` default `claude-opus-5` | API key from `ANTHROPIC_API_KEY` env; effort `VOICE_AGENT_EFFORT`=low; prompt caching |
| `ollama` (on-prem) | `VOICE_AGENT_OLLAMA_MODEL` default `qwen2.5:14b` | `VOICE_AGENT_OLLAMA_URL` (127.0.0.1:11434) |

- **Tools the agent can call:** search_cameras, camera_details, search_detections, search_vehicles, search_people,
  person_journey, officer_locations, search_records, generate_report, create_incident, perform_action,
  ui_navigate, ui_open_camera, end_conversation.
- Role permissions enforced in `policy.py`; write actions need spoken confirmation. Max steps: voice 8, chat 14.
- API: `/api/voice-agent/status|sessions|sessions/<id>/turn|transcribe|reports/…`.

---

## 14. Camera health analyzer (rule-based)

Package: `ml_services/cam_health/` — OpenCV measurements, **no learned model**, CPU only, thresholds hard-coded.

- Measures sharpness, brightness, contrast, dark/bright ratios, noise, glare, colour cast, frozen frame (32×32 fingerprint).
- Uses live detections to check purpose visibility: plates readable on ANPR cameras, faces on attendance cameras.
- **Score** = 0.45 image + 0.30 stream + 0.25 visibility → `healthy` ≥ 75, `degraded` ≥ 50, else `critical`; `offline` if no RTSP.
- Alerts: blur, low_light, overexposure, low_contrast, frozen_frame, low_fps, high_latency, glare,
  plate_visibility / plate_too_small, face_visibility.
- Endpoint: `POST /camera-health/analyze` (multipart, optional `image`, `purposes`, `roi_json`, `camera_key`, `fps`…).
  Called by `backend/camera_health/services.py`.

---

## 15. Other: video recovery

`backend/video_recovery/` uses PyTorch only for GPU Gaussian blur / tensor maths plus OpenCV denoise and frame
interpolation — **no trained model**. Settings: `VIDEO_RECOVERY_USE_GPU`, batch 32.

---

## 16. GPU / CPU / deployment

| Component | Device |
|---|---|
| YOLO models, OSNet | GPU `ML_DEVICE` (0), FP16; CPU fallback if no CUDA |
| Plate YOLO | GPU, FP32 |
| PaddleOCR | `gpu:N` from `ML_DEVICE` (needs `paddlepaddle-gpu`) |
| YuNet / SFace | CPU (GPU only with `ML_FACE_USE_CUDA=1`) |
| faster-whisper | `VOICE_STT_DEVICE` (auto) |
| InsightFace / anti-spoof (backend) | ONNX Runtime: DirectML → CUDA → CPU / anti-spoof CPU |
| Camera health | CPU |

- **PM2:** `ml_services/ecosystem.config.js` → `ml-services` on port 8100 (GPU 0);
  `ecosystem.gpu1.config.js` → `ml-services-gpu1` on port 8101 (`CUDA_VISIBLE_DEVICES=1`).
- RTSP decoding uses FFmpeg with NVDEC (`ML_RTSP_NVDEC=true`) scaled to 1280×720.
- All YOLO predictions share one GPU lock — adding inference workers does not parallelise GPU work.
- Restart after changing weights or `.env`: `pm2 restart ml-services` (and `ml-services-gpu1`).

---

## 17. ML service API reference

| Endpoint | Method | Model(s) | Called by (backend) |
|---|---|---|---|
| `/health` | GET | status of all models | `ml/views.py` |
| `/detect/image` | POST | all 4 YOLO + SFace | `ml/client.py`, `cameras/views.py` |
| `/plates/detect` | POST | plate YOLO + PaddleOCR | (client exists, no caller) |
| `/faces/extract` | POST | YuNet + SFace | `ml/face_sync.py` |
| `/reload/faces` | POST | SFace gallery | `ml/face_sync.py`, identity gallery |
| `/recognize/face` | POST | YuNet + SFace | (client exists, no caller) |
| `/validate/human-face` | POST | YuNet quality gate | `ml/views.py` |
| `/faces/calibrate` | POST | threshold calculator | manual |
| `/reid/extract` | POST | OSNet / parts | manual |
| `/live/register/bulk` | POST | sets camera purposes | `ml/camera_sync.py`, `ops_central` |
| `/live/cam/{key}/detections` | GET | live pipeline output | `cameras/detection_worker.py`, `person_journey`, `camera_health` |
| `/live/cam/{key}/mjpeg`, `/jpeg`, `/jpeg/evidence`, `/mjpeg/raw`, `/mjpeg/attendance` | GET | annotated/raw video | cameras, recognition, warehouse, ops_central |
| `/live/status`, `/live/timing` | GET | performance | manual |
| `/live/motion/status`, `/live/motion/clips…` | GET/DELETE | motion clips | `object_tracking/motion_clips.py` |
| `/journey/register/bulk`, `/journey/status`, `/journey/stop-all` | POST/GET | journey pipeline | `person_journey/journey_worker.py` |
| `/analyze/video`, `/analyze/video/{id}[/file]` | POST/GET | Video AI test | `ml/video_analyze_jobs.py` |
| `/search/video`, `/search/video/{id}` | POST/GET | Find-in-Video | `ml/video_search_jobs.py` |
| `/voice/transcribe` | POST | faster-whisper | `voice_agent/views.py` |
| `/camera-health/analyze` | POST | camera health | `camera_health/services.py` |

---

## 18. Known issues found during the review

These were found while reading the code; they are **not fixed** yet.

| # | Severity | Issue | Where | Effect |
|---|---|---|---|---|
| 1 | ~~High~~ **Fixed 2026-11-19** | Live smoke/fire and weapon detections are dropped. The non-identity branch in `enrich_detections` repeats the same exclusion checks, so it always `continue`s and `out.append` is unreachable. | `ml_services/object_identity.py:299-307` | Weapon/fire/smoke (and best.pt `gun`) never reach live detections, overlays or alerts. Video AI test and `/detect/image` are unaffected. |
| 2 | **High** | Anti-spoof model filename mismatch: code expects `4.0_80x80_MiniFASNetV1SE.onnx`, disk has `4_0_0_80x80_MiniFASNetV1SE.onnx`. | `backend/recognition/services/antispoof.py:18-21` | Liveness never runs (only moiré check). Rename the file to fix. |
| 3 | Medium | Live face threshold hard-coded to 0.28, overriding `ML_FACE_THRESHOLD=0.32` for the whole ML process. | `ml_services/live_stream.py:1090, 1447` | Looser face matching than configured. |
| 4 | Medium | `retrain_faces` command crashes (unpacks a list as a tuple). | `backend/recognition/services/training.py:34` | Use the web Train button instead. |
| 5 | ~~Medium~~ **Fixed 2026-11-19** (bundled `ml_services/osnet.py`, strict-loads the checkpoint; ReID thresholds recalibrated to 0.85 on live data) | `torchreid` not installed → OSNet never loads despite `ML_REID_BACKEND=osnet`. | `reid_extractor.py`, `requirements.txt` | Handcrafted ReID used; weaker re-identification. |
| 6 | Low | FAISS not installed (ML and backend). | `requirements.txt` | Exact linear search — same results, slower at large gallery sizes. |
| 7 | Low | Enrolment quality not sent to ML `/reload/faces`, so `ML_FACE_ENROLLMENT_MIN_QUALITY` has no effect. | `backend/ml/client.py:144-147` | Low-quality enrolment photos are used for matching. |
| 8 | Low | Visitor matching may compare 128-d SFace with 512-d InsightFace vectors (truncated). | `backend/person_journey/live_ingest.py:137`, `visitors/face_gallery.py:52-66` | Visitor matches on journey may be meaningless (uncertain). |
| 9 | Low | Live floor `max(ML_YOLO_CONF, specialist)` makes smoke/weapon 0.25 effectively 0.30. | `live_stream.py:1722` | Slightly fewer specialist detections than configured. |
| 10 | Low | best.pt `face` (class 80) is flagged as an alert (red box). | `inference_engine.py:54` | Custom-object cameras show faces as alerts. |
| 11 | Info | Unused: `track_snapshot.py`, `data/face_embeddings_cache.json`, clustering auto-thread, many `.env` keys (e.g. `ML_INFER_MAX_WIDTH`, `ML_PLATE_RECO_CONF`, `ML_RTSP_FRAME_BUFFER_SIZE`, `ML_PLATE_SAVE_INTERVAL`). | — | Clean-up candidates. |
| 12 | ~~Info~~ **Fixed 2026-11-19** (real-movement + pose filter) | Motion clips volume: ~224 clips/hour, ~12 GB/day, mostly seated people fidgeting. | `motion_state.py` | Tune `ML_MOTION_MOVE_FRAC` / add net-movement filter (pending decision). |

### OSD filter fix (2026-11-19)
`filter_osd_detections` used the box **centre** and an overlapping left/right corner guard (`ML_OSD_LEFT`=`ML_OSD_RIGHT`=0.55 → whole width), so every detection with centre in the top ~30% of the frame was deleted. On the gate camera (cam-35) that removed 127 of 227 vehicle boxes (56%) — the road beyond the gate. Now a box is dropped only when it lies **entirely** inside the top strip, or is a small text-sized box in a corner just below it.
