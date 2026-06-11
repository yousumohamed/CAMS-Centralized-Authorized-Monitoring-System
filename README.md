# CAMS — Centralized Authorized Monitoring System

A high-scale Python/OpenCV face recognition security monitoring system.

## Screenshot

![CAMS dashboard screenshot](./Screenshot%202026-06-11%20111233.png)

---

## Quick-Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download OpenCV DNN models (~10 MB, one time only)
python setup_models.py

# 3. Place target photos in ./target_profile/, then train the model
python train.py

# 4. Launch
python main.py
```

---

## Architecture

```
main.py
 ├── CameraManager (manager.py)
 │    ├── CameraWorker × N  (camera_worker.py)  ← one thread per stream
 │    │    ├── MOG2 motion detection (every frame, cheap)
 │    │    └── FaceRecognitionEngine (recognition.py)
 │    │         ├── DNN SSD detector (res10 caffemodel)
 │    │         └── LBPH recogniser (lbph_model.xml)
 │    └── alert_queue  →  Dashboard
 └── Dashboard (dashboard.py)
      ├── Thumbnail grid  (click to select)
      ├── Fullscreen view  [F]
      ├── Analytics toggle [A] per-cam, [G] global
      └── Alert panel
```

---

## Key Configuration (config.py)

| Setting | Default | Purpose |
|---|---|---|
| `CAMERA_STREAMS` | 2 webcams + 2 demo RTSP | Add your real RTSP URLs here |
| `MATCH_CONFIDENCE_THRESHOLD` | 80 % | Minimum to trigger an alert |
| `LBPH_DISTANCE_THRESHOLD` | 60 | LBPH distance mapped to 100 % confidence |
| `MOTION_DECAY_SECONDS` | 30 s | How long a camera stays "hot" |
| `MOTION_AREA_THRESHOLD` | 3000 px² | Minimum motion contour size |
| `SEARCH_MODE_INTERVAL` | 5 s | Recognition interval for cold cameras |
| `DB_ENCRYPTION_KEY` | placeholder | **Change before first use** |

---

## Dashboard Keyboard Shortcuts

| Key | Action |
|---|---|
| Click thumbnail | Select camera |
| **F** | Toggle fullscreen |
| **A** | Toggle analytics on selected cam |
| **G** | Toggle analytics globally |
| **R** | Restart selected worker |
| **S** | Print status snapshot to terminal |
| **Q / ESC** | Quit |

---

## Face Recognition Pipeline

1. **DNN SSD** detector extracts face bounding boxes (OpenCV res10 model, CPU-only)
2. ROI → greyscale → resized to 100×100
3. **LBPH** recogniser predicts `(label, distance)`
4. `confidence = clamp(1 - distance / LBPH_DISTANCE_THRESHOLD) × 100`
5. If `label == TARGET` and `confidence ≥ 80%`:
   - Crop saved → `found_target/`
   - Row written to encrypted SQLite
   - Alert pushed to dashboard queue

---

## Search Mode Optimisation

Background subtraction (MOG2) runs every frame. When motion is detected:
- Camera marked **"hot"** for `MOTION_DECAY_SECONDS` seconds
- **Hot cameras**: recognition on every frame
- **Cold cameras**: recognition throttled to once per `SEARCH_MODE_INTERVAL` seconds

On a 32-camera fleet with 2 active, CPU usage scales with **2**, not 32.

---

## Security

| Surface | Protection |
|---|---|
| GPS & metadata at rest | Fernet AES-128-CBC (SHA-256 key derivation) |
| RTSP stream credentials | In `config.py` — never commit to version control |
| Crop images | `found_target/` — restrict with OS file ACLs |
| SQLite file | `cams.db` — encrypted fields, restrict file permissions |

---

## Database Viewer

```bash
python db_viewer.py detections --limit 20   # show recent alerts
python db_viewer.py system     --limit 50   # show system events
python db_viewer.py stats                   # per-camera statistics
```

---

## File Layout

```
cams/
├── main.py
├── manager.py          # Manager-Worker controller
├── camera_worker.py    # Per-stream thread
├── recognition.py      # DNN + LBPH engine
├── dashboard.py        # OpenCV GUI
├── db.py               # Encrypted DB layer
├── db_viewer.py        # Log inspection CLI
├── train.py            # LBPH training CLI
├── setup_models.py     # DNN model downloader
├── config.py           # All settings
├── requirements.txt
├── models/             # DNN + LBPH model files
├── target_profile/     # ← place target photos here
│   └── negatives/      # optional false-positive reduction
└── found_target/       # auto-saved matched face crops
```

---

## Extensions

| Goal | How |
|---|---|
| GPU speed | `net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)` in `recognition.py` |
| Multiple targets | Add more LBPH labels (1=Mother, 2=Father, …) |
| SMS/email alert | Second daemon thread consuming `alert_queue` |
| Web view | FastAPI serving MJPEG + `manager.get_status_snapshot()` |
| 30s clip on match | Ring-buffer in `CameraWorker`, flush on target match |
