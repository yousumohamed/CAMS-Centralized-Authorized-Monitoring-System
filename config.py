"""
CAMS - Central Configuration
All tunable parameters live here.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
FOUND_DIR       = os.path.join(BASE_DIR, "found_target")
DB_PATH         = os.path.join(BASE_DIR, "cams.db")          # encrypted wrapper
MODEL_DIR       = os.path.join(BASE_DIR, "models")
TARGET_DIR      = os.path.join(BASE_DIR, "target_profile")
LBPH_MODEL_PATH = os.path.join(MODEL_DIR, "lbph_model.xml")

# ── Face-detection DNN model (OpenCV's res10 SSD) ────────────────────────────
FACE_PROTO  = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL  = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# ── Recognition thresholds ───────────────────────────────────────────────────
MATCH_CONFIDENCE_THRESHOLD = 80   # percent — above this triggers alert
LBPH_DISTANCE_THRESHOLD    = 60   # raw LBPH distance; lower = more similar

# ── Motion / search optimisation ─────────────────────────────────────────────
MOTION_DECAY_SECONDS    = 30   # camera stays "hot" this many seconds after motion
MOTION_AREA_THRESHOLD   = 3000 # min pixel-area of motion contour to count
SEARCH_MODE_INTERVAL    = 5    # seconds between processing non-hot cameras

# ── Streams ───────────────────────────────────────────────────────────────────
# Format: {id: {"url": ..., "name": ..., "lat": ..., "lon": ...}}
#
# url can be:
#   • Integer  (0, 1, 2 …)  — local webcam index
#   • String               — RTSP/HTTP stream, e.g.:
#       "rtsp://user:pass@192.168.1.100:554/stream1"
#       "http://localhost:8081/stream"  (rtsp_simulator.py)
#
CAMERA_STREAMS = {
    # ── YOUR LOCAL WEBCAM ──
    "CAM_LOCAL": {"url": 0, "name": "My Local Webcam",  "lat": 0.0, "lon": 0.0},

    # ── PUBLIC CAMERAS LIST ──
    "CAM_01": {"url": "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg", "name": "Heidelberg Uni", "lat": 49.4, "lon": 8.6},
    "CAM_02": {"url": "http://camera.buffalotrace.com/mjpg/video.mjpg", "name": "Blanton Bottling", "lat": 38.2, "lon": -84.8},
    "CAM_03": {"url": "http://camera.butovo.com/mjpg/video.mjpg", "name": "Butovo Moscow", "lat": 55.5, "lon": 37.5},
    "CAM_04": {"url": "http://webcam01.ecn.purdue.edu/mjpg/video.mjpg", "name": "Purdue Engineering", "lat": 40.4, "lon": -86.9},
    "CAM_05": {"url": "http://61.211.241.239/nphMotionJpeg?Resolution=320x240&Quality=Standard", "name": "Tokyo Japan", "lat": 35.6, "lon": 139.6},
    "CAM_06": {"url": "http://vetter.viewnetcam.com:50000/nphMotionJpeg?Resolution=640x480&Quality=Clarity", "name": "Japan Camera", "lat": 35.6, "lon": 139.6},
    "CAM_07": {"url": "http://tamperehacklab.tunk.org:38001/nphMotionJpeg?Resolution=640x480&Quality=Clarity", "name": "Tampere Hacklab", "lat": 61.4, "lon": 23.7},
    "CAM_08": {"url": "http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30", "name": "Osaka Piano", "lat": 34.6, "lon": 135.5},
    "CAM_09": {"url": "http://clausenrc5.viewnetcam.com:50003/nphMotionJpeg?Resolution=320x240", "name": "Richtung West (CH)", "lat": 46.8, "lon": 8.2},
    "CAM_10": {"url": "http://195.196.36.242/mjpg/video.mjpg", "name": "Soltorget Pajala", "lat": 67.2, "lon": 23.3},
    "CAM_11": {"url": "http://honjin1.miemasu.net/nphMotionJpeg?Resolution=640x480&Quality=Standard", "name": "Tsumago Japan", "lat": 35.5, "lon": 137.5},
    "CAM_12": {"url": "http://67.53.46.161:65123/mjpg/video.mjpg", "name": "Mohouli Park (HI)", "lat": 19.7, "lon": -155.0},
    "CAM_13": {"url": "http://webcam.rhein-taunus-krematorium.de/mjpg/video.mjpg", "name": "Krematorium DE", "lat": 50.1, "lon": 8.2},
    "CAM_14": {"url": "http://77.222.181.11:8080/mjpg/video.mjpg", "name": "Kaiskuru Ski", "lat": 69.9, "lon": 23.0},
    "CAM_15": {"url": "http://webcam.mchcares.com/mjpg/video.mjpg?timestamp=1566232173730", "name": "San Bernardino", "lat": 34.1, "lon": -117.3},
    "CAM_16": {"url": "http://47.51.131.147/-wvhttp-01-/GetOneShot?image_size=1280x720&frame_count=1000000000", "name": "Warrenton OR", "lat": 46.1, "lon": -123.9},
}

# ── Encryption passphrase (change before deploy!) ─────────────────────────────
DB_ENCRYPTION_KEY = b"CHANGE_THIS_SECRET_PASSPHRASE_32B"  # exactly 32 bytes

# -- Dashboard -------------------------------------------------------------------
DASHBOARD_TITLE     = "CAMS :: Centralized Authorized Monitoring System"
THUMBNAIL_WIDTH     = 320
THUMBNAIL_HEIGHT    = 240
ALERT_SOUND_PATH    = os.path.join(BASE_DIR, "assets", "alert.wav")  # optional

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
