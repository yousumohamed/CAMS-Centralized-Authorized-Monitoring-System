"""
CAMS – Camera Worker Thread
Each RTSP / local camera stream runs in its own CameraWorker.
Responsibilities:
  • Grab frames from the source
  • Run background subtraction (motion detection)
  • If analytics enabled AND camera is "hot", run face recognition
  • Push processed frames to a shared frame store
"""
import cv2
import threading
import time
import logging
import numpy as np
from datetime import datetime
from config import (
    MOTION_AREA_THRESHOLD, MOTION_DECAY_SECONDS,
    SEARCH_MODE_INTERVAL, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT
)

log = logging.getLogger(__name__)


class CameraWorker(threading.Thread):
    """
    Runs as a daemon thread.  Controlled by CameraManager.
    """

    def __init__(self, camera_id: str, meta: dict, engine, alert_queue, db_writer):
        super().__init__(daemon=True, name=f"cam-{camera_id}")
        self.camera_id   = camera_id
        self.url         = meta["url"]
        self.name_label  = meta["name"]
        self.lat         = meta["lat"]
        self.lon         = meta["lon"]
        self.engine      = engine       # FaceRecognitionEngine
        self.alert_queue = alert_queue  # queue.Queue → Dashboard
        self.db_writer   = db_writer    # callable(camera_id, name, conf, lat, lon, path)

        # State
        self._stop_event      = threading.Event()
        self._analytics_on    = True
        self._lock            = threading.Lock()

        # Shared frame store (latest frame + overlay + metadata)
        self._latest_raw      : np.ndarray | None = None
        self._latest_display  : np.ndarray | None = None
        self._frame_lock      = threading.Lock()

        # Motion tracking
        self._last_motion_ts  = 0.0        # epoch seconds
        self._motion_detected = False
        self._bg_subtractor   = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False
        )

        # Stats
        self.fps             = 0.0
        self.frame_count     = 0
        self.last_seen_ts    : str | None = None
        self.status          = "STARTING"
        self.detections_today = 0

    # ── Control API ──────────────────────────────────────────────────────────
    def stop(self):
        self._stop_event.set()

    def set_analytics(self, enabled: bool):
        with self._lock:
            self._analytics_on = enabled

    def get_analytics(self) -> bool:
        with self._lock:
            return self._analytics_on

    def is_hot(self) -> bool:
        """True if camera has seen motion within MOTION_DECAY_SECONDS."""
        return (time.time() - self._last_motion_ts) < MOTION_DECAY_SECONDS

    # ── Frame access ─────────────────────────────────────────────────────────
    def get_frame(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Returns (raw_frame, display_frame) thread-safely."""
        with self._frame_lock:
            return (
                self._latest_raw.copy()    if self._latest_raw    is not None else None,
                self._latest_display.copy() if self._latest_display is not None else None,
            )

    def get_thumbnail(self) -> np.ndarray | None:
        """Returns a small JPEG-encoded bytes blob for the dashboard grid."""
        with self._frame_lock:
            src = self._latest_display if self._latest_display is not None else self._latest_raw
            if src is None:
                return None
        thumb = cv2.resize(src, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))
        return thumb

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        self.status = "CONNECTING"
        log.info("[%s] Connecting to source: %s", self.camera_id, self.url)
        cap = self._open_capture()
        if cap is None:
            if self.status != "NO SIGNAL":
                self.status = "ERROR"
            return

        self.status = "LIVE"
        fps_timer   = time.time()
        fps_counter = 0
        last_full_process = 0.0

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                log.warning("[%s] Frame read failed -- reconnecting.", self.camera_id)
                cap.release()
                time.sleep(2)
                cap = self._open_capture()
                if cap is None:
                    if self.status != "NO SIGNAL":
                        self.status = "ERROR"
                    break
                continue

            self.frame_count += 1
            fps_counter      += 1

            # FPS calculation (every second)
            now = time.time()
            if now - fps_timer >= 1.0:
                self.fps = round(fps_counter / (now - fps_timer), 1)
                fps_counter = 0
                fps_timer   = now

            # ── Motion detection (always on — cheap) ────────────────────────
            self._run_motion(frame)

            # ── Analytics (face recognition) ────────────────────────────────
            display = frame.copy()
            with self._lock:
                do_analytics = self._analytics_on

            if do_analytics:
                if self.is_hot() or (now - last_full_process) > SEARCH_MODE_INTERVAL:
                    last_full_process = now
                    display = self._run_recognition(frame, display)
            else:
                # Draw "Analytics OFF" watermark
                cv2.putText(display, "Analytics OFF", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            # ── Overlay HUD ─────────────────────────────────────────────────
            display = self._draw_hud(display)

            with self._frame_lock:
                self._latest_raw     = frame
                self._latest_display = display

            self.last_seen_ts = datetime.utcnow().strftime("%H:%M:%S UTC")

        cap.release()
        self.status = "STOPPED"
        log.info("[%s] Worker stopped.", self.camera_id)

    # ── Private helpers ──────────────────────────────────────────────────────
    def _open_capture(self):
        src = self.url
        is_webcam = isinstance(src, int)

        # Webcam indices: try once only — no retry on missing hardware
        if is_webcam:
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                log.info("[%s] Webcam %d opened.", self.camera_id, src)
                return cap
            log.warning("[%s] Webcam index %d not found — marking NO SIGNAL.",
                        self.camera_id, src)
            self.status = "NO SIGNAL"
            return None

        # RTSP / HTTP URLs: retry with exponential backoff
        for attempt in range(5):
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                log.info("[%s] Stream opened (attempt %d).", self.camera_id, attempt + 1)
                return cap
            log.warning("[%s] Attempt %d failed, retrying...", self.camera_id, attempt + 1)
            time.sleep(2 ** attempt)
        return None

    def _run_motion(self, frame: np.ndarray):
        fg_mask = self._bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = any(cv2.contourArea(c) > MOTION_AREA_THRESHOLD for c in contours)
        if motion:
            self._last_motion_ts  = time.time()
            self._motion_detected = True
        else:
            self._motion_detected = False

    def _run_recognition(self, raw: np.ndarray, display: np.ndarray) -> np.ndarray:
        results = self.engine.analyse_frame(raw)
        display = self.engine.draw_results(display, results)
        for r in results:
            if r["is_target"]:
                self.detections_today += 1
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                crop_path = self.engine.save_crop(raw, r["bbox"], self.camera_id, ts)
                self.db_writer(
                    self.camera_id, self.name_label,
                    r["confidence"], self.lat, self.lon,
                    crop_path
                )
                self.alert_queue.put({
                    "camera_id":   self.camera_id,
                    "camera_name": self.name_label,
                    "confidence":  r["confidence"],
                    "lat":         self.lat,
                    "lon":         self.lon,
                    "crop":        crop_path,
                    "timestamp":   ts,
                })
                log.warning(
                    "TARGET FOUND -- [%s] conf=%.1f%%",
                    self.camera_id, r["confidence"]
                )
        return display

    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        # Semi-transparent top bar
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        # Text info
        cv2.putText(frame,
                    f"{self.camera_id} | {self.name_label} | {self.fps} FPS "
                    f"| {'MOTION' if self._motion_detected else '---'}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        # Status dot
        colour = {"LIVE": (0, 220, 0), "ERROR": (0, 0, 255),
                  "CONNECTING": (0, 165, 255)}.get(self.status, (200, 200, 200))
        cv2.circle(frame, (w - 14, 15), 7, colour, -1)
        return frame
