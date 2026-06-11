"""
CAMS – Ring-Buffer Clip Recorder
Every CameraWorker can be attached to a ClipRecorder that:

  1. Continuously buffers the last N seconds of raw frames in memory.
  2. When a target detection alert fires, flushes the ring buffer to disk as
     an MP4 clip PLUS continues recording for CLIP_POST_SECONDS seconds
     to capture what happens right after detection.

Usage (called from main.py after manager.start_all()):

    from stream_recorder import attach_recorders
    attach_recorders(manager)
"""
import cv2
import threading
import time
import os
import collections
import logging
from datetime import datetime
from config import FOUND_DIR

log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
CLIP_PRE_SECONDS  = 15    # seconds of pre-event footage to save
CLIP_POST_SECONDS = 10    # seconds to continue recording after event
CLIP_FPS          = 15    # target output FPS
CLIP_RESOLUTION   = (640, 480)
CLIPS_DIR         = os.path.join(os.path.dirname(__file__), "clips")


class ClipRecorder(threading.Thread):
    """
    Daemon thread attached to ONE CameraWorker.
    Maintains a ring buffer and writes MP4 clips on demand.
    """

    def __init__(self, worker):
        super().__init__(daemon=True, name=f"rec-{worker.camera_id}")
        self.worker      = worker
        self._buf_size   = CLIP_PRE_SECONDS * CLIP_FPS
        self._buffer     : collections.deque = collections.deque(maxlen=self._buf_size)
        self._recording  = False
        self._post_frames_left = 0
        self._writer     : cv2.VideoWriter | None = None
        self._clip_path  = ""
        os.makedirs(CLIPS_DIR, exist_ok=True)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        interval = 1.0 / CLIP_FPS
        while True:
            start = time.time()
            raw, _ = self.worker.get_frame()
            if raw is not None:
                frame = cv2.resize(raw, CLIP_RESOLUTION)
                self._buffer.append(frame)
                if self._recording:
                    self._write_frame(frame)

            elapsed = time.time() - start
            time.sleep(max(0, interval - elapsed))

    # ── Trigger API ───────────────────────────────────────────────────────────
    def trigger(self, alert: dict):
        """
        Called when a target detection is confirmed for this camera.
        Flushes the pre-event buffer and starts recording post-event frames.
        """
        if self._recording:
            # Extend post-event window if re-triggered
            self._post_frames_left = CLIP_POST_SECONDS * CLIP_FPS
            return

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._clip_path = os.path.join(
            CLIPS_DIR, f"{self.worker.camera_id}_{ts}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self._clip_path, fourcc, CLIP_FPS, CLIP_RESOLUTION
        )
        if not self._writer.isOpened():
            log.error("[%s] Could not open VideoWriter for clip.", self.worker.camera_id)
            self._writer = None
            return

        # Flush ring buffer (pre-event footage)
        log.info(
            "[%s] Saving clip → %s  (%d pre-event frames)",
            self.worker.camera_id, self._clip_path, len(self._buffer)
        )
        for frame in list(self._buffer):
            self._writer.write(frame)

        self._recording        = True
        self._post_frames_left = CLIP_POST_SECONDS * CLIP_FPS

        # Stop recording after post window in background
        threading.Thread(target=self._stop_after_post, daemon=True).start()

    def _write_frame(self, frame: cv2.typing.MatLike):
        if self._writer and self._writer.isOpened():
            self._writer.write(frame)
        self._post_frames_left -= 1
        if self._post_frames_left <= 0:
            self._finalize()

    def _stop_after_post(self):
        # Wait until post window expires  (polled inside _write_frame)
        while self._recording:
            time.sleep(0.1)

    def _finalize(self):
        self._recording = False
        if self._writer:
            self._writer.release()
            self._writer = None
        log.info("[%s] Clip saved → %s", self.worker.camera_id, self._clip_path)


# ── Alert dispatcher thread ───────────────────────────────────────────────────
class RecorderDispatcher(threading.Thread):
    """
    Listens to the manager's alert pub/sub stream and triggers the
    correct ClipRecorder for each camera.
    """

    def __init__(self, manager, recorders: dict[str, ClipRecorder]):
        super().__init__(daemon=True, name="rec-dispatcher")
        self.manager   = manager
        self.recorders = recorders   # cam_id → ClipRecorder
        self._alert_q  = manager.subscribe_alerts()  # dedicated queue

    def run(self):
        log.info("Clip recorder dispatcher started.")
        while True:
            while not self._alert_q.empty():
                try:
                    alert = self._alert_q.get_nowait()
                    cam_id = alert.get("camera_id")
                    rec = self.recorders.get(cam_id)
                    if rec:
                        rec.trigger(alert)
                except Exception:
                    break
            time.sleep(0.2)


# ── Convenience factory ───────────────────────────────────────────────────────
def attach_recorders(manager) -> dict[str, ClipRecorder]:
    """
    Create and start a ClipRecorder for every camera worker.
    Returns the recorder map so callers can inspect or trigger manually.
    """
    recorders: dict[str, ClipRecorder] = {}
    for cam_id, worker in manager.workers.items():
        rec = ClipRecorder(worker)   # no alert_queue — dispatcher subscribes independently
        rec.start()
        recorders[cam_id] = rec
        log.info("ClipRecorder attached to %s", cam_id)

    dispatcher = RecorderDispatcher(manager, recorders)
    dispatcher.start()
    return recorders
