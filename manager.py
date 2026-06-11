"""
CAMS – Camera Manager
Central controller that owns all CameraWorker threads.

Alert fan-out
─────────────
Workers push raw alert dicts to the internal source queue (_src_queue).
A background fan-out thread copies each alert to every registered
subscriber queue so Dashboard, AlertNotifier, and ClipRecorder each
receive every event independently — no race conditions.

Usage:
    my_q = manager.subscribe_alerts()   # call once per consumer
    # later, non-blocking drain:
    while not my_q.empty():
        alert = my_q.get_nowait()
"""
import queue
import logging
import threading
from recognition import FaceRecognitionEngine, get_engine
from camera_worker import CameraWorker
from db import log_detection, log_system_event, init_db
from config import CAMERA_STREAMS

log = logging.getLogger(__name__)


class CameraManager:
    """Lifecycle controller for the whole camera fleet."""

    def __init__(self):
        init_db()
        self.engine   = get_engine()           # process-wide singleton
        self.workers  : dict[str, CameraWorker] = {}
        self._lock    = threading.Lock()
        self._running = False

        # ── Alert pub/sub ─────────────────────────────────────────────────────
        # Workers write to _src_queue; fanout copies to every subscriber queue.
        self._src_queue   : queue.Queue          = queue.Queue()
        self._subscribers : list[queue.Queue]    = []
        self._sub_lock    = threading.Lock()
        threading.Thread(
            target=self._fanout_loop, daemon=True, name="alert-fanout"
        ).start()

    # ── Pub/Sub API ───────────────────────────────────────────────────────────
    def subscribe_alerts(self) -> queue.Queue:
        """
        Register a new alert consumer.
        Returns a dedicated Queue; every future alert dict is put() into it.
        Call once per consumer before manager.start_all().
        """
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._sub_lock:
            self._subscribers.append(q)
        log.debug("Alert subscriber registered (%d total).", len(self._subscribers))
        return q

    def _fanout_loop(self):
        """Background daemon: drain source, fan out to all subscribers."""
        while True:
            try:
                alert = self._src_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._sub_lock:
                for sub in self._subscribers:
                    try:
                        sub.put_nowait(alert)
                    except queue.Full:
                        pass   # slow consumer; drop to avoid blocking

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start_all(self):
        log.info("Starting %d camera worker(s)…", len(CAMERA_STREAMS))
        for cam_id, meta in CAMERA_STREAMS.items():
            self._start_worker(cam_id, meta)
        self._running = True
        log_system_event("INFO", "CameraManager started all workers.")

    def stop_all(self):
        log.info("Stopping all camera workers…")
        with self._lock:
            for w in self.workers.values():
                w.stop()
        for w in self.workers.values():
            w.join(timeout=5)
        self._running = False
        log_system_event("INFO", "CameraManager stopped all workers.")

    def _start_worker(self, cam_id: str, meta: dict):
        w = CameraWorker(
            camera_id   = cam_id,
            meta        = meta,
            engine      = self.engine,
            alert_queue = self._src_queue,   # workers always push to source
            db_writer   = log_detection,
        )
        with self._lock:
            self.workers[cam_id] = w
        w.start()
        log.info("Worker started: %s (%s)", cam_id, meta["name"])

    def restart_worker(self, cam_id: str):
        with self._lock:
            if cam_id in self.workers:
                self.workers[cam_id].stop()
                self.workers[cam_id].join(timeout=5)
        meta = CAMERA_STREAMS.get(cam_id)
        if meta:
            self._start_worker(cam_id, meta)
            log.info("Worker restarted: %s", cam_id)

    # ── Per-camera controls ───────────────────────────────────────────────────
    def set_analytics(self, cam_id: str | None, enabled: bool):
        """Pass cam_id=None to apply to all cameras."""
        with self._lock:
            targets = (
                [self.workers[cam_id]] if cam_id
                else list(self.workers.values())
            )
        for w in targets:
            w.set_analytics(enabled)
        log_system_event(
            "INFO",
            f"Analytics {'ON' if enabled else 'OFF'} for {cam_id or 'ALL'}",
        )

    def get_worker(self, cam_id: str) -> CameraWorker | None:
        with self._lock:
            return self.workers.get(cam_id)

    # ── Status snapshot ───────────────────────────────────────────────────────
    def get_status_snapshot(self) -> list[dict]:
        with self._lock:
            workers = list(self.workers.values())
        return [
            {
                "id":               w.camera_id,
                "name":             w.name_label,
                "status":           w.status,
                "fps":              w.fps,
                "analytics":        w.get_analytics(),
                "is_hot":           w.is_hot(),
                "detections_today": w.detections_today,
                "last_seen":        w.last_seen_ts,
                "lat":              w.lat,
                "lon":              w.lon,
            }
            for w in workers
        ]

    # ── Search optimisation ───────────────────────────────────────────────────
    def hot_cameras(self) -> list[str]:
        """Return IDs of cameras currently showing motion."""
        with self._lock:
            return [cid for cid, w in self.workers.items() if w.is_hot()]
