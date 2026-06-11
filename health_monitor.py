"""
CAMS – Health Monitor & Watchdog
Runs as a daemon thread that:

  • Checks each CameraWorker's status every HEALTH_CHECK_INTERVAL seconds
  • Auto-restarts workers that have crashed (status == 'ERROR') or stalled
    (no new frames for STALL_TIMEOUT seconds)
  • Logs all health events to the DB system_log table
  • Exposes a live summary via get_report()
"""
import threading
import time
import logging
from datetime import datetime
from db import log_system_event

log = logging.getLogger(__name__)

HEALTH_CHECK_INTERVAL = 15    # seconds between health sweeps
STALL_TIMEOUT         = 30    # seconds of no new frame = stalled
MAX_RESTARTS          = 5     # max consecutive restarts before giving up


class HealthMonitor(threading.Thread):
    """
    Watchdog daemon for the camera fleet.
    Pass in the CameraManager instance.
    """

    def __init__(self, manager):
        super().__init__(daemon=True, name="health-monitor")
        self.manager        = manager
        self._restart_count : dict[str, int]   = {}   # cam_id → count
        self._last_frame_ts : dict[str, float] = {}   # cam_id → epoch
        self._report        : list[dict]       = []
        self._lock          = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────
    def get_report(self) -> list[dict]:
        with self._lock:
            return list(self._report)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        log.info("Health monitor started (interval=%ds).", HEALTH_CHECK_INTERVAL)
        while True:
            time.sleep(HEALTH_CHECK_INTERVAL)
            self._sweep()

    def _sweep(self):
        snapshot = self.manager.get_status_snapshot()
        now      = time.time()
        report   = []

        for s in snapshot:
            cam_id = s["id"]
            worker = self.manager.get_worker(cam_id)
            if worker is None:
                continue

            # Update frame-change tracker
            current_count = worker.frame_count
            prev_count    = self._last_frame_ts.get(cam_id + "_fc", 0)
            if current_count != prev_count:
                self._last_frame_ts[cam_id] = now
                self._last_frame_ts[cam_id + "_fc"] = current_count

            last_frame_ago = now - self._last_frame_ts.get(cam_id, now)
            stalled = last_frame_ago > STALL_TIMEOUT and s["status"] == "LIVE"
            crashed = s["status"] == "ERROR"

            health = "OK"
            if crashed:
                health = "CRASHED"
            elif stalled:
                health = "STALLED"

            entry = {
                "camera_id":      cam_id,
                "name":           s["name"],
                "status":         s["status"],
                "health":         health,
                "fps":            s["fps"],
                "restarts":       self._restart_count.get(cam_id, 0),
                "last_frame_ago": round(last_frame_ago, 1),
                "checked_at":     datetime.utcnow().isoformat(),
            }
            report.append(entry)

            if health in ("CRASHED", "STALLED"):
                self._handle_unhealthy(cam_id, health)

        with self._lock:
            self._report = report

        ok  = sum(1 for r in report if r["health"] == "OK")
        bad = len(report) - ok
        log.info(
            "Health sweep: %d/%d OK  |  %d problem(s)",
            ok, len(report), bad
        )

    def _handle_unhealthy(self, cam_id: str, reason: str):
        restarts = self._restart_count.get(cam_id, 0)
        if restarts >= MAX_RESTARTS:
            log.error(
                "[%s] Exceeded max restarts (%d) — giving up.",
                cam_id, MAX_RESTARTS
            )
            log_system_event(
                "ERROR",
                f"{cam_id} exceeded {MAX_RESTARTS} restart attempts — manual intervention required."
            )
            return

        log.warning(
            "[%s] %s detected — restarting (attempt %d/%d).",
            cam_id, reason, restarts + 1, MAX_RESTARTS
        )
        log_system_event(
            "WARNING",
            f"{cam_id} {reason} — auto-restart #{restarts + 1}"
        )
        self.manager.restart_worker(cam_id)
        self._restart_count[cam_id] = restarts + 1
        # Reset stall clock
        self._last_frame_ts[cam_id] = time.time()

    def reset_restart_count(self, cam_id: str):
        """Call when a worker recovers cleanly."""
        self._restart_count.pop(cam_id, None)
