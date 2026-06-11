"""
CAMS – Entry Point
Starts the Camera Manager, Super-Admin Dashboard, and all auxiliary daemons.

Subsystems launched:
  • CameraManager   – Manager-Worker thread pool for all RTSP/webcam streams
  • Dashboard       – OpenCV super-admin GUI (grid view, fullscreen, alerts)
  • AlertNotifier   – Windows toast + email alerts on target match
  • HealthMonitor   – Watchdog that auto-restarts crashed/stalled workers
  • ClipRecorders   – Ring-buffer MP4 recording triggered on target detection
"""
import logging
import sys
import os

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cams.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("CAMS")

from config import LBPH_MODEL_PATH, FOUND_DIR, TARGET_DIR, MODEL_DIR, FACE_MODEL, FACE_PROTO
from manager import CameraManager
from dashboard import Dashboard
from alert_notifier import AlertNotifier
from health_monitor import HealthMonitor
from stream_recorder import attach_recorders


# ── Pre-flight checks ─────────────────────────────────────────────────────────
def preflight_checks():
    os.makedirs(FOUND_DIR, exist_ok=True)
    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,  exist_ok=True)

    issues = []

    if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
        issues.append("DNN model files missing  →  run: python setup_models.py")

    if not os.path.exists(LBPH_MODEL_PATH):
        issues.append(
            "LBPH recogniser not trained  →  place photos in target_profile/ "
            "then run: python train.py"
        )
    else:
        log.info("✅ LBPH model found — recognition active.")

    if issues:
        log.warning("=" * 60)
        for i, issue in enumerate(issues, 1):
            log.warning("  ⚠  [%d] %s", i, issue)
        log.warning("  Face recognition alerts will be DISABLED until resolved.")
        log.warning("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  CAMS – Centralized Authorized Monitoring System")
    log.info("=" * 60)

    preflight_checks()

    # Core controller
    manager   = CameraManager()
    dashboard = Dashboard(manager)

    # Start camera fleet (spawns worker threads)
    manager.start_all()

    # ── Auxiliary daemons (all daemon=True, stop with the process) ────────────
    notifier = AlertNotifier(manager)
    notifier.start()

    watchdog = HealthMonitor(manager)
    watchdog.start()

    recorders = attach_recorders(manager)
    log.info(
        "All systems running: %d camera(s), alert notifier, "
        "health watchdog, %d clip recorder(s).",
        len(manager.workers), len(recorders)
    )

    # ── Dashboard blocks here until the OpenCV window is closed ───────────────
    try:
        dashboard.run()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down.")
    finally:
        manager.stop_all()
        log.info("CAMS shutdown complete.")


if __name__ == "__main__":
    main()
