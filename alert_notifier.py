"""
CAMS – Alert Notifier
Runs as a daemon thread, consumes the manager's alert_queue and dispatches
notifications through multiple channels simultaneously:

  • Windows toast notification  (via win10toast – optional, graceful fallback)
  • Email (SMTP)                (configure SMTP settings in config.py)
  • Sound alert                 (WAV file via winsound – optional)

Usage: AlertNotifier(manager).start()   # called from main.py
"""
import threading
import logging
import smtplib
import os
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage
from datetime             import datetime

from config import ALERT_SOUND_PATH

log = logging.getLogger(__name__)

# ── SMTP Configuration (fill in or leave blank to disable email) ──────────────
SMTP_HOST     = ""               # e.g. "smtp.gmail.com"
SMTP_PORT     = 587
SMTP_USER     = ""               # sender address
SMTP_PASSWORD = ""               # app password
SMTP_TO       = ""               # recipient address


# ── Toast helper ──────────────────────────────────────────────────────────────
def _try_toast(title: str, msg: str):
    try:
        from win10toast import ToastNotifier  # optional
        ToastNotifier().show_toast(title, msg, duration=8, threaded=True)
    except ImportError:
        pass        # library not installed — silent fallback
    except Exception as e:
        log.debug("Toast failed: %s", e)


# ── Sound helper ──────────────────────────────────────────────────────────────
def _try_sound():
    try:
        import winsound
        if os.path.exists(ALERT_SOUND_PATH):
            winsound.PlaySound(ALERT_SOUND_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            winsound.Beep(1000, 400)   # fallback system beep
    except Exception as e:
        log.debug("Sound alert failed: %s", e)


# ── Email helper ──────────────────────────────────────────────────────────────
def _send_email(alert: dict):
    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD, SMTP_TO]):
        return   # email not configured

    try:
        msg = MIMEMultipart("related")
        msg["Subject"] = f"[CAMS] TARGET FOUND — {alert['camera_id']}"
        msg["From"]    = SMTP_USER
        msg["To"]      = SMTP_TO

        html_body = f"""
        <html><body style="font-family:Arial; background:#0f0f1a; color:#eee; padding:20px;">
          <h2 style="color:#ff4444;">⚠ TARGET DETECTED</h2>
          <table style="border-collapse:collapse;">
            <tr><td style="padding:4px 12px;color:#aaa;">Camera</td>
                <td><b>{alert['camera_id']} — {alert['camera_name']}</b></td></tr>
            <tr><td style="padding:4px 12px;color:#aaa;">Confidence</td>
                <td><b style="color:#ff8c00;">{alert['confidence']:.1f}%</b></td></tr>
            <tr><td style="padding:4px 12px;color:#aaa;">Timestamp</td>
                <td>{alert['timestamp']} UTC</td></tr>
            <tr><td style="padding:4px 12px;color:#aaa;">GPS</td>
                <td>{alert['lat']:.5f}, {alert['lon']:.5f}</td></tr>
          </table>
          {'<br><img src="cid:crop" style="max-width:400px;border:2px solid #ff4444;">'
            if os.path.exists(alert.get('crop','')) else ''}
        </body></html>
        """
        msg.attach(MIMEText(html_body, "html"))

        # Attach crop image if available
        crop_path = alert.get("crop", "")
        if os.path.exists(crop_path):
            with open(crop_path, "rb") as f:
                img = MIMEImage(f.read())
            img.add_header("Content-ID", "<crop>")
            msg.attach(img)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASSWORD)
            s.sendmail(SMTP_USER, SMTP_TO, msg.as_string())

        log.info("📧  Email alert sent to %s", SMTP_TO)
    except Exception as exc:
        log.warning("Email alert failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
class AlertNotifier(threading.Thread):
    """
    Daemon thread that drains the alert queue and fires all notification
    channels. De-duplicates alerts within a 60-second window to avoid spam.
    """

    COOLDOWN = 60   # seconds per camera before re-alerting

    def __init__(self, manager):
        super().__init__(daemon=True, name="alert-notifier")
        self.manager    = manager
        self._alert_q   = manager.subscribe_alerts()   # dedicated queue
        self._last_sent : dict[str, float] = {}   # camera_id → epoch

    def run(self):
        log.info("Alert notifier started.")
        while True:
            # Drain our dedicated subscriber queue (non-blocking)
            while not self._alert_q.empty():
                try:
                    alert = self._alert_q.get_nowait()
                    self._dispatch(alert)
                except Exception:
                    break
            time.sleep(0.4)

    def _dispatch(self, alert: dict):
        cam = alert["camera_id"]
        now = time.time()
        if now - self._last_sent.get(cam, 0) < self.COOLDOWN:
            return   # still in cooldown
        self._last_sent[cam] = now

        log.warning(
            "🔔 ALERT — Camera %s | Confidence %.1f%% | %s",
            cam, alert["confidence"], alert["timestamp"]
        )
        _try_sound()
        _try_toast(
            "CAMS — Target Found!",
            f"Camera: {alert['camera_name']}\n"
            f"Confidence: {alert['confidence']:.1f}%\n"
            f"GPS: {alert['lat']:.4f}, {alert['lon']:.4f}"
        )
        # Email runs in its own thread to avoid blocking the loop
        threading.Thread(target=_send_email, args=(alert,), daemon=True).start()
