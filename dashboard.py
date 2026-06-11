"""
CAMS – Super-Admin Dashboard
OpenCV-based full-screen GUI with:
  • Thumbnail grid of all cameras
  • Full-screen single-camera view (click to select)
  • Real-time alert panel
  • Per-camera analytics toggle
  • Search Mode indicator (hot cameras highlighted)
  • Keyboard shortcuts
"""
import cv2
import numpy as np
import logging
import time
from datetime import datetime
from config import (
    DASHBOARD_TITLE, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT
)

log = logging.getLogger(__name__)

# ── Colour Palette (BGR) ─────────────────────────────────────────────────────
C_BG        = (18, 18, 30)
C_PANEL     = (30, 30, 48)
C_ACCENT    = (255, 140, 0)
C_GREEN     = (0, 210, 100)
C_RED       = (0, 80, 255)
C_YELLOW    = (0, 200, 255)
C_WHITE     = (240, 240, 240)
C_GREY      = (120, 120, 140)
C_HOT       = (0, 165, 255)
C_ALERT_BG  = (10, 20, 60)


def _text(img, msg, pos, scale=0.5, color=C_WHITE, thickness=1, bold=False):
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, msg, pos, font, scale, color, thickness, cv2.LINE_AA)


class Dashboard:
    """
    Renders a full OpenCV window.

    Keyboard controls
    -----------------
    Q / ESC   – quit
    F         – fullscreen toggle for selected camera
    A         – toggle analytics on selected camera
    G         – toggle analytics globally (ALL cameras)
    R         – restart selected camera worker
    S         – print status snapshot to terminal
    """

    SIDEBAR_W = 320   # right-side alert panel width

    @property
    def COLS(self):
        """Auto-size columns based on number of cameras (2->2, 3-4->2, 5+->3)."""
        n = len(self.manager.workers)
        if n <= 2:
            return 2
        if n <= 4:
            return 2
        return 3

    def __init__(self, manager):
        self.manager       = manager
        self._selected_id  : str | None = None
        self._fullscreen   = False
        self._alerts       : list[dict] = []          # last N alerts
        self._max_alerts   = 12
        self._alert_flash  = 0          # epoch seconds of last alert (for flash)
        self._running      = False
        # Subscribe to the manager's pub/sub fan-out
        self._alert_q      = manager.subscribe_alerts()

        self._win = DASHBOARD_TITLE
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._win, 1440, 880)
        cv2.setMouseCallback(self._win, self._mouse_cb)

    # ── Public API ───────────────────────────────────────────────────────────
    def run(self):
        self._running = True
        log.info("Dashboard started.")
        while self._running:
            # Drain this dashboard's dedicated alert queue
            new_alerts = []
            while not self._alert_q.empty():
                try:
                    new_alerts.append(self._alert_q.get_nowait())
                except Exception:
                    break
            if new_alerts:
                self._alerts = (new_alerts + self._alerts)[: self._max_alerts]
                self._alert_flash = time.time()

            # Build canvas
            canvas = self._build_canvas()
            cv2.imshow(self._win, canvas)

            key = cv2.waitKey(33) & 0xFF      # ~30 fps render tick
            self._handle_key(key)

            if cv2.getWindowProperty(self._win, cv2.WND_PROP_VISIBLE) < 1:
                self._running = False

        cv2.destroyAllWindows()
        self.manager.stop_all()

    # ── Canvas construction ──────────────────────────────────────────────────
    def _build_canvas(self) -> np.ndarray:
        snapshot = self.manager.get_status_snapshot()
        workers_ids = [s["id"] for s in snapshot]

        if self._fullscreen and self._selected_id:
            return self._render_fullscreen(snapshot)

        # Win size
        win_w = self.COLS * THUMBNAIL_WIDTH + self.SIDEBAR_W + 4
        rows  = max(1, -(-len(snapshot) // self.COLS))   # ceil div
        win_h = max(880, rows * THUMBNAIL_HEIGHT + 120)

        canvas = np.full((win_h, win_w, 3), C_BG, dtype=np.uint8)

        # --- Top bar ---
        cv2.rectangle(canvas, (0, 0), (win_w, 48), C_PANEL, -1)
        _text(canvas, DASHBOARD_TITLE, (12, 30), scale=0.7,
              color=C_ACCENT, thickness=2, bold=True)
        ts = datetime.utcnow().strftime("UTC %Y-%m-%d  %H:%M:%S")
        _text(canvas, ts, (win_w - 280, 30), scale=0.55, color=C_GREY)

        # --- Alert flash overlay ---
        if (time.time() - self._alert_flash) < 0.5:
            flash = canvas.copy()
            cv2.rectangle(flash, (0, 0), (win_w, win_h), (0, 0, 180), -1)
            cv2.addWeighted(flash, 0.15, canvas, 0.85, 0, canvas)
            _text(canvas, "!! TARGET DETECTED !!",
                  (win_w // 2 - 230, win_h // 2),
                  scale=2.0, color=C_RED, thickness=4, bold=True)

        # --- Thumbnail grid ---
        grid_w = self.COLS * THUMBNAIL_WIDTH
        for idx, meta in enumerate(snapshot):
            col = idx % self.COLS
            row = idx // self.COLS
            x0  = col * THUMBNAIL_WIDTH
            y0  = 52 + row * THUMBNAIL_HEIGHT

            w = self.manager.get_worker(meta["id"])
            thumb = w.get_thumbnail() if w else None

            if thumb is None:
                cell = np.full((THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 3),
                               (20, 20, 35), dtype=np.uint8)
                # Draw a border
                cv2.rectangle(cell, (0, 0),
                              (THUMBNAIL_WIDTH-1, THUMBNAIL_HEIGHT-1),
                              C_GREY, 1)
                # Camera name centred
                _text(cell, "NO SIGNAL",
                      (THUMBNAIL_WIDTH//2 - 55, THUMBNAIL_HEIGHT//2 - 10),
                      scale=0.65, color=C_GREY, bold=True)
                _text(cell, meta['name'],
                      (THUMBNAIL_WIDTH//2 - 50, THUMBNAIL_HEIGHT//2 + 16),
                      scale=0.42, color=(80, 80, 100))
            else:
                cell = cv2.resize(thumb, (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))

            # Selection border
            if meta["id"] == self._selected_id:
                cv2.rectangle(cell, (0, 0),
                              (THUMBNAIL_WIDTH - 1, THUMBNAIL_HEIGHT - 1),
                              C_ACCENT, 3)
            # Hot border
            elif meta.get("is_hot"):
                cv2.rectangle(cell, (0, 0),
                              (THUMBNAIL_WIDTH - 1, THUMBNAIL_HEIGHT - 1),
                              C_HOT, 2)

            canvas[y0:y0 + THUMBNAIL_HEIGHT, x0:x0 + THUMBNAIL_WIDTH] = cell

            # Mini label
            label_y = y0 + THUMBNAIL_HEIGHT - 6
            cv2.rectangle(canvas, (x0, label_y - 16), (x0 + THUMBNAIL_WIDTH, label_y + 4),
                          (0, 0, 0), -1)
            _text(canvas, f"{meta['id']} | {meta['name']} | {meta['fps']}fps",
                  (x0 + 4, label_y), scale=0.38, color=C_WHITE)

        # --- Sidebar ---
        sx = grid_w + 2
        self._draw_sidebar(canvas, snapshot, sx, win_h)
        return canvas

    def _draw_sidebar(self, canvas: np.ndarray, snapshot: list,
                      sx: int, win_h: int):
        sw = self.SIDEBAR_W - 4
        # Background
        cv2.rectangle(canvas, (sx, 0), (sx + sw, win_h), C_PANEL, -1)

        # ── Camera status list ──
        _text(canvas, "CAMERA STATUS", (sx + 8, 64),
              scale=0.55, color=C_ACCENT, bold=True)
        for i, s in enumerate(snapshot):
            cy = 80 + i * 52
            dot_c = {
                "LIVE":       C_GREEN,
                "ERROR":      C_RED,
                "CONNECTING": C_YELLOW,
                "NO SIGNAL":  C_GREY,
                "STOPPED":    C_GREY,
            }.get(s["status"], C_GREY)
            cv2.circle(canvas, (sx + 14, cy + 10), 6, dot_c, -1)
            _text(canvas, s["id"], (sx + 26, cy + 14),
                  scale=0.48, color=C_WHITE, bold=True)
            hot_tag = "HOT" if s["is_hot"] else "---"
            _text(canvas,
                  f"  {s['name']}  |  A:{'ON' if s['analytics'] else 'OFF'}"
                  f"  |  {hot_tag}",
                  (sx + 26, cy + 30), scale=0.38, color=C_GREY)
            _text(canvas,
                  f"  Detections: {s['detections_today']}",
                  (sx + 26, cy + 44), scale=0.38, color=C_YELLOW)

        # ── Alert log ──
        divider_y = 80 + len(snapshot) * 52 + 8
        cv2.line(canvas, (sx + 8, divider_y), (sx + sw - 8, divider_y), C_GREY, 1)
        _text(canvas, "ALERT LOG", (sx + 8, divider_y + 20),
              scale=0.55, color=C_RED, bold=True)
        for j, alert in enumerate(self._alerts):
            ay = divider_y + 38 + j * 44
            if ay + 44 > canvas.shape[0]:
                break
            conf = alert.get("confidence", 0)
            _text(canvas,
                  f">> {alert['camera_id']} | {alert['timestamp']}",
                  (sx + 8, ay), scale=0.42, color=C_ACCENT, bold=True)
            _text(canvas,
                  f"  Conf: {conf:.1f}%  Lat:{alert['lat']:.4f}  Lon:{alert['lon']:.4f}",
                  (sx + 8, ay + 16), scale=0.38, color=C_WHITE)
            _text(canvas,
                  f"  {alert['camera_name']}",
                  (sx + 8, ay + 30), scale=0.36, color=C_GREY)

        # ── Key hints ──
        hints = [
            "Q/ESC: Quit",
            "F: Fullscreen cam",
            "A: Toggle analytics",
            "G: Global analytics",
            "R: Restart cam",
        ]
        hy = win_h - len(hints) * 18 - 12
        cv2.line(canvas, (sx + 8, hy - 8), (sx + sw - 8, hy - 8), C_GREY, 1)
        for i, h in enumerate(hints):
            _text(canvas, h, (sx + 10, hy + i * 18),
                  scale=0.38, color=C_GREY)

    def _render_fullscreen(self, snapshot: list) -> np.ndarray:
        w = self.manager.get_worker(self._selected_id)
        if w is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        _, display = w.get_frame()
        if display is None:
            canvas = np.full((720, 1280, 3), C_BG, dtype=np.uint8)
            _text(canvas, "NO FRAME", (500, 360), scale=1.2, color=C_GREY)
            return canvas
        canvas = cv2.resize(display, (1440, 880))
        # Overlay bar
        cv2.rectangle(canvas, (0, 0), (1440, 40), (0, 0, 0), -1)
        _text(canvas,
              f"FULLSCREEN  {self._selected_id}  |  Press F to exit",
              (10, 28), scale=0.65, color=C_ACCENT, bold=True)
        return canvas

    # ── Input handling ────────────────────────────────────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._fullscreen:
            self._fullscreen = False
            return
        snapshot = self.manager.get_status_snapshot()
        grid_w   = self.COLS * THUMBNAIL_WIDTH
        if x >= grid_w:
            return   # click in sidebar
        col = x // THUMBNAIL_WIDTH
        row = (y - 52) // THUMBNAIL_HEIGHT
        if row < 0:
            return
        idx = row * self.COLS + col
        if idx < len(snapshot):
            self._selected_id = snapshot[idx]["id"]
            log.info("Selected camera: %s", self._selected_id)

    def _handle_key(self, key: int):
        if key in (ord("q"), 27):          # Q / ESC
            self._running = False
        elif key == ord("f") and self._selected_id:
            self._fullscreen = not self._fullscreen
        elif key == ord("a") and self._selected_id:
            w = self.manager.get_worker(self._selected_id)
            if w:
                w.set_analytics(not w.get_analytics())
        elif key == ord("g"):
            # Toggle global analytics based on majority state
            snap  = self.manager.get_status_snapshot()
            on    = sum(1 for s in snap if s["analytics"])
            enable = on < len(snap) / 2
            self.manager.set_analytics(None, enable)
        elif key == ord("r") and self._selected_id:
            self.manager.restart_worker(self._selected_id)
        elif key == ord("s"):
            for s in self.manager.get_status_snapshot():
                print(s)
