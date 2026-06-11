"""
CAMS – RTSP Stream Simulator
Turns any local video file (or webcam) into a simulated RTSP-style source
that can be fed into CAMERA_STREAMS for offline testing — without needing
real IP cameras.

How it works
─────────────
• Reads a video file and re-serves frames as a simple MJPEG HTTP stream.
• Point any CAMERA_STREAMS entry to:
      "url": "http://localhost:808X/stream"
  and OpenCV's VideoCapture will consume it like a live feed.

Usage (standalone — run BEFORE main.py)
───────────────────────────────────────
  python rtsp_simulator.py --video path/to/clip.mp4 --port 8081
  python rtsp_simulator.py --webcam 0               --port 8082

Multiple instances can simulate multiple cameras on different ports.
"""
import cv2
import argparse
import threading
import time
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger(__name__)


def _capture_frames(source, shared: dict, stop_event: threading.Event):
    """Background thread: loops a video file endlessly and puts latest frame in shared dict."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source: %s", source)
        stop_event.set()
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # Loop the file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        shared["frame"] = frame
        time.sleep(1.0 / fps)

    cap.release()


class _MJPEGHandler(BaseHTTPRequestHandler):
    shared: dict = {}

    def log_message(self, *args):
        pass   # silence HTTP access logs

    def do_GET(self):
        if self.path != "/stream":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=--frame"
        )
        self.end_headers()

        try:
            while True:
                frame = self.shared.get("frame")
                if frame is None:
                    time.sleep(0.05)
                    continue
                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                data = jpeg.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                time.sleep(0.033)   # ~30 fps to clients
        except (BrokenPipeError, ConnectionResetError):
            pass   # client disconnected


def run_simulator(source, port: int = 8081):
    shared: dict = {}
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=_capture_frames,
        args=(source, shared, stop_event),
        daemon=True,
    )
    capture_thread.start()

    # Give the capture thread a moment to grab the first frame
    for _ in range(20):
        if shared.get("frame") is not None:
            break
        time.sleep(0.1)

    class Handler(_MJPEGHandler):
        pass
    Handler.shared = shared

    server = HTTPServer(("0.0.0.0", port), Handler)
    log.info(
        "RTSP Simulator running → http://localhost:%d/stream  (source: %s)",
        port, source
    )
    print(f"\n  ✅  Stream ready at:  http://localhost:{port}/stream")
    print(f"  Add this to CAMERA_STREAMS in config.py:")
    print(f'      "url": "http://localhost:{port}/stream"\n')
    print("  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.server_close()
        log.info("Simulator stopped.")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="CAMS RTSP/MJPEG stream simulator")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  metavar="FILE", help="Path to a local video file")
    group.add_argument("--webcam", metavar="INDEX", type=int, help="Webcam index (0, 1, …)")
    p.add_argument("--port", type=int, default=8081, help="HTTP port (default: 8081)")
    args = p.parse_args()

    source = args.video if args.video else args.webcam
    run_simulator(source, args.port)


if __name__ == "__main__":
    main()
