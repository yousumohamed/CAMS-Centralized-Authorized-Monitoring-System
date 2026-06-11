"""
CAMS – Face Recognition Engine
Uses OpenCV's LBPH recogniser trained on the target profile image(s).
Detection front-end: res10 SSD (OpenCV DNN) — runs on CPU without dlib.
"""
import os
import cv2
import numpy as np
import logging
import urllib.request
from pathlib import Path
from config import (
    FACE_PROTO, FACE_MODEL, LBPH_MODEL_PATH,
    TARGET_DIR, MODEL_DIR, FOUND_DIR,
    MATCH_CONFIDENCE_THRESHOLD, LBPH_DISTANCE_THRESHOLD
)

log = logging.getLogger(__name__)

# ── Model download URLs (public OpenCV model zoo) ────────────────────────────
_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)


def _download(url: str, dest: str):
    if not os.path.exists(dest):
        import socket
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(10)         # never hang indefinitely
        try:
            log.info("Downloading %s → %s", url, dest)
            urllib.request.urlretrieve(url, dest)
        finally:
            socket.setdefaulttimeout(old_timeout)


def _ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    _download(_PROTO_URL, FACE_PROTO)
    _download(_MODEL_URL, FACE_MODEL)


# ── Singleton so the 10 MB caffemodel is loaded from disk exactly once ────────
_ENGINE_INSTANCE : "FaceRecognitionEngine | None" = None
_ENGINE_LOCK = __import__("threading").Lock()


def get_engine() -> "FaceRecognitionEngine":
    """Return the process-wide singleton engine, creating it on first call."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        with _ENGINE_LOCK:
            if _ENGINE_INSTANCE is None:
                _ENGINE_INSTANCE = FaceRecognitionEngine()
    return _ENGINE_INSTANCE


# ─────────────────────────────────────────────────────────────────────────────
class FaceRecognitionEngine:
    """
    Wraps:
      • OpenCV DNN SSD face detector  (fast, CPU-friendly)
      • LBPH face recogniser          (trained on target profile)
    """

    TARGET_LABEL = 1   # label assigned to the target person

    def __init__(self):
        _ensure_models()
        os.makedirs(FOUND_DIR, exist_ok=True)
        os.makedirs(TARGET_DIR, exist_ok=True)

        # Load DNN detector
        self.net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
        log.info("DNN face detector loaded.")

        # LBPH recogniser
        self.recogniser = cv2.face.LBPHFaceRecognizer_create()
        self._trained = False

        if os.path.exists(LBPH_MODEL_PATH):
            self.recogniser.read(LBPH_MODEL_PATH)
            self._trained = True
            log.info("LBPH model loaded from %s", LBPH_MODEL_PATH)

    # ── Training ──────────────────────────────────────────────────────────────
    def train_from_directory(self, directory: str | None = None) -> bool:
        """
        Train LBPH on all images inside *directory* (default: TARGET_DIR).
        Each *other* person can be put in a subdirectory named with their label
        integer; faces belonging to the target go directly in the root.
        """
        directory = directory or TARGET_DIR
        images, labels = [], []

        for fp in Path(directory).glob("**/*"):
            if fp.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Assign label: 1 = target (root files), else 0 = others
            label = self.TARGET_LABEL if fp.parent == Path(directory) else 0
            faces = self._detect_faces_raw(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            for (x, y, w, h) in faces:
                roi = cv2.resize(img[y:y+h, x:x+w], (100, 100))
                images.append(roi)
                labels.append(label)

        if not images:
            log.warning(
                "No training faces found in %s. "
                "Place the target's photos there and call train_from_directory().",
                directory
            )
            return False

        self.recogniser.train(images, np.array(labels))
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.recogniser.save(LBPH_MODEL_PATH)
        self._trained = True
        log.info("LBPH trained on %d face crops, saved to %s", len(images), LBPH_MODEL_PATH)
        return True

    # ── Detection + recognition ───────────────────────────────────────────────
    def _detect_faces_raw(self, frame: np.ndarray) -> list[tuple[int,int,int,int]]:
        """Return [(x,y,w,h), ...] bounding boxes via DNN."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.5:
                continue
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    def analyse_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Returns a list of detected faces. Each dict:
          {
            "bbox": (x,y,w,h),
            "is_target": bool,
            "confidence": float (0-100),
            "label": int
          }
        """
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for bbox in self._detect_faces_raw(frame):
            x, y, w, h = bbox
            if w <= 0 or h <= 0:  # protect against empty regions
                continue

            roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
            if self._trained:
                label, distance = self.recogniser.predict(roi)
                # Convert LBPH distance to 0-100 confidence
                # distance=0 → 100 %, distance=threshold → 0 %
                raw_conf = max(0.0, 1.0 - distance / LBPH_DISTANCE_THRESHOLD)
                confidence = round(raw_conf * 100, 2)
                is_target = (label == self.TARGET_LABEL
                             and confidence >= MATCH_CONFIDENCE_THRESHOLD)
            else:
                label, confidence, is_target = -1, 0.0, False

            results.append({
                "bbox": bbox,
                "is_target": is_target,
                "confidence": confidence,
                "label": label,
            })
        return results

    def save_crop(self, frame: np.ndarray, bbox: tuple,
                  camera_id: str, timestamp_str: str) -> str:
        """Save a face crop to FOUND_DIR and return the file path."""
        x, y, w, h = bbox
        # Add padding around crop
        ph = max(0, y - 20)
        pw = max(0, x - 20)
        crop = frame[ph: y + h + 20, pw: x + w + 20]
        fname = f"{camera_id}_{timestamp_str}.jpg"
        path = os.path.join(FOUND_DIR, fname)
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return path

    def draw_results(self, frame: np.ndarray, results: list[dict]) -> np.ndarray:
        """Draw bounding boxes and confidence labels onto the frame (in-place)."""
        out = frame.copy()
        for r in results:
            x, y, w, h = r["bbox"]
            colour = (0, 60, 255) if r["is_target"] else (0, 200, 100)
            thickness = 3 if r["is_target"] else 1
            cv2.rectangle(out, (x, y), (x + w, y + h), colour, thickness)
            label = (
                f"TARGET {r['confidence']:.0f}%"
                if r["is_target"]
                else f"Face {r['confidence']:.0f}%"
            )
            cv2.putText(out, label, (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, colour, 1, cv2.LINE_AA)
        return out
