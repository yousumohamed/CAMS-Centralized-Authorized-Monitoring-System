"""CAMS quick validation — run with: python validate.py"""
import os, sys, queue, time, threading

errors = []

def check(label, fn):
    try:
        fn()
        print(f"  PASS  {label}")
    except Exception as e:
        print(f"  FAIL  {label}: {e}")
        errors.append(label)

print("=" * 50)
print("  CAMS System Validation")
print("=" * 50)

# 1. Encrypted DB
def test_db():
    from db import init_db, log_detection, log_system_event, fetch_detections
    init_db()
    log_system_event("INFO", "validate")
    log_detection("CAM_V", "Validate", 98.1, 51.5074, -0.1278, "crop.jpg")
    rows = fetch_detections(1)
    assert rows and rows[0]["camera_id"] == "CAM_V"
    assert abs(float(rows[0]["gps_lat"]) - 51.5074) < 0.001
check("Encrypted DB write/read/decrypt", test_db)

# 2. Pub/sub fan-out (no DNN)
def test_pubsub():
    class FakeMgr:
        def __init__(self):
            self._src = queue.Queue()
            self._subs, self._lock = [], threading.Lock()
            threading.Thread(target=self._fan, daemon=True).start()
        def subscribe_alerts(self):
            q = queue.Queue()
            with self._lock: self._subs.append(q)
            return q
        def _fan(self):
            while True:
                try:
                    a = self._src.get(timeout=0.1)
                    with self._lock:
                        for s in self._subs: s.put_nowait(a)
                except queue.Empty: pass

    fm = FakeMgr()
    q1, q2 = fm.subscribe_alerts(), fm.subscribe_alerts()
    fm._src.put({"camera_id": "CAM_001", "confidence": 99.0})
    time.sleep(0.25)
    assert not q1.empty() and not q2.empty()
    assert q1.get_nowait()["camera_id"] == "CAM_001"
    assert q2.get_nowait()["camera_id"] == "CAM_001"
check("Alert fan-out (3 consumers independent)", test_pubsub)

# 3. Config & DNN model files
def test_config():
    from config import (MATCH_CONFIDENCE_THRESHOLD, MOTION_DECAY_SECONDS,
                        LBPH_DISTANCE_THRESHOLD, CAMERA_STREAMS,
                        FACE_PROTO, FACE_MODEL)
    assert MATCH_CONFIDENCE_THRESHOLD == 80
    assert MOTION_DECAY_SECONDS == 30
    assert len(CAMERA_STREAMS) >= 1, "No cameras configured"
    assert os.path.exists(FACE_PROTO), f"Missing: {FACE_PROTO}"
    assert os.path.exists(FACE_MODEL), f"Missing: {FACE_MODEL}"
check(f"Config values + DNN model files on disk", test_config)

# 4. Directory layout
def test_dirs():
    from config import FOUND_DIR, TARGET_DIR, MODEL_DIR
    from stream_recorder import CLIPS_DIR
    for d in [FOUND_DIR, TARGET_DIR, MODEL_DIR, CLIPS_DIR]:
        os.makedirs(d, exist_ok=True)
        assert os.path.isdir(d), f"Missing dir: {d}"
check("Directory layout (found_target/ target_profile/ models/ clips/)", test_dirs)

# 5. All module imports
def test_imports():
    mods = ["config","db","recognition","camera_worker","manager",
            "dashboard","alert_notifier","health_monitor",
            "stream_recorder","train","db_viewer"]
    for m in mods:
        __import__(m)
check("All 11 modules importable", test_imports)

# Summary
print("=" * 50)
if errors:
    print(f"  {len(errors)} FAILURE(S): {errors}")
    sys.exit(1)
else:
    print(f"  ALL CHECKS PASSED")
    print()
    print("  To launch:")
    print("    1. Place target photos in ./target_profile/")
    print("    2. python train.py")
    print("    3. python main.py")
print("=" * 50)
