"""
CAMS – Encrypted Database Layer
Uses SQLite for storage and Fernet (AES-128-CBC) to encrypt sensitive fields.
"""
import sqlite3
import json
import base64
import hashlib
import logging
from datetime import datetime
from cryptography.fernet import Fernet
from config import DB_PATH, DB_ENCRYPTION_KEY

log = logging.getLogger(__name__)

# ── Derive a valid Fernet key from our arbitrary passphrase ───────────────────
def _derive_key(passphrase: bytes) -> bytes:
    digest = hashlib.sha256(passphrase).digest()
    return base64.urlsafe_b64encode(digest)

_fernet = Fernet(_derive_key(DB_ENCRYPTION_KEY))


def _encrypt(value: str) -> str:
    return _fernet.encrypt(value.encode()).decode()


def _decrypt(value: str) -> str:
    try:
        return _fernet.decrypt(value.encode()).decode()
    except Exception:
        return value          # fallback for plain-text rows during migration


# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS detection_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    camera_id   TEXT    NOT NULL,
    camera_name TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    gps_lat     TEXT    NOT NULL,   -- encrypted
    gps_lon     TEXT    NOT NULL,   -- encrypted
    crop_path   TEXT    NOT NULL,
    extra_meta  TEXT                -- JSON blob, encrypted
);

CREATE TABLE IF NOT EXISTS system_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    level     TEXT NOT NULL,
    message   TEXT NOT NULL
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    log.info("Database initialised at %s", DB_PATH)


# ── Write helpers ─────────────────────────────────────────────────────────────
def log_detection(camera_id: str, camera_name: str, confidence: float,
                  lat: float, lon: float, crop_path: str,
                  extra: dict | None = None):
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO detection_log
                (timestamp, camera_id, camera_name, confidence,
                 gps_lat, gps_lon, crop_path, extra_meta)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                datetime.utcnow().isoformat(),
                camera_id,
                camera_name,
                round(confidence, 2),
                _encrypt(str(lat)),
                _encrypt(str(lon)),
                crop_path,
                _encrypt(json.dumps(extra or {})),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def log_system_event(level: str, message: str):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO system_log (timestamp, level, message) VALUES (?,?,?)",
            (datetime.utcnow().isoformat(), level, message),
        )
        conn.commit()
    finally:
        conn.close()


# ── Read helpers ──────────────────────────────────────────────────────────────
def fetch_detections(limit: int = 100) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM detection_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        conn.close()

    result = []
    for r in rows:
        row = dict(r)
        row["gps_lat"] = _decrypt(row["gps_lat"])
        row["gps_lon"] = _decrypt(row["gps_lon"])
        row["extra_meta"] = json.loads(_decrypt(row["extra_meta"]))
        result.append(row)
    return result


def fetch_system_logs(limit: int = 200) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM system_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
