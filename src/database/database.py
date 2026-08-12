"""SQLite connection + schema management (Section 27, 28)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from src.utils.config import get_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source_file TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('image', 'video')),
    frame_number INTEGER,
    video_timestamp REAL,
    damage_type TEXT NOT NULL CHECK (damage_type IN ('pothole', 'crack')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    width_px REAL NOT NULL,
    height_px REAL NOT NULL,
    area_px REAL NOT NULL,
    relative_area REAL NOT NULL,
    estimated_width_m REAL,
    estimated_height_m REAL,
    severity_score REAL NOT NULL CHECK (severity_score >= 0 AND severity_score <= 100),
    severity_category TEXT NOT NULL,
    -- Future Phase-2 fields (Section 27, 43); NULL in Phase 1
    latitude REAL,
    longitude REAL,
    vibration_magnitude REAL,
    vehicle_speed REAL
);

CREATE INDEX IF NOT EXISTS idx_detections_damage_type ON detections(damage_type);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_source_file ON detections(source_file);
"""


class DatabaseError(Exception):
    pass


def get_db_path(config: dict | None = None) -> Path:
    cfg = config or get_config()
    db_path = cfg.get("database", {}).get("path", "data/road_damage.db")
    return resolve_path(db_path)


def get_connection(config: dict | None = None) -> sqlite3.Connection:
    db_path = get_db_path(config)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to connect to database at '{db_path}': {e}") from e


def init_db(config: dict | None = None) -> None:
    conn = get_connection(config)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
        logger.info(f"Database initialized at: {get_db_path(config)}")
    except sqlite3.Error as e:
        conn.rollback()
        raise DatabaseError(f"Failed to initialize schema: {e}") from e
    finally:
        conn.close()
