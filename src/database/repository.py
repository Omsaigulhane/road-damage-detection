"""
Repository / service layer for detection records (Section 28).

The rest of the application (CLI, dashboard) should go through
DetectionRepository rather than writing SQL directly.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from src.database.database import get_connection, init_db, DatabaseError
from src.database.models import DetectionRecord
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_SOURCE_TYPES = {"image", "video"}
VALID_DAMAGE_TYPES = {"pothole", "crack"}

_INSERT_SQL = """
INSERT INTO detections (
    timestamp, source_file, source_type, frame_number, video_timestamp,
    damage_type, confidence, x1, y1, x2, y2, width_px, height_px, area_px,
    relative_area, estimated_width_m, estimated_height_m, severity_score,
    severity_category, latitude, longitude, vibration_magnitude, vehicle_speed
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class RepositoryError(Exception):
    pass


class DetectionRepository:
    def __init__(self, config: dict | None = None, auto_init: bool = True):
        self.config = config
        if auto_init:
            init_db(config)

    def _validate(self, record: DetectionRecord) -> None:
        if record.source_type not in VALID_SOURCE_TYPES:
            raise RepositoryError(f"Invalid source_type '{record.source_type}'.")
        if record.damage_type not in VALID_DAMAGE_TYPES:
            raise RepositoryError(f"Invalid damage_type '{record.damage_type}'.")
        if not (0.0 <= record.confidence <= 1.0):
            raise RepositoryError(f"confidence out of range [0,1]: {record.confidence}")
        if not (0.0 <= record.severity_score <= 100.0):
            raise RepositoryError(f"severity_score out of range [0,100]: {record.severity_score}")
        if not record.source_file:
            raise RepositoryError("source_file is required.")

    def insert(self, record: DetectionRecord) -> int:
        self._validate(record)
        conn = get_connection(self.config)
        try:
            cur = conn.execute(
                _INSERT_SQL,
                (
                    record.timestamp or datetime.utcnow().isoformat(),
                    record.source_file,
                    record.source_type,
                    record.frame_number,
                    record.video_timestamp,
                    record.damage_type,
                    record.confidence,
                    record.x1, record.y1, record.x2, record.y2,
                    record.width_px, record.height_px, record.area_px,
                    record.relative_area,
                    record.estimated_width_m, record.estimated_height_m,
                    record.severity_score, record.severity_category,
                    record.latitude, record.longitude,
                    record.vibration_magnitude, record.vehicle_speed,
                ),
            )
            conn.commit()
            return cur.lastrowid
        except sqlite3.Error as e:
            conn.rollback()
            raise DatabaseError(f"Failed to insert detection record: {e}") from e
        finally:
            conn.close()

    def insert_many(self, records: list[DetectionRecord]) -> list[int]:
        return [self.insert(r) for r in records]

    def get_all(self, limit: Optional[int] = None) -> list[sqlite3.Row]:
        conn = get_connection(self.config)
        try:
            query = "SELECT * FROM detections ORDER BY id DESC"
            if limit:
                query += f" LIMIT {int(limit)}"
            return conn.execute(query).fetchall()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to fetch detections: {e}") from e
        finally:
            conn.close()

    def get_by_source_file(self, source_file: str) -> list[sqlite3.Row]:
        conn = get_connection(self.config)
        try:
            return conn.execute(
                "SELECT * FROM detections WHERE source_file = ? ORDER BY frame_number ASC",
                (source_file,),
            ).fetchall()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to fetch detections for '{source_file}': {e}") from e
        finally:
            conn.close()

    def summary_stats(self) -> dict:
        conn = get_connection(self.config)
        try:
            total = conn.execute("SELECT COUNT(*) AS c FROM detections").fetchone()["c"]
            potholes = conn.execute(
                "SELECT COUNT(*) AS c FROM detections WHERE damage_type = 'pothole'"
            ).fetchone()["c"]
            cracks = conn.execute(
                "SELECT COUNT(*) AS c FROM detections WHERE damage_type = 'crack'"
            ).fetchone()["c"]
            avg_severity_row = conn.execute("SELECT AVG(severity_score) AS a FROM detections").fetchone()
            avg_severity = avg_severity_row["a"] if avg_severity_row["a"] is not None else 0.0
            high_critical = conn.execute(
                "SELECT COUNT(*) AS c FROM detections WHERE severity_category IN ('High', 'Critical')"
            ).fetchone()["c"]
            return {
                "total_detections": total,
                "pothole_count": potholes,
                "crack_count": cracks,
                "average_severity": round(avg_severity, 2),
                "high_critical_count": high_critical,
            }
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to compute summary stats: {e}") from e
        finally:
            conn.close()

    def delete_all(self) -> None:
        conn = get_connection(self.config)
        try:
            conn.execute("DELETE FROM detections")
            conn.commit()
        except sqlite3.Error as e:
            conn.rollback()
            raise DatabaseError(f"Failed to clear detections: {e}") from e
        finally:
            conn.close()
