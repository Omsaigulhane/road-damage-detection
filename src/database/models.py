"""Data model for a single detection record (Section 27)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionRecord:
    id: Optional[int]
    timestamp: str
    source_file: str
    source_type: str            # "image" | "video"
    frame_number: Optional[int]
    video_timestamp: Optional[float]
    damage_type: str            # "pothole" | "crack"
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    width_px: float
    height_px: float
    area_px: float
    relative_area: float
    estimated_width_m: Optional[float]
    estimated_height_m: Optional[float]
    severity_score: float
    severity_category: str
    # Future fields (Phase 2) — nullable, unused in Phase 1 (Section 27, 43)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    vibration_magnitude: Optional[float] = None
    vehicle_speed: Optional[float] = None
