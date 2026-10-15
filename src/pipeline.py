"""
Shared inference pipeline: Detector -> BoxMetrics -> DimensionEstimator ->
SeverityEngine -> annotated image + structured results.

This is the "Core inference" layer referenced in Section 31 — both the CLI
(scripts/detect.py) and the Streamlit dashboard call into this module so
detection logic is never duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from src.detection.detector import RoadDamageDetector, Detection
from src.detection.postprocess import compute_box_metrics, draw_detections, BoxMetrics
from src.dimensions.estimator import DimensionEstimator, DimensionEstimate
from src.severity.severity_engine import SeverityEngine, SeverityResult
from src.database.models import DetectionRecord
from src.utils.config import get_config


@dataclass
class DetectionOutcome:
    detection: Detection
    metrics: BoxMetrics
    dimension: DimensionEstimate
    severity: SeverityResult

    def to_record(
        self,
        source_file: str,
        source_type: str,
        frame_number: Optional[int] = None,
        video_timestamp: Optional[float] = None,
    ) -> DetectionRecord:
        return DetectionRecord(
            id=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_file=source_file,
            source_type=source_type,
            frame_number=frame_number,
            video_timestamp=video_timestamp,
            damage_type=self.detection.class_name,
            confidence=self.detection.confidence,
            x1=self.detection.x1, y1=self.detection.y1,
            x2=self.detection.x2, y2=self.detection.y2,
            width_px=self.metrics.width_px,
            height_px=self.metrics.height_px,
            area_px=self.metrics.area_px,
            relative_area=self.metrics.relative_area,
            estimated_width_m=self.dimension.estimated_width_m,
            estimated_height_m=self.dimension.estimated_height_m,
            severity_score=self.severity.severity_score,
            severity_category=self.severity.severity_category,
        )


class RoadDamagePipeline:
    def __init__(self, weights_path: str | None = None, config: dict | None = None):
        self.config = config or get_config()
        self.detector = RoadDamageDetector(weights_path=weights_path, config=self.config)
        self.dimension_estimator = DimensionEstimator(config=self.config)
        self.severity_engine = SeverityEngine(config=self.config)

    def process_image(
        self,
        image: np.ndarray,
        vibration_magnitude_g: Optional[float] = None,
    ) -> tuple[list[DetectionOutcome], np.ndarray, float]:
        """Returns (outcomes, annotated_image, inference_time_ms)."""
        result = self.detector.detect(image)
        outcomes: list[DetectionOutcome] = []

        for det in result.detections:
            metrics = compute_box_metrics(
                det,
                result.image_width,
                result.image_height,
            )

            representative_px = max(metrics.width_px, metrics.height_px)

            dimension = self.dimension_estimator.estimate(
                metrics.width_px,
                metrics.height_px,
            )

            estimated_size_m = (
                dimension.estimated_width_m
                if dimension.calibration_available
                else None
            )

            if estimated_size_m is not None and dimension.estimated_height_m:
                estimated_size_m = max(
                    dimension.estimated_width_m,
                    dimension.estimated_height_m,
                )

            severity = self.severity_engine.compute(
                damage_type=det.class_name,
                relative_area=metrics.relative_area,
                confidence=det.confidence,
                estimated_size_m=estimated_size_m,
                vibration_magnitude_g=vibration_magnitude_g,
            )

            outcomes.append(
                DetectionOutcome(
                    detection=det,
                    metrics=metrics,
                    dimension=dimension,
                    severity=severity,
                )
            )

        severities_for_draw = {
            i: {
                "severity_score": o.severity.severity_score,
                "severity_category": o.severity.severity_category,
            }
            for i, o in enumerate(outcomes)
        }

        annotated = draw_detections(
            image,
            result.detections,
            severities_for_draw,
        )

        return outcomes, annotated, result.inference_time_ms