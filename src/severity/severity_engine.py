"""
Transparent mathematical Severity Engine (Sections 21-26).

Deliberately NOT a trained neural network. Phase 1 uses three inputs
(size, dimension, confidence). Phase 2 adds an optional fourth input,
vibration_score, derived from real MPU6050 readings (src/sensors/future_mpu6050.py).

Vibration's weight defaults to 0.0 in config until real drive-test data
exists to calibrate a baseline/max range — adding the input now (with
zero weight) lets the code path be tested without fabricating a
calibration that hasn't been measured yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.utils.config import get_config

DISCLAIMER = (
    "Severity is a rule/weight-based engineering score derived from image "
    "geometry, detector confidence, and (if configured and available) vehicle "
    "vibration. It is NOT a scientifically validated physical damage assessment."
)


class SeverityConfigError(Exception):
    pass


@dataclass
class SeverityResult:
    severity_score: float
    severity_category: str
    size_score: float
    dimension_score: float
    confidence_score: float
    vibration_score: float
    dimension_score_is_fallback: bool
    vibration_score_is_fallback: bool
    weights_used: dict
    disclaimer: str = DISCLAIMER


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SeverityEngine:
    def __init__(self, config: dict | None = None):
        cfg = (config or get_config()).get("severity", {})
        if not cfg:
            raise SeverityConfigError("Missing 'severity' section in config.yaml.")

        self.weights = cfg.get("weights", {})
        self.thresholds = cfg.get("thresholds", {"low": 25, "moderate": 50, "high": 75})
        norm = cfg.get("normalization", {})
        self.size_reference_relative_area = norm.get("size_reference_relative_area", 0.06)
        self.dimension_reference_m = norm.get("dimension_reference_m", {"pothole": 0.6, "crack": 1.5})
        # Vibration normalization: baseline_g (at-rest reading) and max_g
        # (reading that saturates vibration_score to 1.0). Both are null
        # until calibrated against real drive-test data.
        vib_norm = norm.get("vibration_reference_g", {})
        self.vibration_baseline_g = vib_norm.get("baseline_g")
        self.vibration_max_g = vib_norm.get("max_g")

        for damage_type in ("pothole", "crack"):
            if damage_type not in self.weights:
                raise SeverityConfigError(
                    f"No severity weights configured for damage type '{damage_type}'."
                )
            w = self.weights[damage_type]
            total = w.get("size", 0) + w.get("dimension", 0) + w.get("confidence", 0) + w.get("vibration", 0)
            if abs(total - 1.0) > 1e-6:
                raise SeverityConfigError(
                    f"Severity weights for '{damage_type}' must sum to 1.0 (got {total})."
                )

    def _category(self, score: float) -> str:
        if score <= self.thresholds["low"]:
            return "Low"
        if score <= self.thresholds["moderate"]:
            return "Moderate"
        if score <= self.thresholds["high"]:
            return "High"
        return "Critical"

    def _compute_vibration_score(self, vibration_magnitude_g: Optional[float]) -> tuple[float, bool]:
        """Returns (vibration_score, is_fallback). Falls back to 0.0 with
        is_fallback=True when no reading is available or no calibration
        (baseline_g/max_g) has been configured yet."""
        if vibration_magnitude_g is None:
            return 0.0, True
        if self.vibration_baseline_g is None or self.vibration_max_g is None:
            return 0.0, True
        span = self.vibration_max_g - self.vibration_baseline_g
        if span <= 0:
            return 0.0, True
        score = _clamp((vibration_magnitude_g - self.vibration_baseline_g) / span, 0.0, 1.0)
        return score, False

    def compute(
        self,
        damage_type: str,
        relative_area: float,
        confidence: float,
        estimated_size_m: Optional[float] = None,
        vibration_magnitude_g: Optional[float] = None,
    ) -> SeverityResult:
        if damage_type not in ("pothole", "crack"):
            raise ValueError(f"Unknown damage_type '{damage_type}'. Must be 'pothole' or 'crack'.")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be within [0, 1], got {confidence}.")
        if relative_area < 0:
            raise ValueError("relative_area must be non-negative.")

        size_ref = self.size_reference_relative_area
        size_score = _clamp(relative_area / size_ref, 0.0, 1.0) if size_ref > 0 else 0.0

        dimension_is_fallback = False
        if estimated_size_m is not None:
            dim_ref = self.dimension_reference_m.get(damage_type, 1.0)
            dimension_score = _clamp(estimated_size_m / dim_ref, 0.0, 1.0) if dim_ref > 0 else 0.0
        else:
            dimension_score = size_score
            dimension_is_fallback = True

        confidence_score = confidence

        vibration_score, vibration_is_fallback = self._compute_vibration_score(vibration_magnitude_g)

        w = self.weights[damage_type]
        severity_raw = (
            w["size"] * size_score
            + w["dimension"] * dimension_score
            + w["confidence"] * confidence_score
            + w.get("vibration", 0.0) * vibration_score
        )
        severity_score = _clamp(severity_raw * 100.0, 0.0, 100.0)
        category = self._category(severity_score)

        return SeverityResult(
            severity_score=round(severity_score, 2),
            severity_category=category,
            size_score=round(size_score, 4),
            dimension_score=round(dimension_score, 4),
            confidence_score=round(confidence_score, 4),
            vibration_score=round(vibration_score, 4),
            dimension_score_is_fallback=dimension_is_fallback,
            vibration_score_is_fallback=vibration_is_fallback,
            weights_used=w,
        )