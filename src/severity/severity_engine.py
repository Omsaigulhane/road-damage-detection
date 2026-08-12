"""
Transparent mathematical Severity Engine (Sections 21-26).

Deliberately NOT a trained neural network — there is no severity-labelled
ground-truth dataset for this project, so a learned severity model would be
scientifically unsupported. Instead this is an explainable, fully
configurable weighted score:

    severity_raw = W_SIZE * size_score + W_DIMENSION * dimension_score
                   + W_CONFIDENCE * confidence_score
    severity_score = clamp(severity_raw * 100, 0, 100)

All weights, normalization references, and category thresholds are read
from config.yaml (never hard-coded), and are explicitly documented as
initial engineering weights, not scientifically validated ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.utils.config import get_config

DISCLAIMER = (
    "Severity is a rule/weight-based engineering score derived from image "
    "geometry and detector confidence. It is NOT a scientifically validated "
    "physical damage assessment."
)


class SeverityConfigError(Exception):
    pass


@dataclass
class SeverityResult:
    severity_score: float          # 0-100
    severity_category: str         # Low / Moderate / High / Critical
    size_score: float              # 0-1
    dimension_score: float         # 0-1
    confidence_score: float        # 0-1
    dimension_score_is_fallback: bool
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

        for damage_type in ("pothole", "crack"):
            if damage_type not in self.weights:
                raise SeverityConfigError(
                    f"No severity weights configured for damage type '{damage_type}'."
                )
            w = self.weights[damage_type]
            total = w.get("size", 0) + w.get("dimension", 0) + w.get("confidence", 0)
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

    def compute(
        self,
        damage_type: str,
        relative_area: float,
        confidence: float,
        estimated_size_m: Optional[float] = None,
    ) -> SeverityResult:
        if damage_type not in ("pothole", "crack"):
            raise ValueError(f"Unknown damage_type '{damage_type}'. Must be 'pothole' or 'crack'.")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be within [0, 1], got {confidence}.")
        if relative_area < 0:
            raise ValueError("relative_area must be non-negative.")

        # --- size_score ---
        size_ref = self.size_reference_relative_area
        size_score = _clamp(relative_area / size_ref, 0.0, 1.0) if size_ref > 0 else 0.0

        # --- dimension_score ---
        dimension_is_fallback = False
        if estimated_size_m is not None:
            dim_ref = self.dimension_reference_m.get(damage_type, 1.0)
            dimension_score = _clamp(estimated_size_m / dim_ref, 0.0, 1.0) if dim_ref > 0 else 0.0
        else:
            # No physical calibration available -> fall back to size_score
            # (documented limitation, Section 20/24), not a silent fabrication.
            dimension_score = size_score
            dimension_is_fallback = True

        # --- confidence_score ---
        # Used as a reliability weight, not a direct severity proxy (Section 26).
        confidence_score = confidence

        w = self.weights[damage_type]
        severity_raw = (
            w["size"] * size_score
            + w["dimension"] * dimension_score
            + w["confidence"] * confidence_score
        )
        severity_score = _clamp(severity_raw * 100.0, 0.0, 100.0)
        category = self._category(severity_score)

        return SeverityResult(
            severity_score=round(severity_score, 2),
            severity_category=category,
            size_score=round(size_score, 4),
            dimension_score=round(dimension_score, 4),
            confidence_score=round(confidence_score, 4),
            dimension_score_is_fallback=dimension_is_fallback,
            weights_used=w,
        )
