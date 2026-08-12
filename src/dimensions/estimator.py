"""
Approximate Dimension Estimator (Section 20).

This project has NO depth sensor, stereo camera, or calibrated
vehicle-camera geometry. Physical dimensions can therefore never be
claimed as precise measurements — only approximations under stated
assumptions.

Method (when calibration is available): a simple pinhole / flat-ground-plane
projection.

    real_size_m = (pixel_size * camera_height_m) / focal_length_px

This assumes:
  1. The camera looks approximately straight down at the road plane
     (or the camera_height_m / focal_length_px ratio has already been
     adjusted by the user to account for camera tilt).
  2. The road surface is locally flat under the defect.
  3. Lens distortion is negligible.

These are stated, documented assumptions — NOT validated calibration.
If focal_length_px or camera_height_m is missing, no physical estimate
is produced; only pixel measurements are reported, and the result is
explicitly marked unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.utils.config import get_config


@dataclass
class DimensionEstimate:
    pixel_width: float
    pixel_height: float
    estimated_width_m: Optional[float]
    estimated_height_m: Optional[float]
    calibration_available: bool
    method: str
    note: str


class DimensionEstimator:
    def __init__(self, config: dict | None = None):
        cfg = (config or get_config()).get("dimension_estimation", {})
        self.enabled: bool = cfg.get("enabled", True)
        self.method: str = cfg.get("method", "pinhole_ground_plane")
        self.reference_width_m = cfg.get("reference_width_m")
        self.camera_height_m = cfg.get("camera_height_m")
        self.focal_length_px = cfg.get("focal_length_px")

    @property
    def calibration_available(self) -> bool:
        return self.camera_height_m is not None and self.focal_length_px is not None

    def estimate(self, width_px: float, height_px: float) -> DimensionEstimate:
        if not self.enabled:
            return DimensionEstimate(
                pixel_width=width_px,
                pixel_height=height_px,
                estimated_width_m=None,
                estimated_height_m=None,
                calibration_available=False,
                method=self.method,
                note="Dimension estimation disabled in config.",
            )

        if not self.calibration_available:
            return DimensionEstimate(
                pixel_width=width_px,
                pixel_height=height_px,
                estimated_width_m=None,
                estimated_height_m=None,
                calibration_available=False,
                method=self.method,
                note=(
                    "Calibration unavailable (camera_height_m / focal_length_px not set "
                    "in config.yaml). Only pixel-space measurements are reported. This is "
                    "NOT a physical measurement."
                ),
            )

        scale = self.camera_height_m / self.focal_length_px  # meters per pixel, approximate
        estimated_width_m = round(width_px * scale, 4)
        estimated_height_m = round(height_px * scale, 4)

        return DimensionEstimate(
            pixel_width=width_px,
            pixel_height=height_px,
            estimated_width_m=estimated_width_m,
            estimated_height_m=estimated_height_m,
            calibration_available=True,
            method=self.method,
            note=(
                "Approximate estimate under a flat-ground pinhole-camera assumption. "
                "Not a calibrated physical measurement — accuracy depends entirely on "
                "the correctness of camera_height_m and focal_length_px."
            ),
        )
