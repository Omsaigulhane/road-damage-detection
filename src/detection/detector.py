"""
Core YOLO detector wrapper.

This module is the only place that talks to the Ultralytics API, so the
rest of the app (CLI, dashboard) never depends on YOLO internals directly
(Section 31, 33).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import get_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = {0: "pothole", 1: "crack"}


class DetectorError(Exception):
    """Raised for detector load/inference failures. Never swallowed silently."""


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width_px(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height_px(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area_px(self) -> float:
        return self.width_px * self.height_px


@dataclass
class InferenceResult:
    detections: list[Detection] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    inference_time_ms: float = 0.0


def resolve_device(requested: str = "auto") -> str:
    """Resolve 'auto' to cuda or cpu. Never assumes CUDA exists (Section 36)."""
    if requested not in ("auto", "cuda", "cpu"):
        raise DetectorError(f"Invalid device '{requested}'. Must be one of: auto, cuda, cpu.")

    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError as e:
        raise DetectorError(
            "PyTorch is not installed. Run: pip install -r requirements.txt"
        ) from e

    cuda_available = torch.cuda.is_available()

    if requested == "cuda" and not cuda_available:
        raise DetectorError(
            "CUDA was requested (device: cuda) but no CUDA-capable GPU was detected. "
            "Set model.device to 'auto' or 'cpu' in config.yaml, or verify your PyTorch "
            "build includes CUDA support."
        )

    device = "cuda" if cuda_available else "cpu"
    logger.info(f"Selected device: {device}")
    return device


class RoadDamageDetector:
    """Loads a YOLO model and runs pothole/crack detection."""

    def __init__(self, weights_path: str | None = None, config: dict | None = None):
        self.config = config or get_config()
        model_cfg = self.config.get("model", {})

        self.confidence_threshold = model_cfg.get("confidence_threshold", 0.35)
        self.iou_threshold = model_cfg.get("iou_threshold", 0.45)
        self.image_size = model_cfg.get("image_size", 640)
        self.device = resolve_device(model_cfg.get("device", "auto"))

        weights = weights_path or model_cfg.get("trained_weights") or model_cfg.get("name")
        self.weights_path = weights
        self.model = self._load_model(weights)

    def _load_model(self, weights: str):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise DetectorError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from e

        # If `weights` points at a local file that exists relative to the
        # project root (e.g. a trained checkpoint), resolve it. Otherwise
        # pass it through unchanged so Ultralytics can resolve/download a
        # named pretrained model (e.g. "yolo26n.pt").
        local_candidate = resolve_path(weights)
        resolved = str(local_candidate) if local_candidate.exists() else weights

        try:
            model = YOLO(resolved)
        except Exception as e:
            raise DetectorError(
                f"Failed to load model weights '{weights}'. "
                f"Ensure the file exists locally or is a valid Ultralytics model name "
                f"(e.g. 'yolo26n.pt'). Original error: {e}"
            ) from e

        logger.info(f"Loaded model: {weights} on device: {self.device}")
        return model

    def detect(self, image: np.ndarray) -> InferenceResult:
        """Run detection on a single BGR image (as returned by cv2.imread)."""
        if image is None or image.size == 0:
            raise DetectorError("Cannot run detection on an empty/None image.")

        h, w = image.shape[:2]
        start = time.perf_counter()

        try:
            results = self.model.predict(
                source=image,
                imgsz=self.image_size,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            raise DetectorError(f"Inference failed: {e}") from e

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections: list[Detection] = []
        if results:
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    detections.append(
                        Detection(
                            class_id=cls_id,
                            class_name=class_name,
                            confidence=conf,
                            x1=x1, y1=y1, x2=x2, y2=y2,
                        )
                    )

        return InferenceResult(
            detections=detections,
            image_width=w,
            image_height=h,
            inference_time_ms=elapsed_ms,
        )
