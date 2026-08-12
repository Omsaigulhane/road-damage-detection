"""Bounding-box size math (Section 19) and annotated-image rendering."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.detection.detector import Detection

BOX_COLORS = {
    "pothole": (0, 0, 255),   # red (BGR)
    "crack": (0, 165, 255),   # orange (BGR)
}


@dataclass
class BoxMetrics:
    width_px: float
    height_px: float
    area_px: float
    relative_width: float
    relative_height: float
    relative_area: float


def compute_box_metrics(detection: Detection, image_width: int, image_height: int) -> BoxMetrics:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")

    width_px = detection.width_px
    height_px = detection.height_px
    area_px = detection.area_px
    image_area = float(image_width * image_height)

    return BoxMetrics(
        width_px=width_px,
        height_px=height_px,
        area_px=area_px,
        relative_width=width_px / image_width,
        relative_height=height_px / image_height,
        relative_area=area_px / image_area if image_area > 0 else 0.0,
    )


def draw_detections(image: np.ndarray, detections: list[Detection], severities: dict[int, dict] | None = None) -> np.ndarray:
    """Draw bounding boxes + labels (class, confidence, severity) on a copy of the image."""
    annotated = image.copy()
    severities = severities or {}

    for idx, det in enumerate(detections):
        color = BOX_COLORS.get(det.class_name, (255, 255, 255))
        p1 = (int(round(det.x1)), int(round(det.y1)))
        p2 = (int(round(det.x2)), int(round(det.y2)))
        cv2.rectangle(annotated, p1, p2, color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        sev = severities.get(idx)
        if sev is not None:
            label += f" | sev {sev['severity_score']:.0f} ({sev['severity_category']})"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_bg_p1 = (p1[0], max(0, p1[1] - th - baseline - 4))
        text_bg_p2 = (p1[0] + tw + 4, p1[1])
        cv2.rectangle(annotated, text_bg_p1, text_bg_p2, color, -1)
        cv2.putText(
            annotated, label, (p1[0] + 2, p1[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return annotated
