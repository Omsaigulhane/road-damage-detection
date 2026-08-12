"""Video inference pipeline (Section 17, 18)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2

from src.pipeline import RoadDamagePipeline, DetectionOutcome
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class VideoProcessingError(Exception):
    pass


@dataclass
class FrameResult:
    frame_number: int
    video_timestamp_s: float
    outcomes: list[DetectionOutcome]


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class VideoProcessor:
    """
    Processes a video with configurable frame sampling (Section 17).

    Deliberately does NOT implement full multi-object tracking in Phase 1
    (Section 18). An optional, simple IoU + temporal-proximity aggregation
    can be enabled in config.yaml to reduce trivial frame-to-frame
    duplicate counting, but it is not a dependency for Phase-1 completion.
    """

    def __init__(self, pipeline: RoadDamagePipeline, config: dict | None = None):
        self.pipeline = pipeline
        self.config = config or get_config()
        video_cfg = self.config.get("video", {})
        self.frame_sample_rate = max(1, video_cfg.get("frame_sample_rate", 5))
        agg_cfg = video_cfg.get("event_aggregation", {})
        self.aggregation_enabled = agg_cfg.get("enabled", False)
        self.agg_iou_threshold = agg_cfg.get("iou_threshold", 0.5)
        self.agg_max_frame_gap = agg_cfg.get("max_frame_gap", 10)

    def process(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        on_frame: Optional[Callable[[FrameResult], None]] = None,
    ) -> list[FrameResult]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise VideoProcessingError(f"Video file not found: {video_path}")
        if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            raise VideoProcessingError(
                f"Unsupported video format '{video_path.suffix}'. "
                f"Supported: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Failed to open video (possibly corrupted): {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise VideoProcessingError(f"Video appears empty or unreadable: {video_path}")

        writer = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 25.0, (width, height))
            if not writer.isOpened():
                cap.release()
                raise VideoProcessingError(f"Failed to open output video writer at: {output_path}")

        frame_results: list[FrameResult] = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.frame_sample_rate == 0:
                    outcomes, annotated, _ = self.pipeline.process_image(frame)
                    video_timestamp_s = frame_idx / fps if fps > 0 else 0.0
                    fr = FrameResult(frame_number=frame_idx, video_timestamp_s=video_timestamp_s, outcomes=outcomes)
                    frame_results.append(fr)
                    if on_frame:
                        on_frame(fr)
                    if writer:
                        writer.write(annotated)
                else:
                    if writer:
                        writer.write(frame)  # pass-through frame, not re-detected

                frame_idx += 1
        finally:
            cap.release()
            if writer:
                writer.release()

        if self.aggregation_enabled:
            frame_results = self._aggregate_events(frame_results)

        return frame_results

    def _aggregate_events(self, frame_results: list[FrameResult]) -> list[FrameResult]:
        """Optional lightweight event aggregation using IoU + temporal proximity
        + same class (Section 18). Keeps only the highest-confidence detection
        per aggregated event, per frame group — a simple heuristic, not a
        tracker."""
        seen_events: list[dict] = []
        for fr in frame_results:
            keep = []
            for outcome in fr.outcomes:
                det = outcome.detection
                box = (det.x1, det.y1, det.x2, det.y2)
                matched = False
                for event in seen_events:
                    if (
                        event["class_name"] == det.class_name
                        and fr.frame_number - event["last_frame"] <= self.agg_max_frame_gap
                        and _iou(box, event["box"]) >= self.agg_iou_threshold
                    ):
                        matched = True
                        event["last_frame"] = fr.frame_number
                        event["box"] = box
                        if det.confidence <= event["confidence"]:
                            break  # duplicate event, lower/equal confidence -> drop this detection
                        else:
                            event["confidence"] = det.confidence
                if not matched:
                    seen_events.append(
                        {"class_name": det.class_name, "box": box, "last_frame": fr.frame_number, "confidence": det.confidence}
                    )
                    keep.append(outcome)
            fr.outcomes = keep
        return frame_results
