#!/usr/bin/env python3
"""
CLI inference entrypoint (Section 16-17). Works standalone without Streamlit
(Section 31).

Usage:
    python scripts/detect.py --source path/to/image.jpg
    python scripts/detect.py --source path/to/video.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from src.pipeline import RoadDamagePipeline  # noqa: E402
from src.preprocessing.image_processor import load_image, ImageProcessingError, SUPPORTED_IMAGE_EXTENSIONS  # noqa: E402
from src.video.video_processor import VideoProcessor, VideoProcessingError, SUPPORTED_VIDEO_EXTENSIONS  # noqa: E402
from src.database.repository import DetectionRepository  # noqa: E402
from src.utils.config import get_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.detection.detector import DetectorError  # noqa: E402

logger = get_logger("detect")


def handle_image(source: Path, pipeline: RoadDamagePipeline, save: bool, store_db: bool):
    image = load_image(source)
    outcomes, annotated, latency_ms = pipeline.process_image(image)

    logger.info(f"Detected {len(outcomes)} object(s) in {source.name} | inference: {latency_ms:.1f} ms")
    for o in outcomes:
        logger.info(
            f"  {o.detection.class_name} | conf={o.detection.confidence:.2f} | "
            f"size_px={o.metrics.width_px:.0f}x{o.metrics.height_px:.0f} | "
            f"est_dim={o.dimension.estimated_width_m}x{o.dimension.estimated_height_m} m "
            f"({'calibrated' if o.dimension.calibration_available else 'uncalibrated'}) | "
            f"severity={o.severity.severity_score:.1f} ({o.severity.severity_category})"
        )

    if save:
        cfg = get_config()
        out_dir = resolve_path(cfg.get("outputs", {}).get("images_dir", "outputs/images"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"annotated_{source.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)
        logger.info(f"Annotated image saved to: {out_path}")

    if store_db:
        repo = DetectionRepository()
        for o in outcomes:
            repo.insert(o.to_record(source_file=str(source), source_type="image"))
        logger.info(f"Stored {len(outcomes)} record(s) in database.")

    return outcomes


def handle_video(source: Path, pipeline: RoadDamagePipeline, save: bool, store_db: bool):
    processor = VideoProcessor(pipeline)
    cfg = get_config()
    out_dir = resolve_path(cfg.get("video", {}).get("output_dir", "outputs/videos"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"annotated_{source.stem}.mp4" if save else None

    total_detections = 0

    def on_frame(fr):
        nonlocal total_detections
        total_detections += len(fr.outcomes)
        if fr.outcomes:
            logger.info(f"Frame {fr.frame_number} (t={fr.video_timestamp_s:.2f}s): {len(fr.outcomes)} detection(s)")

    frame_results = processor.process(source, output_path=out_path, on_frame=on_frame)

    logger.info(f"Processed {len(frame_results)} sampled frame(s), {total_detections} total detection(s).")
    if save and out_path:
        logger.info(f"Annotated video saved to: {out_path}")

    if store_db:
        repo = DetectionRepository()
        count = 0
        for fr in frame_results:
            for o in fr.outcomes:
                repo.insert(o.to_record(
                    source_file=str(source), source_type="video",
                    frame_number=fr.frame_number, video_timestamp=fr.video_timestamp_s,
                ))
                count += 1
        logger.info(f"Stored {count} record(s) in database.")

    return frame_results


def main():
    parser = argparse.ArgumentParser(description="Run road-damage detection on an image or video.")
    parser.add_argument("--source", required=True, help="Path to an image or video file.")
    parser.add_argument("--weights", default=None, help="Override model weights path.")
    parser.add_argument("--no-save", action="store_true", help="Don't save annotated output.")
    parser.add_argument("--no-store", action="store_true", help="Don't store results in the database.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        logger.error(f"Source file not found: {source}")
        sys.exit(1)

    try:
        pipeline = RoadDamagePipeline(weights_path=args.weights)
    except DetectorError as e:
        logger.error(str(e))
        sys.exit(1)

    suffix = source.suffix.lower()
    try:
        if suffix in SUPPORTED_IMAGE_EXTENSIONS:
            handle_image(source, pipeline, save=not args.no_save, store_db=not args.no_store)
        elif suffix in SUPPORTED_VIDEO_EXTENSIONS:
            handle_video(source, pipeline, save=not args.no_save, store_db=not args.no_store)
        else:
            logger.error(
                f"Unsupported file extension '{suffix}'. "
                f"Supported images: {sorted(SUPPORTED_IMAGE_EXTENSIONS)} | "
                f"Supported videos: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
            )
            sys.exit(1)
    except (ImageProcessingError, VideoProcessingError, DetectorError) as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
