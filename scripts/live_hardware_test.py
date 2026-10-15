#!/usr/bin/env python3
"""
Desk-test the full live loop: fetch a frame from the ESP32-CAM, fetch a
live vibration reading from the sensor ESP32, run detection+severity,
and display the result. No vehicle mounting required for this test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import requests

from src.pipeline import RoadDamagePipeline
from src.sensors.future_mpu6050 import ESP32SensorBoard, MPU6050Provider
from src.utils.config import get_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("live_hardware_test")


def fetch_camera_frame(camera_base_url: str) -> np.ndarray:
    resp = requests.get(f"{camera_base_url}/capture", timeout=5.0)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode image from ESP32-CAM /capture endpoint.")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-url", required=True, help="e.g. http://10.188.x.x")
    args = parser.parse_args()

    cfg = get_config()
    sensor_url = cfg.get("hardware", {}).get("sensor_board_url")
    if not sensor_url:
        logger.error("config.yaml is missing hardware.sensor_board_url")
        sys.exit(1)

    board = ESP32SensorBoard(sensor_url)
    mpu = MPU6050Provider(board)

    logger.info(f"Fetching frame from {args.camera_url}/capture ...")
    image = fetch_camera_frame(args.camera_url)
    logger.info(f"Frame captured: {image.shape[1]}x{image.shape[0]}")

    logger.info("Reading live vibration from sensor board ...")
    vib = mpu.read()
    vib_magnitude = vib.magnitude if vib else None
    logger.info(f"Vibration reading: {vib}")

    pipeline = RoadDamagePipeline()
    outcomes, annotated, latency_ms = pipeline.process_image(image, vibration_magnitude_g=vib_magnitude)

    logger.info(f"Detected {len(outcomes)} object(s) | inference: {latency_ms:.1f} ms")
    for o in outcomes:
        logger.info(
            f"  {o.detection.class_name} | conf={o.detection.confidence:.2f} | "
            f"severity={o.severity.severity_score:.1f} ({o.severity.severity_category}) | "
            f"vibration_score={o.severity.vibration_score:.3f} "
            f"({'fallback' if o.severity.vibration_score_is_fallback else 'live'})"
        )

    out_dir = resolve_path("outputs/images")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "live_hardware_test.jpg"
    cv2.imwrite(str(out_path), annotated)
    logger.info(f"Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()