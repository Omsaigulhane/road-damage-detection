#!/usr/bin/env python3
"""
Model evaluation script (Section 15). Reports ONLY measured results --
never invents target performance (Section 40).

Usage:
    python scripts/evaluate.py --weights models/trained/run/weights/best.pt \
        --data data/dataset.yaml --split test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.detection.detector import resolve_device, DetectorError  # noqa: E402

logger = get_logger("evaluate")


def main():
    cfg = get_config()
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})

    parser = argparse.ArgumentParser(description="Evaluate a trained model on the road-damage dataset.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--data", default=dataset_cfg.get("config_path", "data/dataset.yaml"))
    parser.add_argument("--imgsz", type=int, default=model_cfg.get("image_size", 640))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=model_cfg.get("device", "auto"))
    parser.add_argument("--report-out", default=str(REPO_ROOT / "outputs" / "reports" / "evaluation_report.json"))
    args = parser.parse_args()

    weights_path = resolve_path(args.weights)
    if not weights_path.exists():
        logger.error(f"Weights file not found: {weights_path}. Train a model first (scripts/train.py).")
        sys.exit(1)

    dataset_yaml_path = resolve_path(args.data)
    if not dataset_yaml_path.exists():
        logger.error(f"Dataset config not found: {dataset_yaml_path}")
        sys.exit(1)

    try:
        device = resolve_device(args.device)
    except DetectorError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    model = YOLO(str(weights_path))

    logger.info(f"Evaluating on '{args.split}' split | device={device} | imgsz={args.imgsz}")
    metrics = model.val(data=str(dataset_yaml_path), imgsz=args.imgsz, split=args.split, device=device)

    # ---- Aggregate metrics (measured, not invented) ----
    result = {
        "weights": str(weights_path),
        "split": args.split,
        "precision": None,
        "recall": None,
        "f1": None,
        "map50": None,
        "map50_95": None,
        "class_wise": {},
        "inference_latency_ms": None,
        "fps": None,
        "model_parameters": None,
        "model_size_mb": None,
        "note": "All values below are measured on this run. Nothing here is a target or invented figure.",
    }

    try:
        result["precision"] = float(metrics.box.mp)
        result["recall"] = float(metrics.box.mr)
        p, r = result["precision"], result["recall"]
        result["f1"] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        result["map50"] = float(metrics.box.map50)
        result["map50_95"] = float(metrics.box.map)

        names = metrics.names
        for i, class_name in names.items():
            try:
                result["class_wise"][class_name] = {
                    "precision": float(metrics.box.p[i]) if len(metrics.box.p) > i else None,
                    "recall": float(metrics.box.r[i]) if len(metrics.box.r) > i else None,
                    "ap50": float(metrics.box.ap50[i]) if len(metrics.box.ap50) > i else None,
                }
            except (IndexError, TypeError):
                result["class_wise"][class_name] = "Not measured yet."
    except AttributeError as e:
        logger.warning(f"Some metrics unavailable from this Ultralytics version: {e}")

    # ---- Latency / FPS measurement (single-image, on real dataset sample) ----
    try:
        import cv2
        val_images_dir = None
        with open(dataset_yaml_path) as f:
            import yaml
            ds = yaml.safe_load(f)
        base_path = Path(ds.get("path", dataset_yaml_path.parent))
        if not base_path.is_absolute():
            base_path = (dataset_yaml_path.parent / base_path).resolve()
        split_rel = ds.get(args.split)
        if split_rel:
            val_images_dir = (base_path / split_rel).resolve()

        if val_images_dir and val_images_dir.exists():
            sample_images = list(val_images_dir.glob("*.jpg"))[:20] or list(val_images_dir.glob("*.png"))[:20]
            if sample_images:
                # warmup
                img0 = cv2.imread(str(sample_images[0]))
                model.predict(img0, imgsz=args.imgsz, device=device, verbose=False)

                times = []
                for img_path in sample_images:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    start = time.perf_counter()
                    model.predict(img, imgsz=args.imgsz, device=device, verbose=False)
                    times.append(time.perf_counter() - start)

                if times:
                    avg_latency_s = sum(times) / len(times)
                    result["inference_latency_ms"] = round(avg_latency_s * 1000, 2)
                    result["fps"] = round(1.0 / avg_latency_s, 2) if avg_latency_s > 0 else None
                    result["latency_measured_on"] = f"{len(times)} images from '{args.split}' split, device={device}"
        else:
            result["inference_latency_ms"] = "Not measured yet."
            result["fps"] = "Not measured yet."
    except Exception as e:
        logger.warning(f"Could not measure inference latency: {e}")
        result["inference_latency_ms"] = "Not measured yet."
        result["fps"] = "Not measured yet."

    # ---- Model size / params ----
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
        result["model_parameters"] = n_params
        result["model_size_mb"] = round(weights_path.stat().st_size / (1024 * 1024), 2)
    except Exception as e:
        logger.warning(f"Could not measure model size: {e}")

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Evaluation report written to: {report_path}")
    logger.info(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
