#!/usr/bin/env python3
"""
Fine-tuning script (Section 13-14). Fine-tunes a pretrained lightweight
YOLO model (never trains from scratch) on the pothole/crack dataset.

Usage:
    python scripts/train.py --model yolo26n.pt --data data/dataset.yaml \
        --epochs 100 --batch 16 --imgsz 640 --device auto

Run `python scripts/validate_dataset.py` first — training will refuse to
start if the dataset has critical issues.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_config, load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.detection.detector import resolve_device, DetectorError  # noqa: E402

logger = get_logger("train")


def check_gpu_memory(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        free_gb, total_gb = free / 1e9, total / 1e9
        logger.info(f"GPU memory: {free_gb:.2f} GB free / {total_gb:.2f} GB total.")
        if total_gb <= 4.5:
            logger.warning(
                "Detected ~4GB-class GPU (e.g. RTX 3050). Recommended settings: "
                "batch_size<=16, imgsz=640, amp=true. Reduce batch_size to 8 if you hit "
                "CUDA out-of-memory errors."
            )
    except Exception as e:
        logger.warning(f"Could not query GPU memory: {e}")


def run_dataset_validation(dataset_yaml: str) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "validate_dataset.py"), "--dataset-yaml", dataset_yaml],
    )
    if result.returncode != 0:
        logger.error("Dataset validation failed. Aborting training. Fix dataset issues and re-run.")
        sys.exit(1)


def main():
    cfg = get_config()
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    dataset_cfg = cfg.get("dataset", {})

    parser = argparse.ArgumentParser(description="Fine-tune YOLO on the road-damage dataset.")
    parser.add_argument("--model", default=model_cfg.get("name", "yolo26n.pt"))
    parser.add_argument("--data", default=dataset_cfg.get("config_path", "data/dataset.yaml"))
    parser.add_argument("--imgsz", type=int, default=model_cfg.get("image_size", 640))
    parser.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 100))
    parser.add_argument("--batch", type=int, default=train_cfg.get("batch_size", 16))
    parser.add_argument("--workers", type=int, default=train_cfg.get("workers", 4))
    parser.add_argument("--lr0", type=float, default=train_cfg.get("learning_rate") or None)
    parser.add_argument("--device", default=model_cfg.get("device", "auto"))
    parser.add_argument("--output-dir", default=train_cfg.get("output_dir", "models/trained"))
    parser.add_argument("--patience", type=int, default=train_cfg.get("patience", 20))
    parser.add_argument("--seed", type=int, default=cfg.get("project", {}).get("seed", 42))
    parser.add_argument("--skip-validation", action="store_true", help="Skip dataset validation (not recommended).")
    args = parser.parse_args()

    dataset_yaml_path = resolve_path(args.data)
    if not dataset_yaml_path.exists():
        logger.error(
            f"Dataset config not found at {dataset_yaml_path}. "
            "Prepare your dataset and create data/dataset.yaml first "
            "(see README 'Dataset preparation')."
        )
        sys.exit(1)

    if not args.skip_validation:
        run_dataset_validation(str(dataset_yaml_path))

    try:
        device = resolve_device(args.device)
    except DetectorError as e:
        logger.error(str(e))
        sys.exit(1)

    check_gpu_memory(device)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fine-tuning {args.model} on {dataset_yaml_path} | device={device} | epochs={args.epochs} | batch={args.batch}")

    model = YOLO(args.model)  # loads pretrained COCO weights, NOT training from scratch

    train_kwargs = dict(
        data=str(dataset_yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(output_dir),
        name="run",
        patience=args.patience,
        seed=args.seed,
        amp=cfg.get("training", {}).get("amp", True),
        exist_ok=True,
    )
    if args.lr0:
        train_kwargs["lr0"] = args.lr0

    try:
        results = model.train(**train_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(
                "CUDA out of memory. Reduce --batch (try 8 or 4) or --imgsz, then retry."
            )
        else:
            logger.error(f"Training failed: {e}")
        sys.exit(1)

    logger.info(f"Training complete. Results saved under: {output_dir / 'run'}")
    logger.info("Best weights: models/trained/run/weights/best.pt")
    logger.info("Update config.yaml -> model.trained_weights with this path to use it for inference.")


if __name__ == "__main__":
    main()
