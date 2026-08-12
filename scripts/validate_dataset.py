#!/usr/bin/env python3
"""
Dataset validation script (Section 11).

Checks for missing images/labels, malformed labels, invalid class IDs,
out-of-range boxes, empty annotation files, corrupted images, duplicate
files, and class imbalance. Generates a validation report and produces a
dataset report (Section 9) with real, measured counts only -- never
invented statistics.

Usage:
    python scripts/validate_dataset.py --dataset-yaml data/dataset.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("validate_dataset")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


class DatasetValidationError(Exception):
    pass


def _find_label_for_image(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / (image_path.stem + ".txt")


def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def validate_split(images_dir: Path, labels_dir: Path, num_classes: int, split_name: str) -> dict:
    report = {
        "split": split_name,
        "num_images": 0,
        "num_annotations": 0,
        "class_counts": Counter(),
        "missing_images": [],
        "missing_labels": [],
        "malformed_labels": [],
        "invalid_class_ids": [],
        "out_of_range_boxes": [],
        "empty_annotation_files": [],
        "corrupted_images": [],
        "duplicate_files": [],
    }

    if not images_dir.exists():
        report["missing_images"].append(f"Images directory does not exist: {images_dir}")
        return report

    if not labels_dir.exists():
        report["missing_labels"].append(f"Labels directory does not exist: {labels_dir}")
        return report

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    report["num_images"] = len(image_files)

    seen_hashes: dict[str, str] = {}

    for img_path in image_files:
        # corrupted image check
        img = cv2.imread(str(img_path))
        if img is None:
            report["corrupted_images"].append(str(img_path))
            continue

        # duplicate check (content hash)
        file_hash = _hash_file(img_path)
        if file_hash in seen_hashes:
            report["duplicate_files"].append(f"{img_path} duplicates {seen_hashes[file_hash]}")
        else:
            seen_hashes[file_hash] = str(img_path)

        label_path = _find_label_for_image(img_path, labels_dir)
        if not label_path.exists():
            report["missing_labels"].append(str(img_path))
            continue

        lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
        if not lines:
            report["empty_annotation_files"].append(str(label_path))
            continue

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                report["malformed_labels"].append(f"{label_path}: '{line}'")
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = (float(x) for x in parts[1:])
            except ValueError:
                report["malformed_labels"].append(f"{label_path}: '{line}'")
                continue

            if cls_id < 0 or cls_id >= num_classes:
                report["invalid_class_ids"].append(f"{label_path}: class_id={cls_id}")
                continue

            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                report["out_of_range_boxes"].append(f"{label_path}: '{line}'")
                continue

            report["class_counts"][cls_id] += 1
            report["num_annotations"] += 1

    # orphan labels (label exists, image missing) -- check reverse direction
    label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    image_stems = {p.stem for p in image_files}
    for lp in label_files:
        if lp.stem not in image_stems:
            report["missing_images"].append(str(lp))

    report["class_counts"] = dict(report["class_counts"])
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate the YOLO road-damage dataset.")
    parser.add_argument("--dataset-yaml", default=str(REPO_ROOT / "data" / "dataset.yaml"))
    parser.add_argument("--report-out", default=str(REPO_ROOT / "outputs" / "reports" / "dataset_validation_report.json"))
    args = parser.parse_args()

    dataset_yaml_path = Path(args.dataset_yaml)
    if not dataset_yaml_path.exists():
        logger.error(f"dataset.yaml not found at {dataset_yaml_path}. Run dataset preparation first.")
        sys.exit(1)

    with open(dataset_yaml_path) as f:
        ds_cfg = yaml.safe_load(f)

    names = ds_cfg.get("names", {})
    num_classes = len(names)
    base_path = Path(ds_cfg.get("path", dataset_yaml_path.parent))
    if not base_path.is_absolute():
        base_path = (dataset_yaml_path.parent / base_path).resolve()

    splits = {}
    for split_key in ("train", "val", "test"):
        rel = ds_cfg.get(split_key)
        if not rel:
            continue
        images_dir = (base_path / rel).resolve()
        labels_dir = Path(str(images_dir).replace("images", "labels"))
        splits[split_key] = validate_split(images_dir, labels_dir, num_classes, split_key)

    critical_issues = 0
    for split_name, r in splits.items():
        critical_issues += len(r["missing_images"]) + len(r["missing_labels"]) + len(r["corrupted_images"])
        logger.info(
            f"[{split_name}] images={r['num_images']} annotations={r['num_annotations']} "
            f"class_counts={r['class_counts']}"
        )
        for issue_type in ("malformed_labels", "invalid_class_ids", "out_of_range_boxes", "empty_annotation_files", "duplicate_files"):
            if r[issue_type]:
                logger.warning(f"[{split_name}] {issue_type}: {len(r[issue_type])} issue(s)")

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Validation report written to: {report_path}")

    if critical_issues > 0:
        logger.error(
            f"Dataset validation FAILED with {critical_issues} critical issue(s) "
            "(missing images/labels or corrupted files). Fix these before training."
        )
        sys.exit(1)

    logger.info("Dataset validation PASSED (no critical issues).")


if __name__ == "__main__":
    main()
