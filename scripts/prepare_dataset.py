#!/usr/bin/env python3
"""
Dataset preparation script (Section 9-10).

Converts the RDD2022 dataset (PASCAL-VOC XML annotations, 4 damage classes:
D00 longitudinal crack, D10 transverse crack, D20 alligator crack,
D40 pothole) into this project's 2-class YOLO format:

    0 = pothole   (from D40)
    1 = crack     (from D00, D10, D20 merged)

Why RDD2022: it is a real, published, multi-national (Japan/India/Czech
Republic/Norway/USA/China), publicly available dataset with genuine
bounding-box annotations for exactly the damage types this project targets,
distributed under a license that permits research/educational use.
Reference: Arya et al., "RDD2022: A multi-national image dataset for
automatic Road Damage Detection", and https://github.com/sekilab/RoadDamageDetector

IMPORTANT: This script does NOT download the dataset -- RDD2022 is
distributed as a large zip via FigShare, which is outside this project's
automated-download scope. Download it manually first:

    1. Go to https://github.com/sekilab/RoadDamageDetector
    2. Follow the link to the RDD2022 FigShare archive
    3. Download and extract RDD2022.zip
    4. Point --raw-dir at the extracted country folder(s), e.g.:
         RDD2022/Japan/train/images
         RDD2022/Japan/train/annotations/xmls

Usage:
    python scripts/prepare_dataset.py \\
        --raw-dir /path/to/RDD2022/Japan/train \\
        --output-dir data/processed \\
        --countries Japan India Czech

This script does NOT invent counts -- it reports only what it actually
finds and converts, and writes a dataset report with real numbers.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("prepare_dataset")

# RDD2022 class -> this project's 2-class scheme
RDD_CLASS_MAP = {
    "D00": "crack",   # longitudinal crack
    "D10": "crack",   # transverse crack
    "D20": "crack",   # alligator crack
    "D40": "pothole", # pothole
    # Other RDD categories (D43, D44, D50 etc.) exist in some country subsets
    # but are outside this project's 2-class scope and are skipped.
}
OUR_CLASSES = {"pothole": 0, "crack": 1}


def convert_voc_xml_to_yolo(xml_path: Path) -> tuple[list[str], Counter]:
    """Returns (list of YOLO-format label lines, per-class counter). Skips
    unmapped classes; does not fabricate boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    lines = []
    counts = Counter()

    for obj in root.findall("object"):
        rdd_class = obj.find("name").text.strip()
        our_class = RDD_CLASS_MAP.get(rdd_class)
        if our_class is None:
            continue  # not in our 2-class scope

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        xmin, xmax = max(0, xmin), min(img_w, xmax)
        ymin, ymax = max(0, ymin), min(img_h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue  # degenerate box, skip

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        cls_id = OUR_CLASSES[our_class]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        counts[our_class] += 1

    return lines, counts


def main():
    parser = argparse.ArgumentParser(description="Convert RDD2022 (VOC XML) to project 2-class YOLO format.")
    parser.add_argument("--raw-dir", required=True, help="Path to a RDD2022 country/train folder containing images/ and annotations/xmls/")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    images_dir = raw_dir / "images"
    ann_dir = raw_dir / "annotations" / "xmls"

    if not images_dir.exists() or not ann_dir.exists():
        logger.error(
            f"Expected '{images_dir}' and '{ann_dir}' to exist. "
            "Check --raw-dir points at an extracted RDD2022 country/train folder. "
            "See this script's docstring for download instructions."
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    xml_files = sorted(ann_dir.glob("*.xml"))
    if not xml_files:
        logger.error(f"No XML annotation files found in {ann_dir}.")
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(xml_files)

    n = len(xml_files)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    split_assignment = (
        [("train", f) for f in xml_files[:n_train]]
        + [("val", f) for f in xml_files[n_train:n_train + n_val]]
        + [("test", f) for f in xml_files[n_train + n_val:]]
    )

    total_counts = Counter()
    split_counts = {"train": 0, "val": 0, "test": 0}
    skipped_no_image = []
    skipped_no_annotations = []

    for split, xml_path in split_assignment:
        img_path = images_dir / (xml_path.stem + ".jpg")
        if not img_path.exists():
            img_path = images_dir / (xml_path.stem + ".png")
        if not img_path.exists():
            skipped_no_image.append(str(xml_path))
            continue

        try:
            lines, counts = convert_voc_xml_to_yolo(xml_path)
        except (ET.ParseError, AttributeError, ValueError) as e:
            logger.warning(f"Skipping malformed annotation {xml_path}: {e}")
            continue

        if not lines:
            skipped_no_annotations.append(str(xml_path))
            continue

        shutil.copy(img_path, output_dir / split / "images" / img_path.name)
        (output_dir / split / "labels" / (xml_path.stem + ".txt")).write_text("\n".join(lines))

        total_counts.update(counts)
        split_counts[split] += 1

    report = {
        "source": str(raw_dir),
        "total_xml_files_found": n,
        "images_converted": sum(split_counts.values()),
        "skipped_no_matching_image": len(skipped_no_image),
        "skipped_no_mapped_annotations": len(skipped_no_annotations),
        "split_counts": split_counts,
        "class_distribution": dict(total_counts),
        "total_annotations": sum(total_counts.values()),
    }

    report_path = REPO_ROOT / "outputs" / "reports" / "dataset_preparation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    logger.info(json.dumps(report, indent=2))
    logger.info(f"Report written to: {report_path}")

    # Write dataset.yaml
    dataset_yaml = output_dir.parent / "dataset.yaml"
    dataset_yaml.write_text(
        "path: processed\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "names:\n"
        "  0: pothole\n"
        "  1: crack\n"
    )
    logger.info(f"dataset.yaml written to: {dataset_yaml}")


if __name__ == "__main__":
    main()
