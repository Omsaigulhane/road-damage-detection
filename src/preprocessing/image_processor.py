"""Image loading/validation. YOLO/Ultralytics handles resizing & normalization
internally, so this module focuses on I/O safety (Section 35)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageProcessingError(Exception):
    pass


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise ImageProcessingError(f"Image file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageProcessingError(
            f"Unsupported image format '{path.suffix}'. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    image = cv2.imread(str(path))
    if image is None:
        raise ImageProcessingError(
            f"Failed to decode image '{path}'. The file may be corrupted or not a valid image."
        )
    return image


def load_image_bytes(data: bytes, filename: str = "upload") -> np.ndarray:
    """Decode an in-memory image (e.g. from a Streamlit file upload)."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageProcessingError(
            f"Unsupported image format '{suffix}'. Supported: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
        )
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageProcessingError(f"Failed to decode uploaded image '{filename}'. It may be corrupted.")
    return image
