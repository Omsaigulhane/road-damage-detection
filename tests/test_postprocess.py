import pytest

from src.detection.detector import Detection
from src.detection.postprocess import compute_box_metrics


def make_detection(x1, y1, x2, y2, conf=0.9, class_name="pothole"):
    return Detection(class_id=0 if class_name == "pothole" else 1, class_name=class_name,
                      confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def test_box_metrics_basic():
    det = make_detection(10, 10, 110, 60)  # 100x50
    metrics = compute_box_metrics(det, image_width=1000, image_height=1000)
    assert metrics.width_px == 100
    assert metrics.height_px == 50
    assert metrics.area_px == 5000
    assert metrics.relative_area == pytest.approx(5000 / 1_000_000)


def test_box_metrics_invalid_image_size_raises():
    det = make_detection(0, 0, 10, 10)
    with pytest.raises(ValueError):
        compute_box_metrics(det, image_width=0, image_height=100)


def test_negative_coords_clamped_to_zero_size():
    det = make_detection(50, 50, 10, 10)  # x2<x1, y2<y1 (malformed box)
    metrics = compute_box_metrics(det, image_width=100, image_height=100)
    assert metrics.width_px == 0
    assert metrics.height_px == 0
    assert metrics.area_px == 0
