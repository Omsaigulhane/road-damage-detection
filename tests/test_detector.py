import numpy as np
import pytest

from src.detection.detector import RoadDamageDetector, DetectorError, resolve_device, Detection


def test_resolve_device_invalid_raises():
    with pytest.raises(DetectorError):
        resolve_device("tpu")


def test_resolve_device_cpu_always_ok():
    assert resolve_device("cpu") == "cpu"


class _FakeBox:
    def __init__(self, cls, conf, xyxy):
        import torch
        self.cls = torch.tensor([cls])
        self.conf = torch.tensor([conf])
        self.xyxy = torch.tensor([xyxy])


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLOModel:
    """Stands in for ultralytics.YOLO so tests don't need real weights/network."""
    def __init__(self, detections_to_return=None):
        self._detections = detections_to_return or []

    def predict(self, source, imgsz, conf, iou, device, verbose):
        return [_FakeResult(self._detections)]


def _make_detector_with_fake_model(fake_model) -> RoadDamageDetector:
    detector = RoadDamageDetector.__new__(RoadDamageDetector)
    detector.config = {}
    detector.confidence_threshold = 0.35
    detector.iou_threshold = 0.45
    detector.image_size = 640
    detector.device = "cpu"
    detector.weights_path = "fake.pt"
    detector.model = fake_model
    return detector


def test_detect_on_empty_image_raises():
    detector = _make_detector_with_fake_model(_FakeYOLOModel())
    with pytest.raises(DetectorError):
        detector.detect(np.array([]))


def test_detect_with_no_detections_returns_empty_list():
    detector = _make_detector_with_fake_model(_FakeYOLOModel(detections_to_return=[]))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert result.detections == []
    assert result.image_width == 640
    assert result.image_height == 480


def test_detect_parses_boxes_correctly():
    fake_box = _FakeBox(cls=0.0, conf=0.87, xyxy=[10.0, 20.0, 110.0, 70.0])
    detector = _make_detector_with_fake_model(_FakeYOLOModel(detections_to_return=[fake_box]))
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert len(result.detections) == 1
    det = result.detections[0]
    assert det.class_name == "pothole"
    assert det.confidence == pytest.approx(0.87, abs=1e-2)
    assert det.width_px == 100
    assert det.height_px == 50
