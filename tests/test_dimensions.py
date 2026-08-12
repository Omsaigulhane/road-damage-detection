from src.dimensions.estimator import DimensionEstimator


def test_uncalibrated_returns_no_physical_estimate():
    config = {"dimension_estimation": {"enabled": True, "reference_width_m": None, "camera_height_m": None, "focal_length_px": None}}
    estimator = DimensionEstimator(config=config)
    result = estimator.estimate(100, 50)
    assert result.calibration_available is False
    assert result.estimated_width_m is None
    assert result.estimated_height_m is None
    assert "unavailable" in result.note.lower()


def test_calibrated_returns_physical_estimate():
    config = {"dimension_estimation": {"enabled": True, "camera_height_m": 1.2, "focal_length_px": 800}}
    estimator = DimensionEstimator(config=config)
    result = estimator.estimate(100, 50)
    assert result.calibration_available is True
    assert result.estimated_width_m == round(100 * (1.2 / 800), 4)


def test_disabled_estimator():
    config = {"dimension_estimation": {"enabled": False}}
    estimator = DimensionEstimator(config=config)
    result = estimator.estimate(100, 50)
    assert result.estimated_width_m is None
    assert "disabled" in result.note.lower()
