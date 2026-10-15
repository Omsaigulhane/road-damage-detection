import pytest

from src.severity.severity_engine import SeverityEngine, SeverityConfigError

TEST_CONFIG = {
    "severity": {
        "normalization": {
            "size_reference_relative_area": 0.06,
            "dimension_reference_m": {"pothole": 0.6, "crack": 1.5},
        },
        "weights": {
            "pothole": {"size": 0.40, "dimension": 0.40, "confidence": 0.20},
            "crack": {"size": 0.45, "dimension": 0.35, "confidence": 0.20},
        },
        "thresholds": {"low": 25, "moderate": 50, "high": 75},
    }
}


def test_score_within_bounds():
    engine = SeverityEngine(config=TEST_CONFIG)
    result = engine.compute("pothole", relative_area=0.5, confidence=1.0, estimated_size_m=5.0)
    assert 0 <= result.severity_score <= 100


def test_zero_input_gives_zero_score():
    engine = SeverityEngine(config=TEST_CONFIG)
    result = engine.compute("pothole", relative_area=0.0, confidence=0.0, estimated_size_m=0.0)
    assert result.severity_score == 0
    assert result.severity_category == "Low"


def test_missing_dimension_falls_back_to_size_score():
    engine = SeverityEngine(config=TEST_CONFIG)
    result = engine.compute("crack", relative_area=0.03, confidence=0.8, estimated_size_m=None)
    assert result.dimension_score_is_fallback is True
    assert result.dimension_score == result.size_score


def test_invalid_confidence_raises():
    engine = SeverityEngine(config=TEST_CONFIG)
    with pytest.raises(ValueError):
        engine.compute("pothole", relative_area=0.1, confidence=1.5)


def test_invalid_damage_type_raises():
    engine = SeverityEngine(config=TEST_CONFIG)
    with pytest.raises(ValueError):
        engine.compute("sinkhole", relative_area=0.1, confidence=0.5)


def test_negative_relative_area_raises():
    engine = SeverityEngine(config=TEST_CONFIG)
    with pytest.raises(ValueError):
        engine.compute("pothole", relative_area=-0.1, confidence=0.5)


def test_weights_must_sum_to_one():
    bad_config = {
        "severity": {
            "normalization": {"size_reference_relative_area": 0.06, "dimension_reference_m": {"pothole": 0.6, "crack": 1.5}},
            "weights": {
                "pothole": {"size": 0.5, "dimension": 0.5, "confidence": 0.5},  # sums to 1.5
                "crack": {"size": 0.45, "dimension": 0.35, "confidence": 0.20},
            },
            "thresholds": {"low": 25, "moderate": 50, "high": 75},
        }
    }
    with pytest.raises(SeverityConfigError):
        SeverityEngine(config=bad_config)


def test_category_thresholds():
    engine = SeverityEngine(config=TEST_CONFIG)
    assert engine._category(10) == "Low"
    assert engine._category(40) == "Moderate"
    assert engine._category(60) == "High"
    assert engine._category(90) == "Critical"
def test_vibration_uncalibrated_falls_back_to_zero():
    engine = SeverityEngine(config=TEST_CONFIG)
    result = engine.compute("pothole", relative_area=0.03, confidence=0.8, vibration_magnitude_g=2.5)
    assert result.vibration_score == 0.0
    assert result.vibration_score_is_fallback is True


def test_vibration_calibrated_computes_real_score():
    config_with_vibration = {
        "severity": {
            "normalization": {
                "size_reference_relative_area": 0.06,
                "dimension_reference_m": {"pothole": 0.6, "crack": 1.5},
                "vibration_reference_g": {"baseline_g": 1.0, "max_g": 3.0},
            },
            "weights": {
                "pothole": {"size": 0.30, "dimension": 0.30, "confidence": 0.20, "vibration": 0.20},
                "crack": {"size": 0.45, "dimension": 0.35, "confidence": 0.20, "vibration": 0.0},
            },
            "thresholds": {"low": 25, "moderate": 50, "high": 75},
        }
    }
    engine = SeverityEngine(config=config_with_vibration)
    result = engine.compute("pothole", relative_area=0.03, confidence=0.8, vibration_magnitude_g=2.0)
    # (2.0 - 1.0) / (3.0 - 1.0) = 0.5
    assert result.vibration_score == 0.5
    assert result.vibration_score_is_fallback is False