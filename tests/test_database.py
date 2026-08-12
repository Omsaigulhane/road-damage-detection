import pytest

from src.database.repository import DetectionRepository, RepositoryError
from src.database.models import DetectionRecord


@pytest.fixture()
def repo(tmp_path):
    config = {"database": {"path": str(tmp_path / "test.db")}}
    return DetectionRepository(config=config)


def make_record(**overrides) -> DetectionRecord:
    base = dict(
        id=None, timestamp="2026-08-08T00:00:00", source_file="test.jpg", source_type="image",
        frame_number=None, video_timestamp=None, damage_type="pothole", confidence=0.9,
        x1=0, y1=0, x2=100, y2=50, width_px=100, height_px=50, area_px=5000,
        relative_area=0.01, estimated_width_m=None, estimated_height_m=None,
        severity_score=42.0, severity_category="Moderate",
    )
    base.update(overrides)
    return DetectionRecord(**base)


def test_insert_and_retrieve(repo):
    record_id = repo.insert(make_record())
    assert record_id > 0
    rows = repo.get_all()
    assert len(rows) == 1
    assert rows[0]["damage_type"] == "pothole"


def test_summary_stats(repo):
    repo.insert(make_record(damage_type="pothole", severity_score=80, severity_category="Critical"))
    repo.insert(make_record(damage_type="crack", severity_score=10, severity_category="Low"))
    stats = repo.summary_stats()
    assert stats["total_detections"] == 2
    assert stats["pothole_count"] == 1
    assert stats["crack_count"] == 1
    assert stats["high_critical_count"] == 1


def test_invalid_source_type_rejected(repo):
    with pytest.raises(RepositoryError):
        repo.insert(make_record(source_type="audio"))


def test_invalid_damage_type_rejected(repo):
    with pytest.raises(RepositoryError):
        repo.insert(make_record(damage_type="sinkhole"))


def test_invalid_confidence_rejected(repo):
    with pytest.raises(RepositoryError):
        repo.insert(make_record(confidence=1.5))


def test_invalid_severity_score_rejected(repo):
    with pytest.raises(RepositoryError):
        repo.insert(make_record(severity_score=150))


def test_get_by_source_file(repo):
    repo.insert(make_record(source_file="a.jpg"))
    repo.insert(make_record(source_file="b.jpg"))
    rows = repo.get_by_source_file("a.jpg")
    assert len(rows) == 1
    assert rows[0]["source_file"] == "a.jpg"
