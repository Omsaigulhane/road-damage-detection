import pytest

from src.video.video_processor import VideoProcessor, VideoProcessingError


class FakePipeline:
    """Stand-in for RoadDamagePipeline so video tests don't require a real model."""
    def process_image(self, image):
        return [], image, 5.0


@pytest.fixture()
def processor():
    config = {"video": {"frame_sample_rate": 1, "event_aggregation": {"enabled": False}}}
    return VideoProcessor(FakePipeline(), config=config)


def test_invalid_video_path_raises(processor):
    with pytest.raises(VideoProcessingError):
        processor.process("/nonexistent/path/video.mp4")


def test_unsupported_format_raises(processor, tmp_path):
    bad_file = tmp_path / "clip.txt"
    bad_file.write_text("not a video")
    with pytest.raises(VideoProcessingError):
        processor.process(bad_file)


def test_empty_or_corrupted_video_raises(processor, tmp_path):
    fake_video = tmp_path / "empty.mp4"
    fake_video.write_bytes(b"not really a video file")
    with pytest.raises(VideoProcessingError):
        processor.process(fake_video)
