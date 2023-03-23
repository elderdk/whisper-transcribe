import pytest
from tempfile import NamedTemporaryFile

from whisper_transcribe.main import Transcriber, VideoSource


@pytest.fixture
def instance():
    return Transcriber(
        api_key="api_key",
        video_path="https://www.youtube.com/watch?v=q1HZj40ZQrM",
        output="srt",
    )


def test_init(instance):
    assert instance.video_path == "https://www.youtube.com/watch?v=q1HZj40ZQrM"
    assert instance.output == "srt"


def test_determine_source(instance):
    instance.video_path = "https://www.youtube.com/watch?v=q1HZj40ZQrM"
    assert instance._determine_source() == VideoSource.URL

    with NamedTemporaryFile() as f:
        instance.video_path = f.name
        assert instance._determine_source() == VideoSource.LOCAL

    instance.video_path = "non-existence_video.mp4"
    assert instance._determine_source() == VideoSource.UNDETERMINED
