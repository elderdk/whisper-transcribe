import pytest
from tempfile import NamedTemporaryFile
from pathlib import Path
import yt_dlp

from ..whisper_transcribe.main import Transcriber, VideoSource
from ..whisper_transcribe.helpers import check_if_valid


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


def test_raise_error_on_undetermined_source(instance):
    instance.video_source = VideoSource.UNDETERMINED

    with pytest.raises(ValueError):
        check_if_valid(instance)


def test_raise_error_on_invalid_output(instance):
    instance.output = "invalid_output"

    with pytest.raises(ValueError):
        check_if_valid(instance)


# def test_namedtemporaryfile_deleted(mocker, instance):
#     mocker.patch("yt_dlp.YoutubeDL.download", return_value="")
#     fname = instance._download_video()
#     instance.__exit__()

#     assert Path(fname).is_file() == False
