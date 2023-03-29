import pytest
from tempfile import NamedTemporaryFile
from pathlib import Path
import yt_dlp

from whisper_transcribe.main import Transcriber, VideoSource
from whisper_transcribe.helpers import check_if_valid, chunk_generator, count_tokens


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


def test_namedtemporaryfile_deleted(instance):
    tempfname = NamedTemporaryFile(delete=False).name
    instance.video_path = tempfname
    instance.__exit__("", "", "")

    assert Path(tempfname).is_file() == False


def test_chunk_generator(instance):
    with open("whisper_transcribe/tests/test_data/sample_text.txt", "r") as f:
        text = f.read()

    result_list = [count_tokens(chunk) for chunk in chunk_generator(text, 2000, 10)]
    expected_list = [2671, 2518, 2090]

    assert result_list == expected_list
