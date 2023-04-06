import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import openai
import pytest
import yt_dlp

from whisper_transcribe.helpers import chunk_generator, count_tokens
from whisper_transcribe.main import Transcriber, VideoSource


@pytest.fixture
def instance():
    return Transcriber(
        api_key="2jk2j34kj23n42n3kj4n23k4",
    )


def test_determine_source(instance):
    result = instance._determine_source("http://www.youtube.com/watch?v=9bZkp7q19f0")
    assert result == VideoSource.URL

    with NamedTemporaryFile() as tmp:
        result = instance._determine_source(tmp.name)
        assert result == VideoSource.LOCAL

    with pytest.raises(ValueError):
        instance._determine_source("asasdasdadz")


def test_get_video_path_url(instance, monkeypatch):
    def mock_download(self, video_path):
        return None

    video_path = "http://www.youtube.com/watch?v=9bZkp7q19f0"
    monkeypatch.setattr(yt_dlp.YoutubeDL, "download", mock_download)

    # since no download is made, the file exists without extension. Must get [:-4]
    result = instance._get_video_path(video_path)[:-4]
    assert Path(result).is_file()
    os.remove(result)


def test_get_video_path_local(instance):
    with NamedTemporaryFile(suffix=".mp4") as tmp:
        result = instance._get_video_path(tmp.name)
        assert result == tmp.name


def test_transcribe(instance, monkeypatch):

    with NamedTemporaryFile(delete=False) as tmp:

        def temp_file(self, video_path, ffmpeg_location=None):
            return tmp.name

        def mock_transcribe(self, video_path):
            return "transcribed text"

        video_path = "http://www.youtube.com/watch?v=9bZkp7q19f0"
        monkeypatch.setattr(Transcriber, "_download_video", temp_file)
        monkeypatch.setattr(openai.Audio, "transcribe", mock_transcribe)

        result = instance.transcribe(video_path)
        assert result is not None
        os.remove(tmp.name)


def test_translate(instance, monkeypatch):

    with NamedTemporaryFile(delete=False) as tmp:

        def temp_file(self, video_path, ffmpeg_location=None):
            return tmp.name

        def mock_translate(self, video_path):
            return "translated text"

        video_path = "http://www.youtube.com/watch?v=9bZkp7q19f0"
        monkeypatch.setattr(Transcriber, "_download_video", temp_file)
        monkeypatch.setattr(openai.Audio, "translate", mock_translate)

        result = instance.translate(video_path)
        assert result is not None
        os.remove(tmp.name)


def test_summarize(instance, monkeypatch):
    def mock_create(**kwargs):
        return {"choices": [{"text": "summary"}]}

    monkeypatch.setattr(openai.Completion, "create", mock_create)

    data = {
        "max_tokens": 4096,
        "prompt": "",
        "model": "text-davinci-003",
        "temperature": 1.2,
    }

    summary = instance.summarize("text" * 1000, 500, **data)

    assert summary


def test_transcriber_enter():
    with Transcriber(api_key="2jk2j34kj23n42n3kj4n23k4") as instance:
        assert instance is not None


def test_transcriber_deletes_tempfile_on_exit():
    tf = NamedTemporaryFile(delete=False)
    with Transcriber(api_key="2jk2j34kj23n42n3kj4n23k4") as instance:
        instance.video_source = VideoSource.URL
        instance.video_path = tf.name

    assert Path(tf.name).is_file() == False
