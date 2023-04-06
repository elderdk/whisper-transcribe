import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

import openai
import yt_dlp

from .helpers import VideoSource, count_tokens, chunk_generator


class Transcriber:
    def __init__(self, api_key: str, logging_level: int = logging.INFO) -> None:
        openai.api_key = api_key
        logging.basicConfig(level=logging_level, format="%(levelname)s-%(message)s")

    def _determine_source(self, video_path) -> VideoSource:
        if video_path.startswith("http"):
            self.video_source = VideoSource.URL
        elif Path(video_path).is_file():
            self.video_source = VideoSource.LOCAL
        else:
            self.video_source = VideoSource.UNDETERMINED
            raise ValueError(
                "Unable to determine video source. Please check the video path."
            )

        return self.video_source

    def _get_video_path(self, video_path: str) -> str:
        if self._determine_source(video_path) == VideoSource.URL:
            return self._download_video(video_path)
        else:
            self.video_path = video_path
            return self.video_path

    def _download_video(self, video_path: str, ffmpeg_location: str = None) -> str:

        with NamedTemporaryFile(delete=False) as tmp:
            ydl_opts = {
                "format": "m4a/bestaudio/best",
                "outtmpl": tmp.name + ".m4a",
                "overwrites": True,
                "quiet": True,
                "ffmpeg_location": ffmpeg_location,
                "postprocessors": [
                    {  # Extract audio using ffmpeg
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "m4a",
                    }
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_path])

        self.video_path = tmp.name + ".m4a"

        return self.video_path

    def transcribe(self, video_path, ffmpeg_location=None, **data) -> str:
        """Transcribe a video file or URL

        Example:

            url = "https://www.youtube.com/watch?v=fLeJJPxua3E"

            data = {
                "prompt": "What would you do if you only have 24 hours?",
                "response_format": "text",
                "temperature": 0.5,
            }

            with TranscriberTwo(api_key=api_key) as tt:

                transcription = tt.transcribe(
                    url, **data
                )
                print(transcription)

        Args:
            video_path (str): Path to the video. URL or local path.
            ffmpeg_location (str, optional): ffmpeg required to convert a video
                                             to an audio file for transcription.
                                             Defaults to None. If None, ffmpeg
                                             is assumed to be in the PATH.
            **data: parameters required to call openai.Audio.transcribe
                    Visit https://platform.openai.com/docs/api-reference/audio/create
                    for more information.

                prompt (str, optional): Defaults to None.
                response_format (str, optional): Defaults to "text".
                temperature (float, optional): Defaults to 0.
                language (str, optional): Video's language. Defaults to "en".

        Returns:
            str: transcription
        """

        default_data = {
            "prompt": "What would you do if you only have 24 hours?",
            "response_format": "text",
            "temperature": 0.5,
        }

        default_data.update(**data)

        # Download the video if it's a URL
        if self._determine_source(video_path) == VideoSource.URL:
            video_path = self._download_video(video_path, ffmpeg_location)

        # Call whisperai
        with open(video_path, "rb") as f:
            transcription = openai.Audio.transcribe("whisper-1", f, **default_data)

        return transcription

    def translate(self, video_path, ffmpeg_location=None, **data):
        """Translate a video file or URL into English.

        Example:

            url = "https://www.youtube.com/watch?v=cNplZrRSjeI"

            data = {
                "response_format": "text",
                "temperature": 0.5,
            }

            with TranscriberTwo(api_key=api_key) as tt:

                translation = tt.transcribe(
                    url, **data
                )
                print(translation)

        Args:
            video_path (str): Path to the video. URL or local path.
            ffmpeg_location (str, optional): ffmpeg required to convert a video
                                             to an audio file for transcription.
                                             Defaults to None. If None, ffmpeg
                                             is assumed to be in the PATH.
            **data: parameters required to call openai.Audio.transcribe
                    Visit https://platform.openai.com/docs/api-reference/audio/create
                    for more information.

                prompt (str, optional): Defaults to None.
                response_format (str, optional): Defaults to "text".
                temperature (float, optional): Defaults to 0.
                language (str, optional): Video's language. Defaults to "en".


        Returns:
            str: translation
        """

        default_data = {
            "response_format": "text",
            "temperature": 0.5,
        }

        default_data.update(**data)

        if self._determine_source(video_path) == VideoSource.URL:
            video_path = self._download_video(video_path, ffmpeg_location)

        with open(video_path, "rb") as f:
            translation = openai.Audio.translate("whisper-1", f, **default_data)

        return translation

    def summarize(self, text, prompt_tokens, **data):
        """Summarize a text using OpenAI.

        Exapmle:
            with Transcriber(api_key=api_key) as tt:
                prompt_tokens = 2000
                data = {
                    "max_tokens": 4096,
                    "prompt": "",
                    "model": "text-davinci-003",
                    "temperature": 1.2,
                }

                summary = tt.summarize(text, prompt_tokens, **data)
                print(summary)

        Args:
            text (str): Text to be summarized.
            prompt_tokens (int): How many tokens to use as a prompt. This determines
                                 the length of the text to be fed into OpenAI at one
                                 time. More prompt tokens will lead to shorter summary,
                                 and less prompt token means longer summary, although
                                 not proportionally consistent.

        Returns:
            str: Summary of the text.
        """
        # Update the default data with the user's data
        kwargs = dict()
        kwargs.update(data)

        summary = ""
        max_tokens = kwargs["max_tokens"]
        call_count = 1

        for chunk in chunk_generator(text, prompt_tokens, count_tokens(data["prompt"])):

            prompt = "{}\n\n{}\ntl;dr".format(chunk, data["prompt"])
            prompt_token_count = count_tokens(prompt)

            kwargs["prompt"] = prompt
            kwargs["max_tokens"] = max_tokens - prompt_token_count

            result = openai.Completion.create(**kwargs)

            logging.info(
                " Summarizing...{}, prompt token count: {}, prompt length: {}".format(
                    call_count, prompt_token_count, len(prompt)
                )
            )

            logging.debug(" Prompt: {}".format(prompt))

            call_count += 1

            summary += " " + result["choices"][0]["text"]

        return summary

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, "video_source"):
            if self.video_source == VideoSource.URL and Path(self.video_path).is_file():
                os.remove(self.video_path)
        return False


if __name__ == "__main__":
    pass
