import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

import openai
import yt_dlp

from .helpers import VideoSource, check_if_valid, count_tokens
from .summarizer.api_call import APICaller


class Transcriber:
    """Transcribe audio from a video file or URL

    Args:
        api_key (str): OpenAI API key
        video_path (str): Path to video file or URL
        prompt (str, optional): Prompt to help WhisperAI transcribe the audio. Defaults to None.
        output (str, optional): Output format. Defaults to "text".

    Example:
        >>> from whisper_transcribe import Transcriber

            api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
            video_path = "path/to/a/local/video.mp4"
            prompt = "This prompt will help WhisperAI transcribe the audio."
            output = "srt"

            with Transcriber(
                api_key=api_key, video_path=video_path, output=output, prompt=prompt
            ) as t:
                t.transcribe_and_summarize()

    """

    MAX_TOTAL_TOKENS = 4096
    MAX_PROMPT_TOKENS = 2900

    def __init__(
        self,
        api_key: str,
        video_path: str,
        prompt: str = None,
        output: str = "text",
    ):
        openai.api_key = api_key
        self.api_key = api_key
        self.video_path = video_path
        self.output = output
        self.prompt = prompt
        self.video_source = self._determine_source()
        check_if_valid(self)

    def _determine_source(self) -> VideoSource:
        if self.video_path.startswith("http"):
            return VideoSource.URL
        if Path(self.video_path).is_file():
            return VideoSource.LOCAL
        else:
            return VideoSource.UNDETERMINED

    def _download_video(self):

        with NamedTemporaryFile(delete=False) as tmp:
            ydl_opts = {
                "format": "m4a/bestaudio/best",
                "outtmpl": tmp.name + ".m4a",
                "overwrites": True,
                "quiet": True,
                "postprocessors": [
                    {  # Extract audio using ffmpeg
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "m4a",
                    }
                ],
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.video_path])

        return tmp.name + ".m4a"

    def _summarize(self, text) -> str:
        """Summarize the given text using OpenAI Completipn API

        Split the text into a list of words, then loop through the list
        Add words to the prompt until token hits above 2900
        Count the number of tokens to be in the response
        Make the call.

        Args:
            text (str): Text to be summarized

        Returns:
            response (str): Summarized text
        """
        response = ""
        text_list = text.split()

        while len(text_list) > 0:

            text_place_holder = ""
            token_nums = 0

            # accumulate 2900 tokens worth of texts
            # delete the accumulated texts from the list and make a call
            while (
                token_nums < self.MAX_PROMPT_TOKENS and len(text_list) > 0
            ):  # token_nums limit needs to be optimized for better results
                text_place_holder += " ".join(text_list[:100]) + " "
                del text_list[:100]
                token_nums = count_tokens(text_place_holder)

            prompt = "{}\n\ntl;dr".format(text_place_holder)

            data = {
                "api_key": self.api_key,
                "model": "text-davinci-003",
                "temperature": 1.2,
                "max_tokens": self.MAX_TOTAL_TOKENS - count_tokens(prompt),
                "prompt": prompt,
            }

            api_caller = APICaller(**data)
            result = api_caller.get_text_result()

            response += result

        return response

    def transcribe(self) -> str:
        """Transcribe audio from a video file or URL

        Check the video_source. If URL, initiate download and save into a
        Named temporary file.

        Calls openai.Audio.transcribe.

        Returns:
            transcript (str): Transcribed text

        """

        if self.video_source == VideoSource.URL:
            self.video_path = self._download_video()

        with open(self.video_path, "rb") as f:
            transcript = openai.Audio.transcribe(
                "whisper-1", f, response_format=self.output, prompt=self.prompt
            )

        return transcript

    def transcribe_and_summarize(self) -> Tuple[str, str]:
        """Transcribe and summarize
        Force change the output if it's not text since summarizing from srt
        doesn't make sense.


        Returns:
            Tuple[str, str]: transcribed text, summarized text
        """
        if self.output != "text":
            print(
                "To summarize, output must be text. Force changing output from {} to text.".format(
                    self.output
                )
            )
            self.output = "text"

        transcript = self.transcribe()
        summary = self._summarize(transcript)
        return (transcript, summary)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.video_source == VideoSource.URL:
            os.remove(self.video_path)
        return False


if __name__ == "__main__":
    pass
