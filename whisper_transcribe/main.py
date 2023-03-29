import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple

import openai
import yt_dlp

from .helpers import VideoSource, check_if_valid, count_tokens, chunk_generator
from .summarizer.api_call import APICaller


class Transcriber:
    """Transcribe audio from a video file or URL

    Args:
        api_key (str): OpenAI API key
        video_path (str): Path to video file or URL
        prompt (str, optional): Prompt to help WhisperAI transcribe the audio. Defaults to None.
        output (str, optional): Output format. Defaults to "text".
        prompt_ratio (int, optional): How much token should be given to the request prompt. Defaults to 0.4 (40% of max token).
        logging_level (int, optional): Logging level. Defaults to logging.INFO.

    Example:
        >>> from whisper_transcribe import Transcriber

            api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
            video_path = "path/to/a/local/video.mp4"
            prompt = "This prompt will help WhisperAI transcribe the audio."
            output = "srt"
            prompt_ratio = 0.5
            logging_level = logging.INFO

            with Transcriber(
                api_key=api_key, video_path=video_path, output=output, prompt=prompt, prompt_ratio=prompt_ratio, logging_level=logging_level
            ) as t:
                t.transcribe_and_summarize()

    """

    MAX_TOTAL_TOKENS = 4096

    def __init__(
        self,
        api_key: str,
        video_path: str,
        prompt: str = None,
        output: str = "text",
        prompt_ratio: float = 0.4,
        logging_level: int = logging.INFO,
    ):
        openai.api_key = api_key

        logging.basicConfig(level=logging_level)
        logging.basicConfig(format="%(levelname)s-%(message)s")

        self.api_key = api_key
        self.video_path = video_path
        self.output = output
        self.prompt = prompt
        self.prompt_ratio = prompt_ratio
        self.max_prompt_tokens = int(self.MAX_TOTAL_TOKENS * self.prompt_ratio)
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

        call_count = 1
        response = ""

        for chunk in chunk_generator(
            text, self.max_prompt_tokens, count_tokens(self.prompt)
        ):

            prompt = "{}\n\n{}\ntl;dr".format(self.prompt if self.prompt else "", chunk)
            prompt_token_count = count_tokens(prompt)

            data = {
                "api_key": self.api_key,
                "model": "text-davinci-003",
                "temperature": 1.2,
                "max_tokens": self.MAX_TOTAL_TOKENS - prompt_token_count,
                "prompt": prompt,
            }

            api_caller = APICaller(**data)

            logging.info(
                " Summarizing...{}, prompt token count: {}, prompt length: {}".format(
                    call_count, prompt_token_count, len(prompt)
                )
            )
            call_count += 1
            logging.debug(" Prompt: {}".format(prompt))

            result = api_caller.get_text_result()

            response += " " + result.replace(":", "").strip()

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

        logging.info("Transcribing audio from {}".format(self.video_path))

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
