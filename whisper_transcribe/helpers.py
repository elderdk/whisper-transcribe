from enum import Enum
import tiktoken


class VideoSource(Enum):
    UNDETERMINED = 1
    URL = 2
    LOCAL = 3


def check_if_valid(instance):
    if not instance.output in ["text", "srt"]:
        raise ValueError(
            "Output must be either 'text' or 'srt', not '{}'".format(instance.output)
        )

    if instance.video_source == VideoSource.UNDETERMINED:
        raise ValueError(
            "Unable to determine video source. Please check the video path."
        )


def count_tokens(prompt: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("text-davinci-003")
    tokens = encoding.encode(prompt)
    return len(tokens)
