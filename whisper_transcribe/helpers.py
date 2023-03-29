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


def chunk_generator(
    text: str, max_prompt_tokens: int, prompt_tokens: int, jump_n: int = 100
) -> str:
    """Generate chunk for OpenAI Completion API

    Args:
        text (str): Text to be summarized
        jump_n (int): Number of words to chunk together before counting the tokens

    Returns:
        chunk (str): chunk for OpenAI Completion API
    """

    word_list = text.split()

    place_holder = ""
    for i in range(0, len(word_list), jump_n):
        place_holder += " ".join(word_list[i : i * 2 if i > 0 else jump_n])
        if count_tokens(place_holder) + prompt_tokens > max_prompt_tokens:
            chunk = place_holder
            place_holder = ""
            yield chunk
