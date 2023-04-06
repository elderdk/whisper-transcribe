from whisper_transcribe.helpers import chunk_generator, count_tokens


PROMPT = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."


def test_count_tokens():
    model = "text-davinci-003"
    assert count_tokens(PROMPT, model) == 153


def test_chunk_generator():
    for chunk in chunk_generator(PROMPT, 500, 10):
        assert type(chunk) == str
        assert len(chunk) < 500
