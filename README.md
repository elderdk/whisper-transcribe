# whisper-transcribe
Python video transcriber that uses OpenAI's WhisperAI.

# Trasncribe a video file:
    from whisper_transcribe import Transcriber

    api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

    with Transcriber(api_key=api_key) as tb:
        trasncription = tb.transcribe("path/to/a/local/video.mp4")

    print(transcription)

# Create a subtitle of a video:
    with Transcriber(api_key=api_key) as tb:
        tb.transcribe("path/to/a/local/video.mp4", output="srt")

# Transcribe a YouTube video:
    with Transcriber(api_key=api_key) as tb:
        tb.translate("https://youtube.com/myVideo")

# Translate a video file:
    with Transcriber(api_key=api_key) as tb:
        translation = tb.translate("path/to/a/local/video.mp4")

    print(translation)

# Summarize a text:
    with Transcriber(api_key=api_key) as tb:
        transcription = tb.transcribe("path/to/a/local/video.mp4", output="text")
        summary = tb.summarize(transcription)