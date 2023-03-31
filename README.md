# whisper-transcribe
Python video transcriber that uses OpenAI's WhisperAI.

# Trasncribe a video file:
    from whisper_transcribe import Transcriber

    api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    vid_path = "path/to/a/local/video.mp4" (If you want to transcribe a YouTube video, use its URL.)
    output = "text"
    prompt = "This prompt will help WhisperAI transcribe the audio."

    with Transcriber(api_key=api_key, video_path=vid_path, output=output, prompt=prompt) as tb:
        trasncription = tb.transcribe()

    print(transcription)

# Transcribe AND summarize a file:
    with Transcriber(api_key=api_key, video_path=vid_path, output=output, prompt=prompt) as tb:
            trasncription, summary = tb.transcribe()

    print(transcription)
    print(summary)

# Create a subtitle of a video:
    output = "srt"