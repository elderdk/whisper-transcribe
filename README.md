# whisper-transcribe
Python video transcriber that uses OpenAI's WhisperAI.

# Example:
    from whisper_transcribe import Transcriber

    api_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    vid = "path/to/a/local/video.mp4"
    prompt = "This prompt will help WhisperAI transcribe the audio."
    outpath = "path/to/output/file.txt"
    output = "srt"

    with Transcriber(
        api_key=api_key, video_path=yt_vid, output=output, prompt=prompt
    ) as tber:
        tber.transcribe()
