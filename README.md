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

# Todos:
    Need to count the user provided towards the prompt token
    Need to create a method of allowing the user to control the response token numbers
        i.e.: if set at 0.1, only 10% of the max allowed token will be used to create the prompt, giving 90% for summary.
              if set at 0.9, opposite.
    