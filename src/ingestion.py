from youtube_transcript_api import YouTubeTranscriptApi

def get_transcript(video_id : str) -> str :

    api = YouTubeTranscriptApi()

    transcript = api.fetch(
        video_id=video_id,
        languages=["en"]
    )

    transcript_data = transcript.to_raw_data()

    transcript_text = " ".join(item["text"] for item in transcript_data)

    return transcript_text