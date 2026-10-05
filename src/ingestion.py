import os
import requests

from dotenv import load_dotenv

load_dotenv()


def get_transcript(video_id: str) -> str:

    api_key = os.getenv("FREETRANSCRIPT_API_KEY")

    if not api_key:
        raise RuntimeError(
            "FREETRANSCRIPT_API_KEY is not configured."
        )

    url = "https://api.freetranscriptapi.com/v1/transcript"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    params = {
        "video_url": video_id,
        "lang": "en"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:

        raise RuntimeError(
            f"Transcript API request failed: {e}"
        )

    # API-level error
    if "error" in data:

        error_message = data["error"].get(
            "message",
            "Unknown transcript API error."
        )

        raise RuntimeError(
            f"Could not retrieve transcript: {error_message}"
        )

    transcript_data = data.get("transcript")

    if not transcript_data:

        raise RuntimeError(
            "No transcript was available for this video."
        )

    transcript_text = " ".join(
        item["text"]
        for item in transcript_data
    )

    return transcript_text