from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url: str) -> str:

    url = url.strip().rstrip("/\\")

    parsed = urlparse(url)

    if parsed.hostname in (
        "www.youtube.com",
        "youtube.com",
    ):
        return parse_qs(parsed.query)["v"][0].strip()

    if parsed.hostname == "youtu.be":
        return parsed.path.strip("/\\")

    raise ValueError("Invalid YouTube URL.")


def get_transcript(url: str):
    # Returns the transcript as English text.
    # Raises ValueError if the video has no English captions and none of
    # its captions can be translated to English.

    url = url.strip().rstrip("/\\")
    video_id = extract_video_id(url).strip().rstrip("/\\")

    print(f"URL      : {url}")
    print(f"Video ID : {video_id}")

    api = YouTubeTranscriptApi()

    available = list(api.list(video_id))

    if not available:
        raise ValueError("This video has no captions, so it can't be processed.")

    # 1. Any English track (en, en-US, en-GB, ...). Human-made before auto-generated.
    english = sorted(
        (t for t in available if t.language_code.lower().startswith("en")),
        key=lambda t: t.is_generated,
    )

    translated = False

    if english:
        transcript = english[0]
        print(f"Using English captions ({transcript.language_code})")

    else:
        # 2. No English track: ask YouTube to translate one.
        translatable = sorted(
            (t for t in available if t.is_translatable),
            key=lambda t: t.is_generated,
        )

        if not translatable:
            raise ValueError(
                "This video has no English captions and none of its captions "
                "can be translated to English, so it can't be processed."
            )

        source = translatable[0]

        try:
            transcript = source.translate("en")
        except Exception as error:
            raise ValueError(
                f"This video has no English captions and its {source.language_code} "
                f"captions couldn't be translated to English ({error})."
            ) from error

        translated = True
        print(f"Using {source.language_code} captions, machine-translated to English")

    try:
        fetched = transcript.fetch()
    except Exception as error:
        if translated:
            raise ValueError(
                f"The English translation of this video's captions couldn't be downloaded ({error})."
            ) from error
        raise

    text = " ".join(snippet.text for snippet in fetched)

    if not text.strip():
        raise ValueError("This video's captions are empty, so it can't be processed.")

    return text
