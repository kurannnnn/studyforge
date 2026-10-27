import re

from yt_dlp import YoutubeDL


def get_video_metadata(url: str) -> dict:
    """
    Fetch YouTube video metadata without downloading.
    """

    options = {
        "quiet": True,
        "skip_download": True,
        "no_warnings": True,
    }

    with YoutubeDL(options) as ydl:

        info = ydl.extract_info(
            url,
            download=False,
        )

    return {
        "title": info.get("title", "Unknown Video"),
        "channel": info.get("uploader", "Unknown Channel"),
        "duration": info.get("duration", 0),
        "video_id": info.get("id"),
    }


def sanitize_filename(name: str) -> str:
    """
    Convert video title into a safe folder name.
    """

    name = re.sub(
        r'[<>:"/\\|?*]',
        "",
        name,
    )

    name = re.sub(
        r"\s+",
        "_",
        name,
    )

    return name[:80].strip("_")