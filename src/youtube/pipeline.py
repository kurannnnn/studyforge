from pathlib import Path


def process_transcript(
    transcript: str,
    output_dir: Path,
) -> str:
    """
    Save the transcript and return it.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transcript_file = (
        output_dir / "transcript.txt"
    )

    with open(
        transcript_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(transcript)

    return transcript