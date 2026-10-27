from pathlib import Path


def process_article(
    text: str,
    output_dir: Path,
) -> str:
    """
    Save the extracted article text and return it.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    article_file = (
        output_dir / "article.txt"
    )

    with open(
        article_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(text)

    return text