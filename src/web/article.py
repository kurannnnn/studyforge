import trafilatura


def get_article(url: str):
    """
    Download and extract the main article text.
    """

    downloaded = trafilatura.fetch_url(url)

    if downloaded is None:
        raise ValueError(
            "Failed to download article."
        )

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
    )

    if not text:
        raise ValueError(
            "Failed to extract article."
        )

    metadata = trafilatura.extract_metadata(
        downloaded
    )

    title = (
        metadata.title
        if metadata and metadata.title
        else "Web_Article"
    )

    return title, text