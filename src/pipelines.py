"""
Source pipelines: PDF / YouTube / Web article.

Each function takes a raw input (a file path or URL), produces the
condensed study text and the raw text (used by RAG), caching both
under outputs/<source>/<name>/. Shared by CLI and Streamlit.
"""
from pathlib import Path
from typing import Callable
import hashlib
import time

from src.pdf_loader import load_pdf, extract_text, clean_text, save_text
from src.processor import condense_document, condense_settings_tag
from src.youtube.metadata import get_video_metadata, sanitize_filename
from src.youtube.transcript import get_transcript
from src.youtube.pipeline import process_transcript
from src.web.article import get_article
from src.web.pipeline import process_article

Progress = Callable[[str], None]


def _noop(_message: str) -> None:
    pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condensed_for(text: str, condensed_file: Path, on_progress: Progress) -> str:
    """
    Return the condensed text, from the cache when it was made with the current
    condensing settings, otherwise by condensing again (and caching it).
    """

    settings_file = condensed_file.with_name("condensed_settings.txt")
    tag = condense_settings_tag()

    cached_ok = (
        condensed_file.exists()
        and condensed_file.stat().st_size > 0
        and settings_file.exists()
        and settings_file.read_text(encoding="utf-8").strip() == tag
    )

    if cached_ok:
        on_progress("Loading cached condensed text...")
        condensed_text = condensed_file.read_text(encoding="utf-8")
        on_progress(f"Loaded cached condensed text ({len(condensed_text):,} characters).")
        return condensed_text

    if condensed_file.exists():
        on_progress("Condensing settings changed since the cache was made; condensing again...")

    on_progress("Condensing (this can take a few minutes)...")
    started = time.perf_counter()
    condensed_text = condense_document(text, on_progress=on_progress)
    condensed_file.write_text(condensed_text, encoding="utf-8")
    settings_file.write_text(tag, encoding="utf-8")
    on_progress(f"Condensing completed in {time.perf_counter() - started:.1f}s.")

    return condensed_text


def run_pdf_pipeline(pdf_path: str, on_progress: Progress = _noop):
    """Extract PDF text with a content-validated cache; condense once per PDF."""
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem
    output_dir = Path("outputs") / "pdf" / pdf_name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_text_file = output_dir / "raw_text.txt"
    source_hash_file = output_dir / "source_sha256.txt"
    condensed_file = output_dir / "condensed_text.txt"
    started = time.perf_counter()
    source_hash = _file_sha256(pdf_path)
    on_progress("Checking PDF cache...")

    cached_hash = source_hash_file.read_text(encoding="utf-8").strip() if source_hash_file.exists() else ""
    cache_matches = cached_hash == source_hash and raw_text_file.exists()

    if cache_matches:
        on_progress("Loading cached PDF text...")
        text = raw_text_file.read_text(encoding="utf-8")
    else:
        on_progress("Loading PDF pages...")
        documents = load_pdf(str(pdf_path))
        on_progress(f"Loaded {len(documents)} pages. Extracting text...")
        text = clean_text(extract_text(documents))
        on_progress(f"PDF text extracted in {time.perf_counter() - started:.1f}s ({len(text):,} characters).")
        raw_text_file.write_text(text, encoding="utf-8")
        source_hash_file.write_text(source_hash, encoding="utf-8")
        # A condensed cache from a different PDF with the same filename is stale.
        if condensed_file.exists():
            condensed_file.unlink()
        index_dir = output_dir / "vectorstore"
        if index_dir.exists():
            for stale in (index_dir / "index.faiss", index_dir / "chunks.json"):
                if stale.exists():
                    stale.unlink()

    save_text(text)

    condensed_text = _condensed_for(text, condensed_file, on_progress)

    return condensed_text, text, output_dir


def run_youtube_pipeline(url: str, on_progress: Progress = _noop):
    """Fetch a YouTube transcript, condense it (cached), and return results."""
    on_progress("Fetching video metadata...")
    info = get_video_metadata(url)
    video_title = sanitize_filename(info["title"])
    output_dir = Path("outputs") / "youtube" / video_title
    output_dir.mkdir(parents=True, exist_ok=True)
    condensed_file = output_dir / "condensed_text.txt"
    transcript_file = output_dir / "transcript.txt"

    if transcript_file.exists():
        on_progress("Loading saved transcript...")
        english_text = transcript_file.read_text(encoding="utf-8")
    else:
        on_progress("Downloading transcript...")
        transcript = get_transcript(url)
        on_progress("Processing transcript...")
        english_text = process_transcript(transcript=transcript, output_dir=output_dir)
        transcript_file.write_text(english_text, encoding="utf-8")

    condensed_text = _condensed_for(english_text, condensed_file, on_progress)
    return condensed_text, english_text, output_dir


def run_web_pipeline(url: str, on_progress: Progress = _noop):
    """Download + extract a web article, condense it (cached), and return results."""
    on_progress("Downloading article...")
    title, text = get_article(url)
    output_dir = Path("outputs") / "web" / sanitize_filename(title)
    output_dir.mkdir(parents=True, exist_ok=True)
    condensed_file = output_dir / "condensed_text.txt"
    article_file = output_dir / "article.txt"

    if article_file.exists():
        on_progress("Loading saved article...")
        text = article_file.read_text(encoding="utf-8")
    else:
        text = process_article(text=text, output_dir=output_dir)
        article_file.write_text(text, encoding="utf-8")

    condensed_text = _condensed_for(text, condensed_file, on_progress)
    return condensed_text, text, output_dir


def run_text_pipeline(text: str, title: str = "", on_progress: Progress = _noop):
    """Condense text the user pasted in (cached by its content) and return results."""
    import hashlib

    text = text.strip()
    name = sanitize_filename(title) if title else "Pasted_text"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

    output_dir = Path("outputs") / "text" / f"{name}_{digest}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "article.txt").write_text(text, encoding="utf-8")

    condensed_text = _condensed_for(text, output_dir / "condensed_text.txt", on_progress)
    return condensed_text, text, output_dir
