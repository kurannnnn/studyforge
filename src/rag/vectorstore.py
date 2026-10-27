import os

# See src/model.py for why this must be set before faiss (or torch)
# imports on Windows. Idempotent — harmless if already set.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import hashlib
from pathlib import Path

import faiss
import numpy as np

from src.rag.chunking import chunk_for_retrieval
from src.rag.embeddings import load_embedder


def _index_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "vectorstore"


def _paths(output_dir: Path):
    index_dir = _index_dir(output_dir)
    return index_dir / "index.faiss", index_dir / "chunks.json", index_dir / "source_sha256.txt"


def _embed(embedder, texts: list[str]) -> np.ndarray:
    """Encode texts and L2-normalize for cosine-similarity search."""
    vectors = embedder.encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")
    faiss.normalize_L2(vectors)
    return vectors


def build_index(raw_text: str, output_dir: Path):
    """Chunk, embed and persist the document's FAISS index."""
    chunks = chunk_for_retrieval(raw_text)
    if not chunks:
        raise ValueError("No text available to index for RAG.")

    embedder = load_embedder()
    vectors = _embed(embedder, chunks)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_dir = _index_dir(output_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path, chunks_path, hash_path = _paths(output_dir)
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
    source_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    hash_path.write_text(source_hash, encoding="utf-8")
    return index, chunks


def load_index(output_dir: Path, raw_text: str | None = None):
    """Load a cached index only if its chunk count and source fingerprint match."""
    index_path, chunks_path, hash_path = _paths(output_dir)
    if not (index_path.exists() and chunks_path.exists() and hash_path.exists()):
        return None
    if raw_text is not None:
        expected = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        if hash_path.read_text(encoding="utf-8").strip() != expected:
            return None
    try:
        index = faiss.read_index(str(index_path))
        with open(chunks_path, "r", encoding="utf-8") as file:
            chunks = json.load(file)
    except (OSError, ValueError, RuntimeError):
        return None
    if index.ntotal != len(chunks):
        return None
    return index, chunks


def get_or_build_index(raw_text: str, output_dir: Path):
    """Reuse a cached index only for exactly the same raw document text."""
    cached = load_index(output_dir, raw_text)
    if cached is not None:
        return cached
    return build_index(raw_text, output_dir)


def _needs_following_context(chunk: str) -> bool:
    """Detect a chunk that appears to end mid-sentence/list/definition."""
    text = chunk.rstrip()
    if not text:
        return False
    # A chunk ending in ordinary sentence punctuation is usually complete.
    # Closing quotes/brackets after punctuation are allowed.
    trimmed = text.rstrip('"\'”’)]}')
    if trimmed.endswith((".", "?", "!")):
        return False
    # Common incomplete endings from fixed-size chunk boundaries, including
    # list introductions such as "One is the total computational...".
    return len(text) >= 80


def _needs_previous_context(chunk: str) -> bool:
    """Detect a chunk that appears to begin as a continuation of prior text."""
    text = chunk.lstrip()
    if not text:
        return False
    first = next((c for c in text if c.isalpha()), "")
    return bool(first and first.islower())


def search(index, chunks: list[str], query: str, top_k: int) -> list[str]:
    """Retrieve top semantic matches and expand only for likely split passages.

    The normal result is exactly the top_k matches. If a selected chunk appears
    truncated at a sentence/list boundary, include the adjacent chunk needed to
    complete that context. This avoids unconditional expansion to many chunks.
    """
    if not chunks or top_k <= 0:
        return []

    embedder = load_embedder()
    query_vector = _embed(embedder, [query])
    k = min(top_k, len(chunks))
    scores, indices = index.search(query_vector, k)

    hits = []
    for score, idx in zip(scores[0], indices[0]):
        idx = int(idx)
        if 0 <= idx < len(chunks):
            hits.append((float(score), idx))

    if not hits:
        return []

    selected = {idx for _, idx in hits}
    expansions = []
    # Prefer the highest-scoring hit when multiple hits need boundary context.
    for _, idx in sorted(hits, reverse=True):
        chunk = chunks[idx]
        candidate = None
        if _needs_following_context(chunk) and idx + 1 < len(chunks):
            candidate = idx + 1
        elif _needs_previous_context(chunk) and idx > 0:
            candidate = idx - 1
        if candidate is not None and candidate not in selected and candidate not in expansions:
            expansions.append(candidate)
        if len(expansions) >= 2:
            break

    selected.update(expansions)
    return [chunks[i] for i in sorted(selected)]
