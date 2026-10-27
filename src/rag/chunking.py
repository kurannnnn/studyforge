from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP

_rag_splitter = None


def chunk_for_retrieval(text: str) -> list[str]:
    """
    Split raw (uncondensed) text into small, overlapping chunks
    suited for embedding + retrieval.

    Deliberately separate from src.text_splitter.split_text, which
    is tuned for feeding the LLM's context window during condensing.
    Retrieval wants smaller, more granular chunks so a question can
    be matched to a specific passage rather than a whole section.
    """

    global _rag_splitter

    if _rag_splitter is None:

        _rag_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP,
        )

    chunks = _rag_splitter.split_text(text)

    # Drop near-empty fragments (e.g. trailing whitespace-only chunks)
    return [chunk.strip() for chunk in chunks if chunk.strip()]
