from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME

_embedder = None


def load_embedder() -> SentenceTransformer:
    """
    Load (once) and return the sentence embedding model used for RAG.

    The embedding model is small and CPU-friendly, and it runs inside the
    app, separately from the language model (which runs in llama-server).
    """

    global _embedder

    if _embedder is None:
        # CPU on purpose: MiniLM is tiny, and keeping it off the GPU leaves all
        # of the GPU memory for the language model.
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")

    return _embedder

