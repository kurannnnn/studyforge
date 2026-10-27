import os

# torch and faiss each bundle their own OpenMP runtime. Loading both
# in one process on Windows causes a delayed, deterministic crash
# unless this is set before either imports. Set here (not just in
# app.py/streamlit_app.py) so it's guaranteed regardless of which
# entry point imports torch first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from transformers import AutoTokenizer

from src.config import MODEL_NAME

_tokenizer = None


def load_tokenizer():
    """
    Load (once) the model's tokenizer: a few MB, no model weights.

    The model itself runs in llama-server; the app only needs the tokenizer
    to count tokens and to cut documents into chunks of the right size.
    """

    global _tokenizer

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

    return _tokenizer