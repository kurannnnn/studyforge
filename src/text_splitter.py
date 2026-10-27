from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.model import load_tokenizer
from src.config import CONDENSE_CHUNK_SIZE, CONDENSE_CHUNK_OVERLAP

_text_splitter = None


def split_text(text: str):
    global _text_splitter

    if _text_splitter is None:

        tokenizer = load_tokenizer()

        _text_splitter = (
            RecursiveCharacterTextSplitter
            .from_huggingface_tokenizer(
                tokenizer=tokenizer,
                chunk_size=CONDENSE_CHUNK_SIZE,
                chunk_overlap=CONDENSE_CHUNK_OVERLAP,
            )
        )

    return _text_splitter.split_text(text)