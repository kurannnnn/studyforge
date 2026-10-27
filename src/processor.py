from src.llm import generate_many, count_tokens, TOKENS_PER_WORD
from src.text_splitter import split_text
from src.prompt import CONDENSE_PROMPT
from src.config import (
    CONDENSE_CHUNK_SIZE,
    CONDENSE_CHUNK_OVERLAP,
    CONDENSE_TARGET_TOKENS,
    CONDENSE_MIN_WORDS_PER_CHUNK,
    CONDENSE_MAX_WORDS_PER_CHUNK,
    CONDENSE_MAX_PASSES,
)


def _noop(message: str) -> None:
    pass


def condense_settings_tag() -> str:
    """
    Identifies the settings that shape the condensed text. A cached condensed
    text made with different settings is out of date and gets rebuilt.
    """

    return (
        f"target={CONDENSE_TARGET_TOKENS};chunk={CONDENSE_CHUNK_SIZE};"
        f"overlap={CONDENSE_CHUNK_OVERLAP};"
        f"words={CONDENSE_MIN_WORDS_PER_CHUNK}-{CONDENSE_MAX_WORDS_PER_CHUNK}"
    )


def words_per_chunk(chunk_count: int) -> int:
    """
    Word budget for each chunk so that all the condensed chunks together land
    near CONDENSE_TARGET_TOKENS: gentle compression for short documents,
    harder compression for long ones.
    """

    tokens_each = CONDENSE_TARGET_TOKENS * 0.85 / chunk_count
    words = int(tokens_each / TOKENS_PER_WORD * 1.2)

    return max(CONDENSE_MIN_WORDS_PER_CHUNK, min(CONDENSE_MAX_WORDS_PER_CHUNK, words))


def condense_document(text: str, on_progress=_noop) -> str:
    """
    Condense a long document so the other features can work from it.

    - A document that already fits in CONDENSE_TARGET_TOKENS is returned as is.
    - Otherwise every chunk is condensed (several at a time, depending on the
      server's slots) and the result is condensed again only while it is still
      over the target.
    """

    if count_tokens(text) <= CONDENSE_TARGET_TOKENS:
        on_progress("Document is short enough; skipping condensing.")
        return text

    chunks = split_text(text)
    combined = text

    for pass_number in range(1, CONDENSE_MAX_PASSES + 1):

        words = words_per_chunk(len(chunks))

        prompts = [
            CONDENSE_PROMPT.format(text=chunk, word_count=words)
            for chunk in chunks
        ]

        on_progress(
            f"Condensing pass {pass_number}: {len(prompts)} chunk(s), about {words} words each..."
        )

        condensed = generate_many(
            prompts,
            word_count=words,
            on_done=lambda done, total: on_progress(
                f"Pass {pass_number}: {done}/{total} chunks condensed..."
            ),
        )

        combined = "\n\n".join(condensed)

        if len(condensed) == 1 or count_tokens(combined) <= CONDENSE_TARGET_TOKENS:
            break

        chunks = split_text(combined)

    return combined
