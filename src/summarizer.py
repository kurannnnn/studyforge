from src.llm import generate, output_room, TOKENS_PER_WORD, MAX_OUTPUT_TOKENS
from src.prompt import SUMMARY_PROMPTS


def summary_word_limit(condensed_text: str, summary_type: str = "short") -> int:
    """
    The longest summary (in words) that fits in the model's context window
    together with this document.
    """

    prompt = SUMMARY_PROMPTS[summary_type].format(text=condensed_text, word_count=0)

    room = min(output_room(prompt), MAX_OUTPUT_TOKENS)

    return max(30, int((room - 48) / TOKENS_PER_WORD))


def summarize(
    condensed_text: str,
    summary_type: str = "short",
    word_count: int = 150,
) -> str:
    """
    Generate a summary from condensed text. A request longer than the context
    window allows is shortened to the longest summary that fits.
    """

    word_count = min(word_count, summary_word_limit(condensed_text, summary_type))

    prompt = SUMMARY_PROMPTS[summary_type].format(
        text=condensed_text,
        word_count=word_count,
    )

    return generate(
        prompt=prompt,
        word_count=word_count,
        progress=lambda text: f"Summary: {min(len(text.split()), word_count)}/{word_count} words written",
    )
