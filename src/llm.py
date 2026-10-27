from concurrent.futures import ThreadPoolExecutor, as_completed

from src import llm_server
from src.model import load_tokenizer
from src.config import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_REPETITION_PENALTY,
    SERVER_CTX_PER_SLOT,
)

SYSTEM_PROMPT = (
    "You are a careful, evidence-grounded study assistant. "
    "Answer from the supplied context, preserve the source's "
    "terminology and distinctions, and do not invent missing facts. "
    "When asked for multiple items, keep each item distinct. "
    "If evidence is insufficient, say so clearly."
)

# Tokens the chat template and system prompt add around the user prompt.
_TEMPLATE_MARGIN = 100

TOKENS_PER_WORD = 1.7

# Longest single reply we allow. It only stops runaway output.
MAX_OUTPUT_TOKENS = 1600

# Optional callback for UIs: called with the number of characters generated so
# far (and with None when a generation finishes). Because it runs between
# streamed pieces, a UI that updates itself inside it (for example Streamlit)
# can also react to a Stop button while the model is still working.
_progress_hook = None


def set_progress_hook(hook) -> None:
    global _progress_hook
    _progress_hook = hook


def report(message) -> None:
    """Show a status message in the UI (None clears it), if a UI is listening."""
    if _progress_hook is not None:
        _progress_hook(message)


# ---------------------------------------------------------------------
# Token budgeting
# ---------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(load_tokenizer().encode(text))


def output_room(prompt: str) -> int:
    """How many tokens the model can still write after reading this prompt."""
    return SERVER_CTX_PER_SLOT - count_tokens(prompt) - _TEMPLATE_MARGIN


def words_to_tokens(words: int) -> int:
    return int(words * TOKENS_PER_WORD)


def default_max_tokens(word_count: int) -> int:
    # Room for bullets, punctuation and the occasional long word.
    return min(max(words_to_tokens(word_count) + 48, 64), MAX_OUTPUT_TOKENS)


def _fit(prompt: str, wanted: int) -> int:
    """Never ask for more tokens than the context window has room for."""

    room = output_room(prompt)

    if room < 64:
        raise ValueError(
            "The text is too long for the model's context window. "
            "Try a shorter document."
        )

    return min(wanted, room)


# ---------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------

def generate(
    prompt: str,
    word_count: int = 150,
    max_new_tokens: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    progress=None,
) -> str:
    """
    Generate a response from the language model.

    progress, if given, is a function that turns the text written so far into
    a short status message for the UI, for example "Quiz: 3/10 questions made".
    """

    if max_new_tokens is None:
        max_new_tokens = default_max_tokens(word_count)

    max_new_tokens = _fit(prompt, max_new_tokens)

    if _progress_hook is None:
        return llm_server.chat(
            prompt, SYSTEM_PROMPT, max_new_tokens,
            temperature, top_p, repetition_penalty,
        )

    stream = llm_server.chat_stream(
        prompt, SYSTEM_PROMPT, max_new_tokens,
        temperature, top_p, repetition_penalty,
    )

    pieces = []
    text = ""

    try:
        for piece in stream:
            pieces.append(piece)
            text += piece
            _progress_hook(
                progress(text) if progress else f"Generating… {len(text)} characters so far"
            )
    finally:
        # Closing the stream drops the connection, which tells the
        # server to stop generating if we were interrupted.
        stream.close()

    _progress_hook(None)

    return "".join(pieces).strip()


def generate_many(
    prompts: list[str],
    word_count: int = 80,
    repetition_penalty: float = 1.05,
    on_done=None,
    on_result=None,
    max_new_tokens=None,
    temperature: float = 0.0,
) -> list[str]:
    """
    Answer several independent prompts at the same time, as many at once as
    the server has slots, and return the answers in the same order.

    Decoding is memory-bound, so two requests together take much less time
    than the same two one after the other. on_done(done, total) is called
    from the calling thread after each answer, so a UI can update safely.
    """

    from src.server_manager import get_slots

    # max_new_tokens may be one number for all prompts or one per prompt.
    if max_new_tokens is None:
        max_new_tokens = default_max_tokens(word_count)

    wanted = (
        list(max_new_tokens)
        if isinstance(max_new_tokens, (list, tuple))
        else [max_new_tokens] * len(prompts)
    )

    # Token counting happens here, on the calling thread, before the workers start.
    budgets = [_fit(prompt, tokens) for prompt, tokens in zip(prompts, wanted)]

    def _one(index: int) -> str:
        return llm_server.chat(
            prompts[index], SYSTEM_PROMPT, budgets[index],
            temperature, DEFAULT_TOP_P, repetition_penalty,
        )

    results: list[str] = [""] * len(prompts)

    pool = ThreadPoolExecutor(max_workers=get_slots())

    try:
        futures = {pool.submit(_one, i): i for i in range(len(prompts))}

        done = 0

        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done += 1

            if on_result:
                on_result(futures[future], results[futures[future]])

            if on_done:
                on_done(done, len(prompts))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return results


def generate_stream(
    prompt: str,
    max_new_tokens: int = 220,
    temperature: float = 0.0,
    top_p: float = DEFAULT_TOP_P,
    repetition_penalty: float = 1.0,
):
    """
    Yield the answer piece by piece as it is generated, so a UI can show
    text immediately instead of waiting for the whole response.
    """

    yield from llm_server.chat_stream(
        prompt, SYSTEM_PROMPT, _fit(prompt, max_new_tokens),
        temperature, top_p, repetition_penalty,
    )
