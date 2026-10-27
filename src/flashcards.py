import re

from src.llm import generate, generate_many, report, words_to_tokens
from src.prompt import FLASHCARD_PROMPTS
from src.text_splitter import split_text
from src.config import (
    OUTPUT_TOKENS_PER_CALL,
    FLASHCARDS_MAX_PER_CALL,
    FLASHCARDS_PER_SECTION,
)

# Labels the model writes, allowing markdown such as **Question:** or ### Question:
_QUESTION_LABEL = re.compile(r"(?im)^[ \t>#*_-]*question[ \t]*\d*[ \t]*[:.)][ \t*_]*")
_ANSWER_LABEL = re.compile(r"(?im)^[ \t>#*_-]*answer[ \t]*[:.)][ \t*_]*")
_CARD_HEADER = re.compile(r"(?im)^[ \t>#*_-]*flashcard[ \t]*\d*[ \t]*[:.)]?[ \t*_]*$")


def parse_flashcards(text: str) -> list[dict]:
    """
    Parse the model's flashcards into a list of {"question": ..., "answer": ...}
    dicts, for UIs that want to render cards individually rather than as one
    block of text.

    Cards are found by their Question/Answer labels, not by the "Flashcard N"
    headers, because the model sometimes bolds, renumbers or skips a header,
    and a parser that relied on the headers would swallow the next card.
    A card cut off before its answer is skipped.
    """

    cards = []

    starts = list(_QUESTION_LABEL.finditer(text))

    for index, start in enumerate(starts):

        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.end():end]

        answer_label = _ANSWER_LABEL.search(block)

        if not answer_label:
            continue

        question = block[:answer_label.start()]
        answer = block[answer_label.end():]

        # Drop a "Flashcard N" header that belongs to the next card.
        answer = "\n".join(
            line for line in answer.splitlines() if not _CARD_HEADER.match(line)
        )

        question = question.strip(" \t\r\n*_#")
        answer = answer.strip(" \t\r\n*_#")

        if question and answer:
            cards.append({"question": question, "answer": answer})

    return cards


def _clean(text: str) -> str:
    """Join the lines of a field and drop stray one- or two-character lines."""

    lines = [line.strip() for line in text.splitlines()]

    return " ".join(line for line in lines if len(line) > 2).strip()


# Question words and fillers that most questions share. Ignoring them keeps
# "role of the retriever" and "role of the generator" from looking identical.
_FILLER = {
    "what", "which", "who", "why", "how", "when", "where", "is", "are", "was",
    "were", "do", "does", "did", "the", "a", "an", "of", "in", "on", "at", "to",
    "for", "and", "or", "by", "with", "as", "this", "that", "these", "those",
    "it", "its", "can", "be", "from",
}


def _words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-z0-9]+", text.lower())
        if word not in _FILLER
    }


def _similarity(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _find_duplicate(question: str, answer: str, known: list[tuple[set, set]]):
    """
    Index of the earlier card this one repeats, or None. A repeat is almost the
    same question, or a similar question answered with the same facts.
    """

    q_words, a_words = _words(question), _words(answer)

    for index, (known_q, known_a) in enumerate(known):

        question_match = _similarity(q_words, known_q)

        if question_match >= 0.8:
            return index

        if question_match >= 0.5 and _similarity(a_words, known_a) >= 0.6:
            return index

    return None


def _is_duplicate(question: str, answer: str, known: list[tuple[set, set]]) -> bool:
    return _find_duplicate(question, answer, known) is not None


def _render(cards: list[dict]) -> str:
    return "\n\n".join(
        f"Flashcard {i}\n\nQuestion:\n{card['question']}\n\nAnswer:\n{card['answer']}"
        for i, card in enumerate(cards, start=1)
    )


def _spread(items: list, count: int, shift: float = 0.5) -> list:
    """Pick `count` items spread evenly across the list (repeating if needed)."""

    return [
        items[min(len(items) - 1, int((i + shift) * len(items) / count))]
        for i in range(count)
    ]


def _shares(total: int, parts: int) -> list[int]:
    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]


def generate_flashcards(
    condensed_text: str,
    num_flashcards: int,
    words_per_card: int,
    source_text: str | None = None,
) -> str:
    """
    Generate flashcards.

    When the original text is available, the cards are written from sections
    spread evenly across it, a few per section, several sections at a time.
    That way a long document can supply many different cards, where the
    condensed text alone would run out of new points and start repeating.
    Without it, the condensed text is split into parts instead.

    Repeats and malformed cards are dropped, and if the total is still short
    the model is asked again for the rest.
    """

    # Tokens one card needs: the answer, the question and the labels.
    per_card = words_to_tokens(words_per_card) + 60

    per_section = max(1, min(FLASHCARDS_PER_SECTION, OUTPUT_TOKENS_PER_CALL // per_card))
    per_call = max(1, min(FLASHCARDS_MAX_PER_CALL, OUTPUT_TOKENS_PER_CALL // per_card))

    cards: list[dict] = []
    known: list[tuple[set, set]] = []

    shown = [0]   # the count only goes up, even if repeats are removed later

    def progress_for(text: str) -> str:
        shown[0] = max(shown[0], len(cards) + len(_QUESTION_LABEL.findall(text)))
        return f"Flashcards: {min(shown[0], num_flashcards)}/{num_flashcards} made"

    def prompt_for(source: str, wanted: int) -> str:

        prompt = FLASHCARD_PROMPTS["default"].format(
            text=source,
            num_flashcards=wanted,
            words_per_card=words_per_card,
        )

        if cards:
            existing = "\n".join(f"- {card['question']}" for card in cards[-40:])
            prompt += (
                "\n\nThese flashcard questions already exist. Do NOT repeat or "
                "rephrase them; cover different concepts:\n" + existing
            )

        return prompt

    def absorb(text: str, wanted: int, label: str) -> None:

        parsed = parse_flashcards(text)
        repeats = too_short = 0

        for card in parsed:

            question = _clean(card["question"])
            answer = _clean(card["answer"])

            if len(question.split()) < 3 or len(answer.split()) < 3:
                too_short += 1
                continue

            match = _find_duplicate(question, answer, known)

            if match is not None:
                repeats += 1
                print(f"[flashcards]   dropped as repeat: {question!r} (same as {cards[match]['question']!r})")
                continue

            known.append((_words(question), _words(answer)))
            cards.append({"question": question, "answer": answer})

        print(
            f"[flashcards] {label}: asked {wanted}, parsed {len(parsed)}, "
            f"repeats dropped {repeats}, too short {too_short}, total now {len(cards)}"
        )

        if len(parsed) < wanted:
            print(
                f"[flashcards]   model wrote {text.count('Question:')} 'Question:' and "
                f"{text.count('Answer:')} 'Answer:' labels; output ended with {text[-70:]!r}"
            )

    # Main pass.
    sections = split_text(source_text) if source_text else []

    if len(sections) >= 2:

        parts = -(-num_flashcards // per_section)
        shares = _shares(num_flashcards, parts)
        chosen = _spread(sections, parts)

        # The sections are different parts of the document, so the calls do not
        # depend on each other and can run side by side.
        prompts = [prompt_for(section, share) for section, share in zip(chosen, shares)]

        report(f"Flashcards: 0/{num_flashcards} made")
        partial: dict[int, int] = {}

        def on_result(index: int, text: str) -> None:
            partial[index] = len(parse_flashcards(text))
            shown[0] = max(shown[0], sum(partial.values()))
            report(f"Flashcards: {min(shown[0], num_flashcards)}/{num_flashcards} made")

        texts = generate_many(
            prompts,
            max_new_tokens=[max(share * per_card, 256) for share in shares],
            temperature=0.2,
            repetition_penalty=1.1,
            on_result=on_result,
        )

        report(None)

        for i, (text, share) in enumerate(zip(texts, shares), start=1):
            absorb(text, share, f"section {i}/{parts}")

    else:

        parts = -(-num_flashcards // per_call)
        paragraphs = [p.strip() for p in condensed_text.split("\n\n") if p.strip()]

        if parts > 1 and len(paragraphs) >= parts:
            size = len(paragraphs) / parts
            groups = [
                "\n\n".join(paragraphs[round(i * size):round((i + 1) * size)])
                for i in range(parts)
            ]
        else:
            groups = [condensed_text] * parts

        for i, (group, share) in enumerate(zip(groups, _shares(num_flashcards, parts)), start=1):
            text = generate(
                prompt=prompt_for(group, share),
                max_new_tokens=max(share * per_card, 256),
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
                progress=progress_for,
            )
            absorb(text, share, f"part {i}/{parts}")

    # Top-up: ask again for whatever is still missing, from other sections.
    for attempt in range(1, 4):

        if len(cards) >= num_flashcards:
            break

        missing = num_flashcards - len(cards)
        wanted = min(per_call, missing)

        if len(sections) >= 2:
            source = sections[int((attempt * 0.382 % 1.0) * len(sections))]
        else:
            source = condensed_text

        text = generate(
            prompt=prompt_for(source, wanted),
            max_new_tokens=max(wanted * per_card, 256),
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            progress=progress_for,
        )

        absorb(text, wanted, f"top-up {attempt}")

    return _render(cards[:num_flashcards])

