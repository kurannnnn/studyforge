from pathlib import Path

from src.llm import generate, generate_stream, count_tokens, output_room
from src.prompt import RAG_PROMPT
from src.config import RAG_TOP_K, RAG_MAX_NEW_TOKENS, RAG_REPETITION_PENALTY
from src.rag.vectorstore import get_or_build_index, search


def _fit_context(chunks: list[str], question: str) -> list[str]:
    """
    Keep as many retrieved chunks as fit in the model's context window
    next to the question and the room reserved for the answer.
    """

    empty_prompt = RAG_PROMPT.format(context="", question=question)
    budget = output_room(empty_prompt) - RAG_MAX_NEW_TOKENS

    kept, used = [], 0

    for chunk in chunks:
        cost = count_tokens(chunk) + 8   # +8 for the separator between chunks

        if used + cost > budget:
            break

        kept.append(chunk)
        used += cost

    return kept


def answer_question(
    question: str,
    raw_text: str,
    output_dir: Path,
    top_k: int = RAG_TOP_K,
    stream: bool = False,
) -> dict:
    """
    Answer a document question and return the source chunks used.

    With stream=True the result holds "answer_stream" (a generator of text
    pieces) instead of "answer", so a UI can display it as it is written.
    """
    index, chunks = get_or_build_index(raw_text, output_dir)
    relevant_chunks = _fit_context(search(index, chunks, question, top_k), question)

    if not relevant_chunks:
        return {
            "answer": "I couldn't find anything relevant to that question in the document.",
            "sources": [],
        }

    context = "\n\n---\n\n".join(relevant_chunks)
    prompt = RAG_PROMPT.format(context=context, question=question)

    if stream:
        return {
            "answer_stream": generate_stream(
                prompt=prompt,
                max_new_tokens=RAG_MAX_NEW_TOKENS,
                temperature=0.0,
                repetition_penalty=RAG_REPETITION_PENALTY,
            ),
            "sources": relevant_chunks,
        }

    answer = generate(
        prompt=prompt,
        max_new_tokens=RAG_MAX_NEW_TOKENS,
        temperature=0.0,
        repetition_penalty=RAG_REPETITION_PENALTY,
    )

    return {"answer": answer, "sources": relevant_chunks}
