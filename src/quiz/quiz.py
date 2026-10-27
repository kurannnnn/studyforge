import json
import re

from src.llm import generate, words_to_tokens
from src.prompt import QUIZ_PROMPTS
from src.quiz.shuffle import shuffle_options
from src.config import (
    JSON_GENERATION_RETRIES,
    OUTPUT_TOKENS_PER_CALL,
    QUIZ_MAX_PER_CALL,
)

_QUESTION_KEY = re.compile(r'"question"\s*:')

REQUIRED_KEYS = {
    "question",
    "A",
    "B",
    "C",
    "D",
    "answer",
    "explanation",
}


def _tokens_per_question(words_per_explanation: int) -> int:
    # Question, four options, the answer letter, JSON punctuation and the explanation.
    return 120 + words_to_tokens(words_per_explanation)


def _generate_quiz_once(
    condensed_text: str,
    num_questions: int,
    words_per_explanation: int,
    avoid: list[str] | None = None,
    done_before: int = 0,
    total: int | None = None,
):
    """
    Single attempt at generating + validating a quiz.
    Raises ValueError on any structural problem.
    """

    prompt = QUIZ_PROMPTS["default"].format(
        text=condensed_text,
        num_questions=num_questions,
        words_per_explanation=words_per_explanation,
    )

    if avoid:
        prompt += (
            "\n\nThese questions already exist. Do NOT repeat or rephrase them; "
            "cover different concepts:\n" + "\n".join(f"- {q}" for q in avoid[-40:])
        )

    total = total or (done_before + num_questions)

    response = generate(
        prompt=prompt,
        max_new_tokens=max(
            int(num_questions * _tokens_per_question(words_per_explanation) * 1.3) + 150,
            512,
        ),
        progress=lambda text: (
            f"Quiz: {min(done_before + len(_QUESTION_KEY.findall(text)), total)}"
            f"/{total} questions made"
        ),
    )

    start = response.find("[")
    end = response.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(
            "Model did not return valid JSON."
        )

    json_text = response[start:end + 1]

    try:

        quiz = json.loads(json_text)

    except json.JSONDecodeError as error:

        print("\n============= MODEL OUTPUT =============\n")
        print(response)
        print("\n========================================\n")

        raise ValueError(
            "Model did not return valid JSON."
        ) from error

    if not isinstance(quiz, list):
        raise ValueError(
            "Model output must be a list of questions."
        )

    if len(quiz) != num_questions:
        raise ValueError(
            f"Expected {num_questions} questions, got {len(quiz)}."
        )

    for i, question in enumerate(quiz, start=1):

        if not isinstance(question, dict):
            raise ValueError(
                f"Question {i} is not a dictionary."
            )

        missing = REQUIRED_KEYS - question.keys()

        if missing:
            raise ValueError(
                f"Question {i} is missing keys: {missing}"
            )

        if question["answer"] not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"Question {i} has an invalid answer: {question['answer']}"
            )

    return quiz


def _generate_batch(
    condensed_text: str,
    num_questions: int,
    words_per_explanation: int,
    avoid: list[str],
    done_before: int = 0,
    total: int | None = None,
):
    """
    One batch of questions, retried once (JSON_GENERATION_RETRIES) if the model
    returns malformed or structurally invalid JSON, since local LLMs
    occasionally slip on strict formatting instructions.
    """

    attempts = JSON_GENERATION_RETRIES + 1
    last_error: ValueError | None = None

    for attempt in range(1, attempts + 1):

        try:
            return _generate_quiz_once(
                condensed_text=condensed_text,
                num_questions=num_questions,
                words_per_explanation=words_per_explanation,
                avoid=avoid,
                done_before=done_before,
                total=total,
            )

        except ValueError as error:

            last_error = error

            if attempt < attempts:
                print(
                    f"\nQuiz generation attempt {attempt} failed "
                    f"({error}). Retrying...\n"
                )

    raise last_error


def generate_quiz(
    condensed_text: str,
    num_questions: int = 5,
    words_per_explanation: int = 20,
):
    """
    Generate a multiple-choice quiz.

    Large quizzes are produced in several batches that fit the model's context
    window, each told which questions already exist so they do not repeat.
    """

    per_call = max(
        1,
        min(QUIZ_MAX_PER_CALL, OUTPUT_TOKENS_PER_CALL // _tokens_per_question(words_per_explanation)),
    )

    quiz: list[dict] = []

    while len(quiz) < num_questions:

        wanted = min(per_call, num_questions - len(quiz))

        quiz.extend(
            _generate_batch(
                condensed_text,
                wanted,
                words_per_explanation,
                avoid=[q["question"] for q in quiz],
                done_before=len(quiz),
                total=num_questions,
            )
        )

    # Randomize question order and option order
    return shuffle_options(quiz)