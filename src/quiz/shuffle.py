import random


def shuffle_options(quiz: list[dict]) -> list[dict]:
    """
    Randomize both the question order and
    the option order while preserving the
    correct answers.
    """

    rng = random.Random()

    shuffled_quiz = []

    for question in quiz:

        options = [
            ("A", question["A"]),
            ("B", question["B"]),
            ("C", question["C"]),
            ("D", question["D"]),
        ]

        correct_text = question[
            question["answer"]
        ]

        rng.shuffle(options)

        new_question = {
            "question": question["question"],
            "A": options[0][1],
            "B": options[1][1],
            "C": options[2][1],
            "D": options[3][1],
            "explanation": question["explanation"],
        }

        for letter in ["A", "B", "C", "D"]:

            if new_question[letter] == correct_text:

                new_question["answer"] = letter
                break

        shuffled_quiz.append(
            new_question
        )

    # Shuffle question order
    rng.shuffle(shuffled_quiz)

    return shuffled_quiz