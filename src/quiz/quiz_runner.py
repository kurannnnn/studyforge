def run_quiz(questions: list[dict]) -> None:
    """
    Run an interactive multiple-choice quiz.
    """

    score = 0
    total = len(questions)

    print("\n" + "=" * 60)
    print("QUIZ STARTED")
    print("=" * 60)

    for i, question in enumerate(questions, start=1):

        print(f"\nQuestion {i}/{total}")
        print("-" * 60)

        print(question["question"])
        print()

        print(f"A. {question['A']}")
        print(f"B. {question['B']}")
        print(f"C. {question['C']}")
        print(f"D. {question['D']}")

        # Keep asking until valid input
        while True:

            user_answer = input(
                "\nYour answer (A/B/C/D): "
            ).strip().upper()

            if user_answer in ["A", "B", "C", "D"]:
                break

            print("Please enter A, B, C or D.")

        if user_answer == question["answer"]:

            print("\nCorrect!")
            score += 1

        else:

            print("\nIncorrect.")
            print(f"Correct Answer: {question['answer']}")

        print("\nExplanation:")
        print(question["explanation"])

        if i != total:
            input("\nPress Enter for the next question...")

    print("\n" + "=" * 60)
    print("QUIZ COMPLETED")
    print("=" * 60)

    print(f"Score: {score}/{total}")

    percentage = (score / total) * 100

    print(f"Percentage: {percentage:.1f}%")

    if percentage >= 90:
        print("Excellent!")

    elif percentage >= 75:
        print("Very Good!")

    elif percentage >= 50:
        print("Good! Keep Practicing.")

    else:
        print("Needs More Revision.")