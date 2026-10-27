from pathlib import Path

from src.summarizer import summarize
from src.flashcards import generate_flashcards
from src.quiz.quiz import generate_quiz
from src.quiz.quiz_runner import run_quiz
from src.mindmap import generate_mindmap
from src.rag.qa import answer_question


def feature_menu(
    condensed_text: str,
    raw_text: str,
    output_dir: Path,
):
    """
    Display the feature menu and generate outputs.
    """

    while True:

        print("\n" + "=" * 60)
        print("AI LEARNING COMPANION")
        print("=" * 60)

        print("1. Summarize")
        print("2. Flashcards")
        print("3. Quiz")
        print("4. Mindmap")
        print("5. Ask a Question (RAG)")
        print("0. Back")

        choice = input(
            "\nChoose an option: "
        ).strip()

        if choice == "1":

            print("\nSummary Types")
            print("1. Short")
            print("2. Detailed")
            print("3. Bullet Points")
            print("4. Study Notes")
            print("5. Exam Revision Notes")

            summary_types = {
                "1": "short",
                "2": "detailed",
                "3": "bullets",
                "4": "study_notes",
                "5": "exam_notes",
            }

            summary_choice = input(
                "\nChoose summary type: "
            ).strip()

            summary_type = summary_types.get(
                summary_choice,
                "short",
            )

            words = input(
                "Desired summary length [150]: "
            ).strip()

            word_count = (
                int(words)
                if words
                else 150
            )

            print("\nGenerating summary...\n")

            result = summarize(
                condensed_text=condensed_text,
                summary_type=summary_type,
                word_count=word_count,
            )

            filename = "summary.txt"

        elif choice == "2":

            num_flashcards = int(
                input(
                    "Number of flashcards: "
                )
            )

            words_per_card = int(
                input(
                    "Words per flashcard: "
                )
            )

            print(
                "\nGenerating flashcards...\n"
            )

            result = generate_flashcards(
                condensed_text=condensed_text,
                num_flashcards=num_flashcards,
                words_per_card=words_per_card,
                source_text=raw_text,
            )

            filename = "flashcards.txt"

        elif choice == "3":

            num_questions = int(
                input(
                    "Number of questions: "
                )
            )

            words = int(
                input(
                    "Words per explanation: "
                )
            )

            print("\nGenerating quiz...\n")

            quiz = generate_quiz(
                condensed_text=condensed_text,
                num_questions=num_questions,
                words_per_explanation=words,
            )

            import json

            with open(
                output_dir / "quiz.json",
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    quiz,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            print(
                "\nQuiz saved to outputs.\n"
            )

            run_quiz(quiz)

            continue

        elif choice == "4":

            print(
                "\nGenerating mindmap...\n"
            )

            result = generate_mindmap(
                condensed_text=condensed_text,
            )

            filename = "mindmap.txt"

        elif choice == "5":

            print(
                "\nAsk questions about the document. "
                "Type 'back' to return to the menu.\n"
            )

            while True:

                question = input("\nYour question: ").strip()

                if not question:
                    continue

                if question.lower() in {"back", "0", "exit", "quit"}:
                    break

                print("\nSearching document...\n")

                try:
                    result = answer_question(
                        question=question,
                        raw_text=raw_text,
                        output_dir=output_dir,
                    )
                except Exception as error:
                    print(f"\nCouldn't answer that: {error}")
                    continue

                print("=" * 60)
                print(result["answer"])
                print("=" * 60)

            continue

        elif choice == "0":

            break

        else:

            print("\nInvalid choice.")
            continue

        print("=" * 60)
        print(result)
        print("=" * 60)

        with open(
            output_dir / filename,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(result)

        print(
            f"\nSaved to {output_dir / filename}"
        )