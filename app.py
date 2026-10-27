from src.utils import feature_menu
from src.pipelines import (
    run_pdf_pipeline,
    run_youtube_pipeline,
    run_web_pipeline,
)


def prepare_pdf():
    pdf_path = input("\nEnter PDF path: ").strip()
    return run_pdf_pipeline(pdf_path, on_progress=lambda msg: print(f"\n{msg}"))


def prepare_youtube():
    url = input("\nEnter YouTube URL: ").strip()
    return run_youtube_pipeline(url, on_progress=lambda msg: print(f"\n{msg}"))


def prepare_web():
    url = input("\nEnter Article URL: ").strip()
    return run_web_pipeline(url, on_progress=lambda msg: print(f"\n{msg}"))


def main():

    while True:

        print("\n" + "=" * 60)
        print("AI LEARNING COMPANION")
        print("=" * 60)

        print("Choose Input Source\n")

        print("1. PDF")
        print("2. YouTube")
        print("3. Web Article")
        print("0. Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":

            try:
                condensed_text, raw_text, output_dir = prepare_pdf()
            except Exception as error:
                print(f"\nCouldn't process that PDF: {error}")
                continue

            feature_menu(
                condensed_text=condensed_text,
                raw_text=raw_text,
                output_dir=output_dir,
            )

        elif choice == "2":

            try:
                condensed_text, raw_text, output_dir = prepare_youtube()
            except Exception as error:
                print(f"\nCouldn't process that video: {error}")
                continue

            feature_menu(
                condensed_text=condensed_text,
                raw_text=raw_text,
                output_dir=output_dir,
            )

        elif choice == "3":

            try:
                condensed_text, raw_text, output_dir = prepare_web()
            except Exception as error:
                print(f"\nCouldn't process that article: {error}")
                continue

            feature_menu(
                condensed_text=condensed_text,
                raw_text=raw_text,
                output_dir=output_dir,
            )

        elif choice == "0":

            print("\nGoodbye!")

            break

        else:

            print("\nInvalid choice.")


if __name__ == "__main__":
    main()
