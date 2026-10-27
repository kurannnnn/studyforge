from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path


def load_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    return documents


def extract_text(documents):
    pages = []

    for document in documents:
        pages.append(document.page_content)

    full_text = "\n".join(pages)
    return full_text


def clean_text(full_text):
    cleaned_text = " ".join(full_text.split())
    return cleaned_text


def save_text(cleaned_text, output_path="outputs/extracted_text.txt"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(cleaned_text)