"""
Streamlit front end for the StudyForge.

Reuses the exact same pipelines and generation functions as the
CLI (app.py) — this file is purely presentation.
"""

import os

# torch and faiss each bundle their own OpenMP runtime. Loading both
# in one process on Windows causes a delayed, deterministic crash
# (observed as an access violation in an unrelated DLL, sometimes
# minutes after startup) unless this is set before either imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import pyarrow before the ML/data stack to avoid a Windows native import-order crash.
import pyarrow  # noqa: F401

import sys
import asyncio

if sys.platform == "win32":
    # Streamlit's server defaults to Windows' ProactorEventLoop, which
    # has a known access-violation conflict when it shares a process
    # with PyTorch's native threads. Forcing the simpler
    # SelectorEventLoop avoids the crash. Must run before torch or
    # streamlit create any event loop.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import torch

# Streamlit's file watcher inspects every imported module's __path__
# to detect changes for auto-reload. torch.classes exposes a custom
# __path__ object that isn't a real path list, which raises a noisy
# (but harmless) RuntimeError during that inspection. Emptying it
# here avoids the error without affecting torch itself.
torch.classes.__path__ = []

import json
import tempfile
from pathlib import Path

import streamlit as st

from src.pipelines import (
    run_text_pipeline,
    run_pdf_pipeline,
    run_youtube_pipeline,
    run_web_pipeline,
)
from src.summarizer import summarize, summary_word_limit
from src.flashcards import generate_flashcards, parse_flashcards
from src.quiz.quiz import generate_quiz
from src.mindmap import generate_mindmap_data, render_tree
from src.rag.qa import answer_question
from src.prompt import SUMMARY_PROMPTS
from src.llm import set_progress_hook
from src.server_manager import ensure_running, stop_server


st.set_page_config(
    page_title="StudyForge",
    page_icon="📚",
    layout="wide",
)


# ---------------------------------------------------------------------
# Model server: started automatically, downloaded only the first time
# ---------------------------------------------------------------------

@st.cache_resource(
    show_spinner="Starting the model server (the first time this downloads llama.cpp, ~1 GB)..."
)
def _start_model_server():
    return ensure_running()


try:
    _server = _start_model_server()
except Exception as error:
    st.error(f"Couldn't start the model server.\n\n{error}")
    st.stop()


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

DEFAULTS = {
    "condensed_text": None,
    "raw_text": None,
    "output_dir": None,
    "source_name": None,
    "last_summary": None,
    "last_flashcards": None,
    "quiz": None,
    "quiz_submitted": False,
    "quiz_answers": {},
    "mindmap_data": None,
    "chat_history": [],
}

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_generated_content():
    """Clear per-document results when a new source is processed."""
    for key in (
        "last_summary",
        "last_flashcards",
        "quiz",
        "quiz_submitted",
        "quiz_answers",
        "mindmap_data",
        "chat_history",
    ):
        st.session_state[key] = DEFAULTS[key]


# ---------------------------------------------------------------------
# Sidebar: source selection + processing
# ---------------------------------------------------------------------

st.sidebar.title("📚 StudyForge")
st.sidebar.caption(
    "Turn a PDF, YouTube video, or web article into "
    "summaries, flashcards, quizzes, mindmaps — and ask it questions."
)

source = st.sidebar.radio(
    "Source type",
    ["PDF", "YouTube", "Web Article", "Pasted Text"],
    horizontal=False,
)

pdf_file = None
url_input = ""
pasted_text = ""
pasted_title = ""

if source == "PDF":
    pdf_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
elif source == "YouTube":
    url_input = st.sidebar.text_input("YouTube URL")
elif source == "Web Article":
    url_input = st.sidebar.text_input("Article URL")
else:
    pasted_title = st.sidebar.text_input("Title (optional)")
    pasted_text = st.sidebar.text_area("Paste the article text", height=220)

process_clicked = st.sidebar.button(
    "Process", type="primary", use_container_width=True
)

status_box = st.sidebar.empty()


def report(message: str) -> None:
    status_box.info(message)


# Live progress while the model writes. Updating the page from inside the
# generation loop is also what lets Streamlit's Stop button take effect
# straight away instead of after the whole answer has been generated.
gen_box = st.sidebar.empty()


def _show_progress(message) -> None:
    if message is None:
        gen_box.empty()
    else:
        gen_box.caption(f"⏳ {message}. Press ⏹ Cancel generation to stop.")


set_progress_hook(_show_progress)


st.sidebar.button("⏹ Cancel generation", use_container_width=True)


# ---------------------------------------------------------------------
# Shut down the app (and the model server) from the page
# ---------------------------------------------------------------------

st.sidebar.divider()

if st.sidebar.button("⏻ Shut down app", use_container_width=True):
    st.session_state["confirm_shutdown"] = True

if st.session_state.get("confirm_shutdown"):

    stop_server_too = st.sidebar.checkbox(
        "Also stop llama-server (frees the GPU)", value=True
    )

    confirm_col, cancel_col = st.sidebar.columns(2)

    if cancel_col.button("Cancel", use_container_width=True):
        st.session_state["confirm_shutdown"] = False
        st.rerun()

    if confirm_col.button("Confirm", type="primary", use_container_width=True):
        import time

        st.sidebar.success("Shutting down. You can close this tab.")

        if stop_server_too:
            stop_server()

        time.sleep(1.5)  # let the message reach the browser
        os._exit(0)


if process_clicked:

    try:
        if source == "PDF":

            if pdf_file is None:
                st.sidebar.error("Upload a PDF first.")
                st.stop()

            temp_dir = Path(tempfile.mkdtemp())
            temp_path = temp_dir / pdf_file.name
            temp_path.write_bytes(pdf_file.getvalue())

            with st.sidebar, st.spinner("Processing PDF..."):
                condensed_text, raw_text, output_dir = run_pdf_pipeline(
                    str(temp_path), on_progress=report
                )

            source_name = pdf_file.name

        elif source == "YouTube":

            if not url_input.strip():
                st.sidebar.error("Enter a YouTube URL first.")
                st.stop()

            with st.sidebar, st.spinner("Processing video..."):
                condensed_text, raw_text, output_dir = run_youtube_pipeline(
                    url_input.strip(), on_progress=report
                )

            source_name = url_input.strip()

        elif source == "Pasted Text":

            if len(pasted_text.split()) < 50:
                st.sidebar.error("Paste at least a few paragraphs of text first.")
                st.stop()

            with st.sidebar, st.spinner("Processing text..."):
                condensed_text, raw_text, output_dir = run_text_pipeline(
                    pasted_text, pasted_title.strip(), on_progress=report
                )

            source_name = pasted_title.strip() or "Pasted text"

        else:

            if not url_input.strip():
                st.sidebar.error("Enter an article URL first.")
                st.stop()

            with st.sidebar, st.spinner("Processing article..."):
                condensed_text, raw_text, output_dir = run_web_pipeline(
                    url_input.strip(), on_progress=report
                )

            source_name = url_input.strip()

        st.session_state.condensed_text = condensed_text
        st.session_state.raw_text = raw_text
        st.session_state.output_dir = output_dir
        st.session_state.source_name = source_name

        reset_generated_content()

        status_box.success("Ready!")

    except Exception as error:
        status_box.error(f"Couldn't process that source: {error}")


# ---------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------

st.title("📚 StudyForge")

if not st.session_state.condensed_text:
    st.info(
        "Choose a source in the sidebar and click **Process** "
        "to get started."
    )
    st.stop()

st.caption(f"Loaded: **{st.session_state.source_name}**")

condensed_text = st.session_state.condensed_text
raw_text = st.session_state.raw_text
output_dir: Path = st.session_state.output_dir

tab_summary, tab_flashcards, tab_quiz, tab_mindmap, tab_ask = st.tabs(
    ["Summary", "Flashcards", "Quiz", "Mindmap", "Ask (RAG)"]
)


# ---- Summary ----
with tab_summary:

    col1, col2 = st.columns([2, 1])

    with col1:
        summary_type = st.selectbox(
            "Summary type",
            list(SUMMARY_PROMPTS.keys()),
            format_func=lambda s: s.replace("_", " ").title(),
        )

    with col2:
        word_count = st.number_input(
            "Word count", min_value=30, max_value=1000, value=150, step=10
        )

    word_limit = summary_word_limit(condensed_text, summary_type)

    if int(word_count) > word_limit:
        st.caption(
            f"This document supports summaries up to about {word_limit} words "
            "with this model, so longer requests are shortened to that."
        )

    if st.button("Generate Summary", key="gen_summary"):
        with st.spinner("Generating summary..."):
            result = summarize(
                condensed_text=condensed_text,
                summary_type=summary_type,
                word_count=int(word_count),
            )
        (output_dir / "summary.txt").write_text(result, encoding="utf-8")
        st.session_state.last_summary = result

    if st.session_state.last_summary:
        st.markdown(st.session_state.last_summary)


# ---- Flashcards ----
with tab_flashcards:

    col1, col2 = st.columns(2)

    with col1:
        num_flashcards = st.number_input(
            "Number of flashcards", min_value=1, max_value=50, value=10
        )

    with col2:
        words_per_card = st.number_input(
            "Words per answer", min_value=5, max_value=200, value=40
        )

    if st.button("Generate Flashcards", key="gen_flashcards"):
        with st.spinner("Generating flashcards..."):
            result = generate_flashcards(
                condensed_text=condensed_text,
                num_flashcards=int(num_flashcards),
                words_per_card=int(words_per_card),
                source_text=raw_text,
            )
        (output_dir / "flashcards.txt").write_text(result, encoding="utf-8")
        st.session_state.last_flashcards = parse_flashcards(result)
        st.session_state.last_flashcards_requested = int(num_flashcards)

    if st.session_state.last_flashcards:
        made = len(st.session_state.last_flashcards)
        asked = st.session_state.get("last_flashcards_requested", made)

        if made < asked:
            st.info(
                f"Made {made} of the {asked} flashcards you asked for. The model "
                "wrote fewer cards than asked, or some were repeats and were "
                "removed. Generate again to fill the rest."
            )
        else:
            st.caption(f"{made} flashcards")

        for i, card in enumerate(st.session_state.last_flashcards, start=1):
            with st.expander(f"**{i}.** {card['question']}"):
                st.write(card["answer"])


# ---- Quiz ----
with tab_quiz:

    col1, col2 = st.columns(2)

    with col1:
        num_questions = st.number_input(
            "Number of questions", min_value=1, max_value=20, value=5
        )

    with col2:
        words_per_explanation = st.number_input(
            "Words per explanation", min_value=5, max_value=100, value=20
        )

    if st.button("Generate Quiz", key="gen_quiz"):
        try:
            with st.spinner("Generating quiz..."):
                quiz = generate_quiz(
                    condensed_text=condensed_text,
                    num_questions=int(num_questions),
                    words_per_explanation=int(words_per_explanation),
                )

            (output_dir / "quiz.json").write_text(
                json.dumps(quiz, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )

            st.session_state.quiz = quiz
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}

        except ValueError as error:
            st.error(f"Couldn't generate a valid quiz: {error}")

    quiz = st.session_state.quiz

    if quiz:

        with st.form("quiz_form"):

            for i, question in enumerate(quiz):

                st.markdown(f"**{i + 1}. {question['question']}**")

                st.radio(
                    "Choose one",
                    ["A", "B", "C", "D"],
                    format_func=lambda letter, q=question: f"{letter}. {q[letter]}",
                    index=None,
                    key=f"quiz_choice_{i}",
                    label_visibility="collapsed",
                )

            submitted = st.form_submit_button("Submit Quiz")

        if submitted:
            st.session_state.quiz_submitted = True
            st.session_state.quiz_answers = {
                i: st.session_state.get(f"quiz_choice_{i}")
                for i in range(len(quiz))
            }

        if st.session_state.quiz_submitted:

            answers = st.session_state.quiz_answers
            score = 0

            for i, question in enumerate(quiz):

                chosen = answers.get(i)
                correct = chosen == question["answer"]

                if correct:
                    score += 1

                icon = "✅" if correct else "❌"
                correct_letter = question["answer"]

                st.markdown(
                    f"{icon} **Q{i + 1}.** Correct answer: "
                    f"**{correct_letter}. {question[correct_letter]}**"
                )
                st.caption(question["explanation"])

            percentage = (score / len(quiz)) * 100
            st.metric("Score", f"{score}/{len(quiz)}", f"{percentage:.1f}%")


# ---- Mindmap ----
with tab_mindmap:

    if st.button("Generate Mindmap", key="gen_mindmap"):
        try:
            with st.spinner("Generating mindmap..."):
                data = generate_mindmap_data(condensed_text)
            st.session_state.mindmap_data = data
        except ValueError as error:
            st.session_state.mindmap_data = None
            st.error(f"Couldn't generate a valid mindmap: {error}")

    data = st.session_state.mindmap_data

    if data:

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        def _escape(label: str) -> str:
            return label.replace('"', "'")

        dot_lines = [
            "digraph G {",
            "rankdir=LR;",
            'node [shape=box, style="rounded,filled", '
            'fillcolor="#eef3ff", fontname="Helvetica"];',
        ]

        for node in nodes:
            dot_lines.append(f'"{_escape(node)}";')

        for edge in edges:
            if len(edge) == 2:
                parent, child = edge
                dot_lines.append(
                    f'"{_escape(parent)}" -> "{_escape(child)}";'
                )

        dot_lines.append("}")

        st.graphviz_chart("\n".join(dot_lines), use_container_width=True)

        with st.expander("Text tree view"):
            st.text(render_tree(data))


# ---- Ask (RAG) ----
with tab_ask:

    st.caption(
        "Ask a question and get an answer grounded in the "
        "original document, not just the condensed summary."
    )

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask something about this document...")

    if question:

        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            source_chunks = []
            try:
                with st.spinner("Searching document..."):
                    result = answer_question(
                        question=question,
                        raw_text=raw_text,
                        output_dir=output_dir,
                        stream=True,
                    )
                source_chunks = result.get("sources", [])

                if "answer_stream" in result:
                    answer = st.write_stream(result["answer_stream"])
                else:
                    answer = result["answer"]
                    st.markdown(answer)
            except Exception as error:
                answer = f"Couldn't answer that: {error}"
                st.markdown(answer)

            # Temporary RAG diagnostics: inspect exactly what the LLM received.
            if source_chunks:
                with st.expander(f"Retrieved source chunks ({len(source_chunks)})"):
                    st.caption(
                        "These are the passages passed to the LLM for this answer. "
                        "If the required fact is missing here, investigate retrieval; "
                        "if it is present but the answer is wrong, investigate the prompt/model."
                    )
                    for source_index, source_chunk in enumerate(source_chunks, start=1):
                        st.markdown(f"**Chunk {source_index}**")
                        st.text(source_chunk)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )


