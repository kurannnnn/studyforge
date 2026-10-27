CONDENSE_PROMPT = """
You are an expert at compressing information.

Condense the following text while preserving all important information.

Rules:
- Keep approximately {word_count} words.
- Preserve key concepts.
- Preserve technical terms.
- Preserve formulas.
- Preserve definitions.
- Preserve important facts.
- Remove repetition.
- Remove filler sentences.
- Remove unnecessary examples unless essential for understanding.
- Do NOT explain.
- Do NOT add information.
- Keep the original language.
- Return ONLY the condensed text.

Text:
{text}
"""


SUMMARY_PROMPTS = {

    "short": """
Summarize the following text.

Rules:
- Produce approximately {word_count} words.
- Return ONLY the summary.
- Preserve technical concepts.
- Do not add information.
- Generate the summary entirely in English, regardless of the input language.

Text:
{text}
""",

    "detailed": """
Create a detailed summary.

Rules:
- Produce approximately {word_count} words.
- Include important technical details.
- Preserve terminology.
- Return only the summary.
- Generate the summary entirely in English, regardless of the input language.

Text:
{text}
""",

    "bullets": """
Summarize the following text as bullet points.

Rules:
- Approximately {word_count} words.
- Use concise bullet points.
- Preserve technical concepts.
- Generate the bullet points entirely in English, regardless of the input language.

Text:
{text}
""",

    "study_notes": """
Convert the following text into study notes.

Rules:
- Approximately {word_count} words.
- Use headings.
- Use bullet points where appropriate.
- Keep all technical terms.
- Generate the notes entirely in English, regardless of the input language.

Text:
{text}
""",

    "exam_notes": """
Create exam revision notes.

Rules:
- Approximately {word_count} words.
- Include formulas if present.
- Include definitions.
- Focus only on examinable concepts.
- Generate the notes entirely in English, regardless of the input language.

Text:
{text}
"""
}


FLASHCARD_PROMPTS = {

    "default": """
You are an expert teacher.

Create study flashcards from the following text.

Rules:

- Generate EXACTLY {num_flashcards} flashcards.
- Every flashcard must contain ONE Question and ONE Answer.
- Each answer should contain approximately {words_per_card} words.
- Focus on the most important concepts.
- Preserve technical terminology.
- Do not invent information.
- Generate ALL flashcards in English regardless of the input language.
- Return ONLY the flashcards.

Format:

Flashcard 1

Question:
...

Answer:
...

Flashcard 2

Question:
...

Answer:
...

Text:
{text}
"""
}


QUIZ_PROMPTS = {

    "default": """
You are an expert teacher.

Generate EXACTLY {num_questions} high-quality multiple-choice questions.

The questions should collectively cover different sections of the text. Avoid testing the same concept more than once unless absolutely necessary.

Rules:

- Return ONLY valid JSON.
- Do NOT include markdown.
- Do NOT include ```json.
- Cover different parts of the text.
- Do NOT ask two questions about the same concept.
- Avoid repeating questions from previous generations whenever possible.
- Prefer a mix of definitions, concepts, comparisons, applications, reasoning, and factual questions.
- Prioritize less obvious but important concepts after covering the basics.
- Do NOT include any text before or after the JSON.
- Generate EXACTLY {num_questions} questions.
- Each question must have exactly four options.
- The answer must be exactly one of: "A", "B", "C", or "D".
- Each explanation should contain approximately {words_per_explanation} words.
- Use ONLY the information provided.
- Do NOT invent facts.
- Generate the entire quiz in English regardless of the input language.

Return EXACTLY this structure:

[
    {{
        "question": "...",
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "...",
        "answer": "B",
        "explanation": "..."
    }}
]

Text:
{text}
"""
}


RAG_PROMPT = """
Answer the user's question about the document using the supplied context as evidence.

Guidelines:
- Use the context as the source of truth; do not add unsupported facts.
- Answer the question that was actually asked, not a nearby or broader question.
- Preserve important terminology, distinctions, qualifications, and relationships from the source.
- If asked for multiple items, identify distinct items separately and do not merge overlapping concepts.
- Do not mistake an example, consequence, advantage, or side observation for a requested category or criterion.
- For comparisons, state the dimension being compared and explain it using evidence from the context.
- If the context is incomplete or ambiguous, say what cannot be established instead of guessing.
- Be concise but include enough evidence to answer every part of the question.
- Answer in the language requested by the user; use English by default.

Context:
{context}

Question:
{question}

Answer:
"""


MINDMAP_PROMPTS = """
You are an expert knowledge graph generator.

Convert the study material into a JSON knowledge graph.

Rules:

- Return ONLY valid JSON.
- No markdown.
- No explanations.
- No extra text.

Format:

{{
"nodes":[
"Artificial Intelligence",
"Machine Learning",
"Regression"
],

"edges":[

    ["Artificial Intelligence","Machine Learning"],

    ["Machine Learning","Regression"]

]

}}

Rules:

- Keep node names short.
- Remove duplicate nodes.
- Maximum 30 nodes.
- Every edge must connect existing nodes.
- Build a hierarchical concept map.

Study Material:

{text}
"""