"""
Central configuration for the StudyForge.

Keeping these values here instead of scattered across modules
means tuning the model, chunking, or retrieval behavior only
requires editing one file.
"""

# ---- Model server (llama.cpp's llama-server, started automatically) ----
# MODEL_NAME is only used for its tokenizer (to count tokens and size chunks).
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# The model itself is a GGUF file. If GGUF_PATH points to a file on disk it is
# used directly; otherwise it is downloaded once into the Hugging Face cache.
GGUF_PATH = ""
GGUF_REPO_ID = "alphaduriendur/Qwen3-4B-Instruct-2507-Q4_K_M-GGUF"
GGUF_FILENAME = "qwen3-4b-instruct-2507-q4_k_m.gguf"

SERVER_URL = "http://127.0.0.1:8080"

# The server handles SERVER_SLOTS requests at the same time (used to condense
# several chunks in parallel). Each slot gets SERVER_CTX_PER_SLOT tokens of
# context, and the server is started with slots x context in total.
# Memory use grows with the total, so if the server cannot start, the app
# falls back to a single slot automatically.
SERVER_SLOTS = 2
SERVER_CTX_PER_SLOT = 4096

DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1

# ---- Condensing / chunking (token-based, tied to the LLM's tokenizer) ----
CONDENSE_CHUNK_SIZE = 1024
CONDENSE_CHUNK_OVERLAP = 100

# Each chunk is condensed to a word budget worked out from the target size
# (CONDENSE_TARGET_TOKENS, set below), so short documents are compressed gently
# and long ones harder.
CONDENSE_MIN_WORDS_PER_CHUNK = 40
CONDENSE_MAX_WORDS_PER_CHUNK = 320
CONDENSE_MAX_PASSES = 4

# ---- Generation sizing ----
# Large requests are split into several calls of at most this many output
# tokens, so any number of flashcards or questions fits the context window.
OUTPUT_TOKENS_PER_CALL = 1400
FLASHCARDS_MAX_PER_CALL = 8
# Flashcards are written from the original text, a few per section of the document.
FLASHCARDS_PER_SECTION = 4
QUIZ_MAX_PER_CALL = 6

# ---- RAG (character-based, tuned for retrieval rather than generation) ----
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RAG_CHUNK_SIZE = 800
RAG_CHUNK_OVERLAP = 150
RAG_TOP_K = 4
RAG_MAX_NEW_TOKENS = 220
# 1.0 = off. A penalty above 1 discourages repeating words from the context,
# which hurts factual answers that must reuse the source's terminology.
RAG_REPETITION_PENALTY = 1.0

# ---- Retry behavior for structured (JSON) LLM outputs ----
JSON_GENERATION_RETRIES = 1

# ---- Size of the condensed text ----
# The condensed text, the prompt wrapped around it, and the answer all have to
# fit in one slot's context window. This takes everything that is left after
# room for a full-size answer and the prompt template, so the condensed text
# grows automatically when you raise SERVER_CTX_PER_SLOT:
#     2 slots x 4096  -> about 2,400 tokens   (fast: two chunks condensed at once)
#     1 slot  x 6144  -> about 4,400 tokens   (more detail, slower condensing)
#     1 slot  x 8192  -> about 6,500 tokens   (most detail, slowest)
# A bigger target keeps more of the document for summaries, quizzes and mindmaps.
CONDENSE_TARGET_TOKENS = SERVER_CTX_PER_SLOT - OUTPUT_TOKENS_PER_CALL - 300
