"""
Central config. Fill in API keys via environment variables, don't hardcode them.
"""
import os

# --- Vector store (traditional RAG) ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

# Three separate Chroma collections — never mix them.
#
#   knowledge_base     — ingested documents (textbooks, articles, KB docs).
#                        Stable, human-curated. Written at ingest time only.
#
#   qa_memory          — verified (question, answer) pairs written back after
#                        a successful pipeline run.  Queried first by the router
#                        before touching the KB or web.
#
#   conversation_memory — per-session turn history (user + assistant turns).
#                        Not yet used for retrieval; stored for future context
#                        window reconstruction and RL replay.
#
# Legacy alias: CHROMA_COLLECTION_NAME still maps to knowledge_base so that
# any existing ingestion scripts continue to work unchanged.
CHROMA_COLLECTION_NAME          = os.getenv("CHROMA_COLLECTION_NAME",     "knowledge_base")
CHROMA_KB_COLLECTION            = os.getenv("CHROMA_KB_COLLECTION",       "knowledge_base")
CHROMA_QA_MEMORY_COLLECTION     = os.getenv("CHROMA_QA_MEMORY_COLLECTION","qa_memory")
CHROMA_CONV_MEMORY_COLLECTION   = os.getenv("CHROMA_CONV_MEMORY_COLLECTION","conversation_memory")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Confidence threshold: if the best traditional-RAG chunk scores below this,
# we treat the KB as "doesn't have it" and fall back to web RAG.
TRADITIONAL_RAG_CONFIDENCE_THRESHOLD = float(os.getenv("TRAD_RAG_THRESHOLD", "0.72"))

# --- Google Custom Search (web RAG retrieval) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
GOOGLE_SEARCH_NUM_RESULTS = 10
FAST_MODE = os.getenv("FAST_MODE", "1").lower() not in {"0", "false", "no"}

# --- Reranker ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------------------
# LLM backend — Groq (primary, fast) + Ollama (local fallback)
# ---------------------------------------------------------------------------

# ── Groq API ────────────────────────────────────────────────────────────────
# Get a free key at https://console.groq.com  (14,400 req/day, ~500 tok/s)
# Set via environment:  $env:GROQ_API_KEY="gsk_..."
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Best free-tier model on Groq:
#   llama-3.1-8b-instant  — ~500 tok/s, great for factoid + biography queries
#   llama-3.3-70b-versatile — slower but stronger reasoning (use for MATH/CODING)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Set to "0" to force Ollama even if GROQ_API_KEY is set (e.g. for offline use)
USE_GROQ = os.getenv("USE_GROQ", "1").lower() not in {"0", "false", "no"}

# ── Ollama (local fallback) ──────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 15 s fast-fail: when Groq is primary, Ollama is rarely reached.
# If you are running Ollama-only (no Groq key), raise this via the env var.
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "15"))


def _get_default_ollama_model() -> str:
    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model
    try:
        import requests as _req
        resp = _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            names = [m.get("name", "") for m in models]
            preferred = [
                "phi3", "phi3:mini", "llama3.2", "llama3",
                "mistral", "qwen2.5:3b", "qwen2.5:7b",
            ]
            for pref in preferred:
                for name in names:
                    if pref in name:
                        return name
            if names:
                return names[0]
    except Exception:
        pass
    return "phi3"


OLLAMA_MODEL = _get_default_ollama_model()

# ---------------------------------------------------------------------------
# Generation quality / speed controls
# ---------------------------------------------------------------------------

# 1 attempt = one generate + one verify.
# More retries only help when the LLM is slow enough that the second try
# meaningfully differs; with Groq at ~500 tok/s the first attempt is
# almost always good.  Set to 2 if you want one correction pass.
MAX_SELF_CORRECTION_RETRIES = int(os.getenv("MAX_RETRIES", "1"))

MAX_REASONING_ATTEMPTS = int(os.getenv("MAX_REASONING_ATTEMPTS", "2"))
VERIFIER_PASS_THRESHOLD = float(os.getenv("VERIFIER_PASS_THRESHOLD", "0.75"))

# Skip the crew_validator LLM call (analyze_and_validate) at the end of
# web_rag.  It costs one full LLM round-trip and almost never changes the
# routing decision.  Set to "1" to re-enable.
SKIP_CREW_VALIDATOR = os.getenv("SKIP_CREW_VALIDATOR", "1").lower() not in {"0", "false", "no"}

# --- Episode logging ---
EPISODE_LOG_PATH = os.getenv("EPISODE_LOG_PATH", "./data/episodes.jsonl")
