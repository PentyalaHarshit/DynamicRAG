"""
Central config. Fill in API keys via environment variables, don't hardcode them.
"""
import os

# --- Vector store (traditional RAG) ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Confidence threshold: if the best traditional-RAG chunk scores below this,
# we treat the KB as "doesn't have it" and fall back to web RAG.
TRADITIONAL_RAG_CONFIDENCE_THRESHOLD = float(os.getenv("TRAD_RAG_THRESHOLD", "0.72"))

# --- Google Custom Search (web RAG retrieval) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")   # Custom Search Engine ID
GOOGLE_SEARCH_NUM_RESULTS = 10   # Improved Web RAG: Top-10 pool for embedding filter

# --- Reranker ---
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Generation (local, free) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

def _get_default_ollama_model() -> str:
    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model
    try:
        import requests
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            names = [m.get("name", "") for m in models]
            preferred = ["phi3", "phi3:mini", "llama3.2", "llama3", "mistral", "qwen2.5:3b", "qwen2.5:7b"]
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

# --- Verification / self-correction loop ---
MAX_SELF_CORRECTION_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # NOT 1000 - see README
VERIFIER_PASS_THRESHOLD = float(os.getenv("VERIFIER_PASS_THRESHOLD", "0.75"))

# --- Episode logging (for future offline RL / fine-tuning) ---
EPISODE_LOG_PATH = os.getenv("EPISODE_LOG_PATH", "./data/episodes.jsonl")
