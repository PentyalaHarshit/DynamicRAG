"""
Process-level model singleton cache.

All modules that need the embedding model or cross-encoder must import
from here instead of constructing their own instances.  This guarantees
the models are loaded exactly once per Python process, no matter how many
modules call get_embedding_model() or get_cross_encoder().

Previous state: reranker.py had _embedding_model + _cross_encoder_model,
sentence_retriever.py had a separate _embedder — the log showed
"Loading weights: 100%" appearing 3-4 times per query.

After this change: one load, shared everywhere.
"""
from __future__ import annotations
from typing import Optional

_embedding_model = None
_cross_encoder_model = None


def get_embedding_model():
    """Return the process-level SentenceTransformer singleton."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        import config
        print(f"[ModelCache] Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


def get_cross_encoder():
    """Return the process-level CrossEncoder singleton."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        from sentence_transformers import CrossEncoder
        import config
        print(f"[ModelCache] Loading cross-encoder: {config.RERANKER_MODEL}")
        _cross_encoder_model = CrossEncoder(config.RERANKER_MODEL)
    return _cross_encoder_model
