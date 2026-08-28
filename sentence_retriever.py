"""
Fine-Grained Chunking (Sentence/Paragraph Level) & Sentence Selection Module:
Splits Top-1 Chunk into individual sentences, computes exact semantic similarity scores
for each sentence against the query (e.g. Sentence 1 -> 0.82, Sentence 2 -> 0.95 ✅),
and extracts the best sentences to form high-density prompt context.
"""
import re
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer, util

import config
from model_cache import get_embedding_model

_embedder = None


def _get_embedder():
    return get_embedding_model()


def split_into_sentences(text: str) -> List[str]:
    """Fine-grained chunking into clean individual sentences."""
    from generator import split_clean_sentences

    sentences = [s for s in split_clean_sentences(text) if len(s) > 15]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


from answerability_agent import _expected_entity_type, _extract_entities


def fine_grained_sentence_selection(
    query: str,
    chunk: str,
    top_k: int = 3,
) -> Tuple[List[str], List[Dict[str, Any]], float]:
    """
    Fine-Grained Chunking & Selection:
    1. Splits chunk into sentences.
    2. Ranks each sentence against query with an entity-presence bonus.
    3. Returns (selected_top_sentences, all_sentence_scores, best_sentence_score).
    """
    sentences = split_into_sentences(chunk)
    if not sentences:
        return [chunk], [{"sentence": chunk, "score": 1.0}], 1.0

    embedder = _get_embedder()
    query_emb = embedder.encode(query, convert_to_tensor=True)
    sent_embs = embedder.encode(sentences, convert_to_tensor=True)

    base_scores = util.cos_sim(query_emb, sent_embs)[0].tolist()

    # Determine expected entity type for entity bonus
    expected_type = _expected_entity_type(query)

    final_sentence_scores = []
    for i, (s, sc) in enumerate(zip(sentences, base_scores)):
        score = float(sc)
        # Entity bonus: prioritize sentences containing the answer entity or key role markers
        if expected_type != "NONE" and _extract_entities(s, expected_type):
            score += 0.35
        elif "incumbent" in s.lower() or "current" in s.lower():
            score += 0.20

        final_sentence_scores.append({
            "index": i + 1,
            "sentence": s,
            "score": round(score, 3),
            "base_score": round(float(sc), 3),
        })

    # Sort sentences by score descending
    sorted_sentences = sorted(final_sentence_scores, key=lambda x: x["score"], reverse=True)
    top_items = sorted_sentences[:min(top_k, len(sorted_sentences))]

    # Maintain consecutive natural order among top items for coherent narrative
    top_items_in_order = sorted(top_items, key=lambda x: x["index"])

    top_sentences = [item["sentence"] for item in top_items_in_order]
    best_score = float(sorted_sentences[0]["score"]) if sorted_sentences else 0.0

    return top_sentences, final_sentence_scores, best_score


# Backward compatibility wrappers
def extract_consecutive_passage_window(query: str, chunk: str, window_size: int = 3):
    top_sents, _, _ = fine_grained_sentence_selection(query, chunk, top_k=window_size)
    return top_sents, [1.0] * len(top_sents)


def extract_top_sentences(query: str, chunk: str, top_k: int = 3):
    return extract_consecutive_passage_window(query, chunk, window_size=top_k)
