"""
Unified Hybrid Combiner:
Merges traditional RAG and web RAG chunks into a single pool,
then runs them through a unified filtering pipeline:
  Combined Pool -> Embedding Filter (Top-5) -> Cross-Encoder Rerank (Top-3)

Guards added:
  1. Minimum CE score gate: if best cross-encoder score < 0.0 the chunks are
     irrelevant to the query — evidence_gate_passed=False is returned so the
     graph escalates to web RAG rather than serving the wrong answer.
  2. Query-subject term check: at least one top-3 chunk must contain an
     important term from the query (e.g. "tesla", "edison"). Catches cases
     where the vector store returns a semantically similar but topically wrong
     chunk (e.g. returning a Tesla chunk for an Edison query).

Returns all Top-3 chunks (not just top-1) to the LLM for richer context.
"""
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from reranker import filter_embedding_top_k, rerank_cross_encoder_top_k
from sentence_retriever import fine_grained_sentence_selection
from generator import strip_retrieval_chrome
from answerability_agent import _important_question_terms

# Cross-encoder scores below this threshold indicate the chunk is NOT
# relevant to the query. Negative CE scores are a reliable signal of mismatch.
_MIN_CE_SCORE = 0.0


@dataclass
class HybridChunk:
    """Represents a chunk with source tracking."""
    text: str
    source: str  # "trad_rag", "web_rag", "memory"
    embedding_score: float = 0.0
    cross_encoder_score: float = 0.0


def combine_chunks(
    trad_chunks: List[str],
    trad_sources: List[str],
    web_chunks: List[str],
    web_sources: List[str],
) -> Tuple[List[HybridChunk], List[str]]:
    """
    Combines traditional RAG and web RAG chunks into a single unified pool.
    Tracks source for each chunk for debugging and auditing.
    """
    hybrid_chunks: List[HybridChunk] = []
    all_texts: List[str] = []
    
    # Add traditional RAG chunks
    for chunk, source in zip(trad_chunks or [], trad_sources or []):
        hybrid_chunks.append(HybridChunk(text=chunk, source=f"trad_rag:{source}"))
        all_texts.append(chunk)
    
    # Add web RAG chunks
    for chunk, source in zip(web_chunks or [], web_sources or []):
        hybrid_chunks.append(HybridChunk(text=chunk, source=f"web_rag:{source}"))
        all_texts.append(chunk)
    
    return hybrid_chunks, all_texts


def unified_hybrid_funnel(
    question: str,
    trad_chunks: List[str],
    trad_sources: List[str],
    web_chunks: List[str],
    web_sources: List[str],
    top_final: int = 3,
) -> Tuple[List[HybridChunk], List[str], Dict[str, Any]]:
    """
    Unified funnel that combines traditional and web RAG chunks,
    then filters through embedding and cross-encoder to get top-3.
    
    Returns:
      - top_hybrid_chunks: List of HybridChunk objects (top-3)
      - context: Concatenated text of all top-3 chunks
      - meta: Metadata about the filtering process
    """
    # Step 1: Combine chunks
    hybrid_chunks, all_texts = combine_chunks(
        trad_chunks, trad_sources, web_chunks, web_sources
    )
    
    if not hybrid_chunks:
        return [], "", {
            "combined_pool_size": 0,
            "embedding_filter_count": 0,
            "cross_encoder_final_count": 0,
            "sources_used": [],
        }
    
    # Step 2: Embedding filter → Top-K (scales with top_final)
    emb_top_k = min(max(top_final + 3, 5), len(all_texts))
    top5_texts, top5_scores = filter_embedding_top_k(
        query=question,
        chunks=all_texts,
        top_k=emb_top_k,
    )
    
    # Rebuild hybrid chunks list with embedding scores
    top5_hybrid = []
    for text, score in zip(top5_texts, top5_scores):
        for hc in hybrid_chunks:
            if hc.text == text:
                hc.embedding_score = score
                top5_hybrid.append(hc)
                break
    
    # Step 3: Cross-encoder rerank → Top-3 (or top_final)
    top_ce_texts, top_ce_scores = rerank_cross_encoder_top_k(
        query=question,
        chunks=top5_texts,
        top_k=min(top_final, len(top5_texts)),
    )

    # ── Guard 1: Minimum CE score gate ──────────────────────────────────────
    # A negative cross-encoder score means the chunk is NOT relevant to the
    # query. If every reranked chunk scores below the threshold, block
    # generation so the graph escalates to web RAG.
    best_ce_score = max(top_ce_scores) if top_ce_scores else -999.0
    if best_ce_score < _MIN_CE_SCORE:
        print(
            f"[HybridCombiner] CE gate FAIL: best score {best_ce_score:.3f} < "
            f"{_MIN_CE_SCORE} — chunks irrelevant to query, blocking generation."
        )
        meta = {
            "combined_pool_size": len(hybrid_chunks),
            "trad_rag_count": len(trad_chunks or []),
            "web_rag_count": len(web_chunks or []),
            "embedding_filter_count": len(top5_texts),
            "embedding_scores": top5_scores,
            "cross_encoder_final_count": len(top_ce_texts),
            "cross_encoder_scores": top_ce_scores,
            "sources_used": [],
            "hybrid_chunks": [],
            "evidence_gate_passed": False,
            "gate_failure_reason": f"Best CE score {best_ce_score:.3f} below threshold {_MIN_CE_SCORE}",
        }
        return [], "", meta

    # ── Guard 2: Query-subject term check ────────────────────────────────────
    # At least one top chunk must mention an important term from the query
    # (e.g. "edison", "tesla"). This catches subject-entity mismatches where
    # a semantically similar but topically wrong chunk slips through.
    query_terms = _important_question_terms(question)
    subject_covered = False
    if query_terms:
        for text in top_ce_texts:
            text_lower = text.lower()
            if any(t in text_lower for t in query_terms):
                subject_covered = True
                break
    else:
        subject_covered = True  # No specific terms → can't filter

    if not subject_covered:
        print(
            f"[HybridCombiner] Subject gate FAIL: none of the top chunks "
            f"mention query terms {query_terms} — blocking generation."
        )
        meta = {
            "combined_pool_size": len(hybrid_chunks),
            "trad_rag_count": len(trad_chunks or []),
            "web_rag_count": len(web_chunks or []),
            "embedding_filter_count": len(top5_texts),
            "embedding_scores": top5_scores,
            "cross_encoder_final_count": len(top_ce_texts),
            "cross_encoder_scores": top_ce_scores,
            "sources_used": [],
            "hybrid_chunks": [],
            "evidence_gate_passed": False,
            "gate_failure_reason": f"No chunk covers query subject terms: {query_terms}",
        }
        return [], "", meta

    # Rebuild hybrid chunks with cross-encoder scores
    top_hybrid_final = []
    for text, ce_score in zip(top_ce_texts, top_ce_scores):
        for hc in top5_hybrid:
            if hc.text == text:
                hc.cross_encoder_score = ce_score
                top_hybrid_final.append(hc)
                break

    # Step 4: Concatenate top chunks as clean prose (no source tags or Wikipedia labels)
    context_parts = []
    seen = set()
    for hc in top_hybrid_final:
        cleaned = strip_retrieval_chrome(hc.text)
        if not cleaned:
            continue
        key = cleaned[:180].lower()
        if key in seen:
            continue
        seen.add(key)
        context_parts.append(cleaned)
    context = "\n\n".join(context_parts)

    # Metadata
    meta = {
        "combined_pool_size": len(hybrid_chunks),
        "trad_rag_count": len(trad_chunks or []),
        "web_rag_count": len(web_chunks or []),
        "embedding_filter_count": len(top5_texts),
        "embedding_scores": top5_scores,
        "cross_encoder_final_count": len(top_ce_texts),
        "cross_encoder_scores": top_ce_scores,
        "dqn_selected_index": 0,
        "sources_used": [hc.source for hc in top_hybrid_final],
        "hybrid_chunks": [
            {
                "source": hc.source,
                "embedding_score": round(hc.embedding_score, 4),
                "cross_encoder_score": round(hc.cross_encoder_score, 4),
                "text_preview": hc.text[:100] + "..."
            }
            for hc in top_hybrid_final
        ],
        "evidence_gate_passed": True,
    }

    return top_hybrid_final, context, meta


def extract_top3_sentences(
    question: str,
    top_hybrid_chunks: List[HybridChunk],
) -> List[str]:
    """
    Extracts fine-grained sentences from the top-3 hybrid chunks.
    Uses the existing sentence_retriever for consistency.
    """
    if not top_hybrid_chunks:
        return []
    
    # Combine all top-3 chunks
    combined_text = " ".join([hc.text for hc in top_hybrid_chunks])
    
    # Extract top sentences (returns tuple: sentences, scores_dict, best_score)
    top_sentences, _, _ = fine_grained_sentence_selection(
        query=question,
        chunk=combined_text,
        top_k=5,  # Get up to 5 sentences from the combined top-3 chunks
    )
    
    return top_sentences
