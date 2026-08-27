"""
Unified Hybrid Combiner:
Merges traditional RAG and web RAG chunks into a single pool,
then runs them through a unified filtering pipeline:
  Combined Pool -> Embedding Filter (Top-5) -> Cross-Encoder Rerank (Top-3)
  
Returns all Top-3 chunks (not just top-1) to the LLM for richer context.
"""
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from reranker import filter_embedding_top_k, rerank_cross_encoder_top_k
from sentence_retriever import fine_grained_sentence_selection


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
    
    # Step 2: Embedding filter → Top-5
    top5_texts, top5_scores = filter_embedding_top_k(
        query=question,
        chunks=all_texts,
        top_k=min(5, len(all_texts)),
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
    
    # Rebuild hybrid chunks with cross-encoder scores
    top_hybrid_final = []
    for text, ce_score in zip(top_ce_texts, top_ce_scores):
        for hc in top5_hybrid:
            if hc.text == text:
                hc.cross_encoder_score = ce_score
                top_hybrid_final.append(hc)
                break
    
    # Step 4: Concatenate all top-3 chunks into context
    context = "\n\n---\n\n".join([f"[{hc.source}]\n{hc.text}" for hc in top_hybrid_final])
    
    # Metadata
    meta = {
        "combined_pool_size": len(hybrid_chunks),
        "trad_rag_count": len(trad_chunks or []),
        "web_rag_count": len(web_chunks or []),
        "embedding_filter_count": len(top5_texts),
        "embedding_scores": top5_scores,
        "cross_encoder_final_count": len(top_ce_texts),
        "cross_encoder_scores": top_ce_scores,
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
