"""
Multi-Stage Hierarchical Funnel Module with Fine-Grained Sentence Selection.

Architecture (two-phase design with Answerability Gate):

Phase 1 — Retrieval + Ranking (runs before Answerability Agent):
  Input: Top-20 candidate chunks
  Step 1: Embedding Similarity  ->  Top-5 chunks
  Step 2: QA Cross-Encoder      ->  Top-3 chunks
  Output: (top3_chunks, top3_emb_scores, top3_ce_scores)

  <- Answerability Agent runs here on Top-3 ->

Phase 2 — Selection + Extraction (runs only after answer is confirmed):
  Step 3: Rich DQN Selector (10D state, two-gate architecture) -> Top-1 chunk
  Step 4: Fine-Grained Sentence Selection -> Best 2-3 sentences
  Output: (context, sentences, score, meta)

multi_stage_funnel() = Phase1 + Answerability Gate + Phase2 combined.
  Used by traditional RAG path.
  Returns early with empty context + answerability_failed=True if gate fails.
"""
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder, util

import config
from dqn_selector import select_top1_rich_dqn, RichDQNResult
from sentence_retriever import fine_grained_sentence_selection
from answerability_agent import check_answerability

_embedding_model = None
_cross_encoder_model = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder_model
    if _cross_encoder_model is None:
        _cross_encoder_model = CrossEncoder(config.RERANKER_MODEL)
    return _cross_encoder_model


# ---------------------------------------------------------------------------
# Individual stage functions (public, used by web_rag.py)
# ---------------------------------------------------------------------------

def filter_embedding_top_k(
    query: str,
    chunks: List[str],
    top_k: int = 5,
) -> Tuple[List[str], List[float]]:
    """Stage 1: Embedding cosine similarity -> Top-k chunks."""
    if not chunks:
        return [], []
    model = _get_embedding_model()
    query_emb = model.encode(query, convert_to_tensor=True)
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, chunk_embs)[0].tolist()
    scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top_pairs = scored[:min(top_k, len(scored))]
    return [c for c, _ in top_pairs], [float(s) for _, s in top_pairs]


def rerank_cross_encoder_top_k(
    query: str,
    chunks: List[str],
    top_k: int = 3,
) -> Tuple[List[str], List[float]]:
    """Stage 2: QA-intent cross-encoder reranking -> Top-k chunks."""
    if not chunks:
        return [], []
    ce = _get_cross_encoder()
    pairs = [(f"Question: {query}", f"Candidate Answer: {c}") for c in chunks]
    scores = ce.predict(pairs)
    scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top_pairs = scored[:min(top_k, len(scored))]
    return [c for c, _ in top_pairs], [float(s) for _, s in top_pairs]


# ---------------------------------------------------------------------------
# Phase 1: Top-20 -> embedding Top-5 -> cross-encoder Top-3
# ---------------------------------------------------------------------------

def funnel_phase1(
    query: str,
    chunks: List[str],
    sources: Optional[List[str]] = None,
    top_emb: int = 5,
    top_ce: int = 3,
) -> Tuple[List[str], List[float], List[float], Dict[str, Any]]:
    """
    Phase 1 of the funnel: retrieval + embedding filter + cross-encoder reranking.

    Returns:
        top3_chunks       - Top-3 chunks after cross-encoder (best first)
        top3_emb_scores   - Embedding similarity scores for the top-3 chunks
        top3_ce_scores    - Cross-encoder scores for the top-3 chunks
        phase1_meta       - Dict with intermediate score arrays for logging
    """
    if not chunks:
        raise ValueError("No chunks provided to funnel_phase1.")

    sources = sources or ["web"] * len(chunks)

    # Stage 1: embedding filter -> Top-5
    top5_chunks, emb_scores = filter_embedding_top_k(query, chunks, top_k=top_emb)

    # Stage 2: cross-encoder rerank -> Top-3
    top3_chunks, ce_scores = rerank_cross_encoder_top_k(query, top5_chunks, top_k=top_ce)

    # Carry embedding scores forward to top-3
    top3_emb_scores = []
    for c in top3_chunks:
        if c in top5_chunks:
            idx = top5_chunks.index(c)
            top3_emb_scores.append(emb_scores[idx])
        else:
            top3_emb_scores.append(0.5)

    phase1_meta = {
        "initial_chunks_count":    len(chunks),
        "top5_embedding_scores":   [round(s, 3) for s in emb_scores],
        "top3_cross_encoder_scores": [round(s, 3) for s in ce_scores],
    }
    return top3_chunks, top3_emb_scores, ce_scores, phase1_meta


# ---------------------------------------------------------------------------
# Phase 2: confirmed Top-3 -> DQN (7D) -> sentence retrieval
# ---------------------------------------------------------------------------

def funnel_phase2(
    query: str,
    top3_chunks: List[str],
    top3_emb_scores: List[float],
    top3_ce_scores: List[float],
    sources: Optional[List[str]] = None,
    intent_type: str = "FACTOID",
    phase1_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, List[str], float, Dict[str, Any]]:
    """
    Phase 2 of the funnel: DQN selection + fine-grained sentence extraction.
    Only called after Answerability Agent confirms the answer exists.

    Returns:
        final_context    - Joined selected sentences (used as LLM context)
        top_sentences    - List of selected sentences
        confidence       - DQN confidence score
        meta             - Full metadata dict including DQN state vectors
    """
    sources = sources or ["web"] * len(top3_chunks)

    # Stage 3: Rich DQN Selector (9D state, two-gate architecture) -> Top-1 chunk
    dqn_res: RichDQNResult = select_top1_rich_dqn(
        query=query,
        chunks=top3_chunks,
        embedding_scores=top3_emb_scores,
        cross_encoder_scores=top3_ce_scores,
        sources=sources,
        intent_type=intent_type,
    )
    top1_chunk = dqn_res.selected_chunk

    # Stage 4: Fine-Grained Sentence Selection
    selected_sentences, sentence_scores_list, best_sent_score = fine_grained_sentence_selection(
        query, top1_chunk, top_k=3
    )
    final_context = " ".join(selected_sentences)

    meta = {
        **(phase1_meta or {}),
        "dqn_selected_index":   dqn_res.selected_index,
        "dqn_q_values":         dqn_res.q_values,
        "dqn_rich_states":      dqn_res.rich_states,
        "evidence_gate_passed": dqn_res.evidence_gate_passed,
        "topic_gate_passed":    dqn_res.topic_gate_passed,
        "sentence_scores_list": sentence_scores_list,
        "best_sentence_score":  best_sent_score,
    }
    return final_context, selected_sentences, dqn_res.confidence, meta


# ---------------------------------------------------------------------------
# Combined wrapper: used by traditional RAG path (no answerability split needed)
# ---------------------------------------------------------------------------

def multi_stage_funnel(
    query: str,
    chunks: List[str],
    sources: Optional[List[str]] = None,
    intent_type: str = "FACTOID",
) -> Tuple[str, List[str], float, Dict[str, Any]]:
    """
    Full funnel: Phase 1 + Answerability Gate + Phase 2 combined.
    Used by the traditional RAG path (vector DB pre-confirmed relevance).

    If the Answerability Agent determines no chunk contains the required
    answer entity, returns an empty context with the gate failure signalled
    in the meta dict:
        meta["evidence_gate_passed"] = False
        meta["answerability_failed"] = True
        meta["answerability_reason"] = <reason string>

    Callers (main.py) should inspect these flags and escalate to web RAG.
    """
    top3_chunks, top3_emb, top3_ce, p1_meta = funnel_phase1(query, chunks, sources)

    # ── Answerability Gate: entity check before DQN / sentence selection ──────────
    answer_found, answerability_reason = check_answerability(query, top3_chunks)
    if not answer_found:
        print(
            f"[Multi-Stage Funnel] Answerability Agent: FAILED. "
            f"No required entity found in Top-3 chunks. "
            f"Reason: {answerability_reason}. "
            f"Returning empty context — caller should escalate to Web RAG."
        )
        failure_meta = {
            **p1_meta,
            "evidence_gate_passed": False,
            "answerability_failed": True,
            "answerability_reason": answerability_reason,
            "dqn_selected_index":   0,
            "dqn_q_values":         [],
            "dqn_rich_states":      [],
            "topic_gate_passed":    False,
            "sentence_scores_list": [],
            "best_sentence_score":  0.0,
        }
        return "", [], 0.0, failure_meta

    return funnel_phase2(
        query, top3_chunks, top3_emb, top3_ce,
        sources=sources, intent_type=intent_type, phase1_meta=p1_meta
    )


def rerank_top1(query: str, chunks: List[str]) -> Tuple[str, float]:
    final_context, _, conf, _ = multi_stage_funnel(query, chunks)
    return final_context, conf
