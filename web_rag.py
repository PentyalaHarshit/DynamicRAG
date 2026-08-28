"""
Improved Web RAG Flow
=====================

User Query
     │
     ▼
Intent Detection  (caller sets intent_type)
     │
     ▼  (CURRENT_FACT -> year embedded in all queries)
ReAct Agent  ->  Google / MCP Tool
     │
     ▼
Retrieve Top-10 Results
     │
     ▼
Fetch Web Pages + Extract Snippets
     │
     ▼
Build Flat Chunk Pool  (page chunks + result snippets)
     │
     ▼
Phase 1 — Embedding Filter  ->  Top-5
     │
     ▼
Phase 1 — QA Cross-Encoder  ->  Top-3
     │
     ▼
Answerability Agent  (entity-based: PERSON / DATE / NUMBER / LOCATION)
     │
     ├─────────────────────────┐
     │                         │
 Answer Found             Answer Missing
     │                         │
     ▼                         ▼
Phase 2 — DQN          Query Expansion
     │                    (4 targeted queries)
     ▼                         │
Fine-Grained               New Web Search
Sentence Selection              │
     │                    Phase 1 again -> Top-5 -> Top-3
     ▼                         │
  LLM Context           Answerability re-check
     │                         │
     ▼                         ▼
Crew Validator          Phase 2 — DQN (best available)
     │
     ▼
Return Result
"""
from typing import List, Optional, Tuple

import config
from agents.react_agent import run_react_search
from agents.search_tool import fetch_page_text, google_search, extract_chunks_from_page
from agents.crew_validator import analyze_and_validate
from reranker import funnel_phase1, funnel_phase2
from answerability_agent import check_answerability
from query_expander import expand_and_search
from adaptive_retriever import derive_retrieval_spec, RetrievalSpec


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_complete_sentence(text: str) -> str:
    """
    Ensures a chunk ends with a complete sentence (full stop, question mark, or exclamation).
    If the chunk doesn't end with punctuation, truncates to the last complete sentence.
    """
    text = text.strip()
    if not text:
        return text
    
    # If already ends with sentence-ending punctuation, return as-is
    if text[-1] in '.!?':
        return text
    
    # Find the last sentence-ending punctuation
    for i in range(len(text) - 1, -1, -1):
        if text[i] in '.!?':
            return text[:i + 1]
    
    # No punctuation found — add a period
    return text + '.'


def _chunk_text(text: str, chunk_size: int = 800) -> List[str]:
    """
    Splits plain text into chunk_size-word chunks, ensuring each chunk ends with a complete sentence.
    Discards near-empty tails.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i: i + chunk_size]).strip()
        if len(chunk) > 80:
            chunk = _ensure_complete_sentence(chunk)
            chunks.append(chunk)
    return chunks


from concurrent.futures import ThreadPoolExecutor, as_completed


def _build_chunk_pool(
    results,
    max_pool: int = 20,
    chunk_words: int = 500,
) -> Tuple[List[str], List[str]]:
    """
    Builds the chunk pool fed into Phase 1 (embedding filter).
    Uses ThreadPoolExecutor for fast parallel web page retrieval.

    chunk_words controls the size of each page-extracted chunk.
    It is derived adaptively by derive_retrieval_spec() based on
    the query intent and depth — not set by a fixed constant.

    Per result:
      1. Snippet added first.
      2. Full-page article chunks retrieved in parallel across threads.
    """
    all_chunks: List[str] = []
    all_sources: List[str] = []

    # 1. Add all search engine snippets first (instant & reliable)
    for result in results:
        snippet = (result.snippet or "").strip()
        if snippet:
            chunk_text = f"{result.title}: {snippet}" if result.title else snippet
            if len(chunk_text.split()) >= 4:
                all_chunks.append(chunk_text)
                all_sources.append(result.link)

    # 2. Fetch full article pages concurrently in parallel
    # max_chunks per page scales with chunk_words: larger chunks -> fewer per page
    max_chunks_per_page = max(2, min(5, 1200 // max(chunk_words, 1)))
    urls_to_fetch = [r.link for r in results[:4]]

    def _fetch_one(url: str):
        try:
            return url, extract_chunks_from_page(
                url, chunk_words=chunk_words, max_chunks=max_chunks_per_page
            )
        except Exception:
            return url, []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_fetch_one, url) for url in urls_to_fetch]
        for future in as_completed(futures):
            url, page_chunks = future.result()
            for c in page_chunks:
                if len(all_chunks) >= max_pool:
                    break
                all_chunks.append(c)
                all_sources.append(url)

    return all_chunks[:max_pool], all_sources[:max_pool]


def _run_phase1_phase2(
    question: str,
    chunks: List[str],
    sources: List[str],
    intent_type: str,
) -> dict:
    """
    Runs Phase 1 (embedding -> Top-5 -> cross-encoder -> Top-3) then
    Phase 2 (DQN + sentence selection) on the given chunk pool.

    Returns a result dict with answer_found, context, sentences, score, meta.
    """
    if not chunks:
        return {
            "answer_found": False,
            "reason": "Empty chunk pool.",
            "best_chunk": "",
            "top_sentences": [],
            "rerank_score": 0.0,
            "funnel_meta": {},
        }

    # ── Phase 1: Top-20 pool -> Top-5 (embedding) -> Top-3 (cross-encoder) ──
    top3_chunks, top3_emb, top3_ce, p1_meta = funnel_phase1(
        query=question,
        chunks=chunks,
        sources=sources,
        top_emb=5,
        top_ce=3,
    )

    print(
        f"[Web RAG] Phase 1 complete: "
        f"pool={len(chunks)} -> emb_top5 -> ce_top3={len(top3_chunks)} | "
        f"CE scores: {[round(s, 2) for s in top3_ce]}"
    )

    # ── Get corresponding sources for top-3 chunks ──
    top3_sources = []
    for chunk in top3_chunks:
        for i, c in enumerate(chunks):
            if c == chunk:
                top3_sources.append(sources[i] if i < len(sources) else "web")
                break
    
    # ── Store all chunks pool info in meta for hybrid RAG ──
    p1_meta["_all_chunks"] = chunks
    p1_meta["_all_sources"] = sources

    # ── Answerability Agent: entity check on Top-3 ──
    answer_found, answerability_reason = check_answerability(question, top3_chunks)

    if not answer_found:
        return {
            "answer_found": False,
            "reason": answerability_reason,
            "best_chunk": "",
            "top_sentences": [],
            "rerank_score": 0.0,
            "funnel_meta": p1_meta,
            # Carry Top-3 forward so the caller can decide whether to merge
            "_top3_chunks": top3_chunks,
            "_top3_sources": top3_sources,
            "_top3_emb":    top3_emb,
            "_top3_ce":     top3_ce,
        }

    # ── Phase 2: DQN selection + fine-grained sentence extraction ──
    final_context, top_sentences, score, full_meta = funnel_phase2(
        query=question,
        top3_chunks=top3_chunks,
        top3_emb_scores=top3_emb,
        top3_ce_scores=top3_ce,
        sources=top3_sources,
        intent_type=intent_type,
        phase1_meta=p1_meta,
    )

    return {
        "answer_found":  True,
        "reason":        "Entity found in Top-3 chunks.",
        "best_chunk":    final_context,
        "top_sentences": top_sentences,
        "rerank_score":  score,
        "funnel_meta":   full_meta,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_web_rag(
    question: str,
    intent_type: str = "FACTOID",
    answer_style=None,
    operation_pattern: Optional[str] = None,
) -> dict:
    """
    Full Web RAG pipeline with adaptive retrieval granularity.

    Args:
        question:          The original user question.
        intent_type:       From intent_detector (CURRENT_FACT embeds year in queries).
        answer_style:      AnswerStyle dataclass from detect_answer_style() — controls depth.
        operation_pattern: Operation pattern from problem_analyzer (e.g. "DEFINITION").

    Returns a dict with keys:
        success, best_chunk, top_sentences, rerank_score, funnel_meta,
        validation, thoughts, queries_used,
        answer_found, answerability_reason, query_expansion_triggered.
    """

    # -- Step 0: Derive adaptive retrieval granularity from query meaning --
    spec: RetrievalSpec = derive_retrieval_spec(
        question=question,
        answer_style=answer_style,
        operation_pattern=operation_pattern,
    )
    print(
        f"[Web RAG] Retrieval granularity: {spec.granularity} "
        f"(chunk_words={spec.chunk_words}, max_chunks={spec.max_chunks})"
    )

    # -- Step 1: ReAct agent -> formulate search query -> fire search --
    trace = run_react_search(question, intent_type=intent_type)

    if not trace.results:
        return {
            "success":                   False,
            "reason":                    "No search results returned by ReAct agent.",
            "chunks":                    [],
            "generation_blocked":        True,
            "answer_found":              False,
            "best_chunk":                "",
            "top_sentences":             [],
            "rerank_score":              0.0,
            "funnel_meta":               {"evidence_gate_passed": False},
            "thoughts":                  trace.thoughts,
            "queries_used":              trace.queries_used,
        }

    print(f"[Web RAG] ReAct returned {len(trace.results)} results for query: '{question}'")

    # -- Step 2: Fetch pages + build chunk pool with adaptive chunk_words --
    chunks, sources = _build_chunk_pool(
        trace.results, max_pool=20, chunk_words=spec.chunk_words
    )

    if not chunks:
        return {
            "success":                   False,
            "reason":                    "Could not extract any text from search results.",
            "chunks":                    [],
            "generation_blocked":        True,
            "answer_found":              False,
            "best_chunk":                "",
            "top_sentences":             [],
            "rerank_score":              0.0,
            "funnel_meta":               {"evidence_gate_passed": False},
            "thoughts":                  trace.thoughts,
            "queries_used":              trace.queries_used,
        }

    print(f"[Web RAG] Chunk pool built: {len(chunks)} chunks from {len(trace.results)} results")

    # ── Step 3: Phase 1 -> Answerability -> Phase 2 (first pass) ──
    run1 = _run_phase1_phase2(question, chunks, sources, intent_type)

    query_expansion_triggered = False
    answer_found = run1["answer_found"]
    answerability_reason = run1.get("reason", "")

    if answer_found:
        final_context  = run1["best_chunk"]
        top_sentences  = run1["top_sentences"]
        score          = run1["rerank_score"]
        funnel_meta    = run1["funnel_meta"]

    else:
        # ── Step 4: Query Expansion -> new search -> re-run pipeline ──
        print(
            f"[Answerability Agent] Answer MISSING ({answerability_reason}). "
            f"Triggering Query Expansion..."
        )
        query_expansion_triggered = True

        new_snippets = expand_and_search(question, intent_type=intent_type, max_new_chunks=10)

        if new_snippets:
            exp_sources = ["expanded_search"] * len(new_snippets)
            run2 = _run_phase1_phase2(question, new_snippets, exp_sources, intent_type)

            if run2["answer_found"]:
                print("[Answerability Agent] Answer FOUND after Query Expansion.")
                answer_found      = True
                answerability_reason = run2["reason"]
                final_context     = run2["best_chunk"]
                top_sentences     = run2["top_sentences"]
                score             = run2["rerank_score"]
                funnel_meta       = run2["funnel_meta"]

            else:
                # Both passes failed — no chunk in either run contains the required
                # answer entity.  Do NOT call funnel_phase2 or produce context;
                # the LLM must not be allowed to hallucinate from empty evidence.
                print(
                    "[Answerability Agent] Answer still not found after Query Expansion. "
                    "Both passes exhausted. Returning structured failure — "
                    "LLM generation is BLOCKED."
                )
                answerability_reason = run2.get("reason", answerability_reason)
                # Merge chunk pools from both passes for hybrid combine to access
                all_chunks = run1.get("_top3_chunks", []) + run2.get("_top3_chunks", [])
                all_sources = run1.get("_top3_sources", []) + run2.get("_top3_sources", [])
                funnel_meta = run2.get("funnel_meta", run1.get("funnel_meta", {}))
                if "_all_chunks" not in funnel_meta and run2.get("_top3_chunks"):
                    funnel_meta["_all_chunks"] = run2.get("_top3_chunks", [])
                    funnel_meta["_all_sources"] = run2.get("_top3_sources", [])
                return {
                    "success":                   False,
                    "best_chunk":                "",
                    "top_sentences":             [],
                    "rerank_score":              0.0,
                    "funnel_meta":               funnel_meta,
                    "validation":                {"relevant": False, "valid": False},
                    "thoughts":                  trace.thoughts,
                    "queries_used":              trace.queries_used,
                    "answer_found":              False,
                    "answerability_reason":      answerability_reason,
                    "query_expansion_triggered": True,
                    "generation_blocked":        True,
                    "reason": (
                        "Evidence gate failed: required answer entity not found in any "
                        "retrieved chunk after initial search and query expansion. "
                        f"Detail: {answerability_reason}"
                    ),
                    "_top3_chunks": all_chunks,
                    "_top3_sources": all_sources,
                }

        else:
            # expand_and_search returned nothing — no new evidence at all.
            # Do NOT fall through to funnel_phase2; return a clean failure.
            print(
                "[Query Expansion] No new snippets found. "
                "Returning structured failure — LLM generation is BLOCKED."
            )
            return {
                "success":                   False,
                "best_chunk":                "",
                "top_sentences":             [],
                "rerank_score":              0.0,
                "funnel_meta":               run1.get("funnel_meta", {}),
                "validation":                {"relevant": False, "valid": False},
                "thoughts":                  trace.thoughts,
                "queries_used":              trace.queries_used,
                "answer_found":              False,
                "answerability_reason":      answerability_reason,
                "query_expansion_triggered": True,
                "generation_blocked":        True,
                "reason": (
                    "Evidence gate failed: query expansion returned no new snippets. "
                    f"Detail: {answerability_reason}"
                ),
            }

    # ── Step 5: Relevance / credibility sanity check ──────────────────────
    # Skipped by default (SKIP_CREW_VALIDATOR=1) to save one LLM round-trip.
    # Re-enable by setting SKIP_CREW_VALIDATOR=0 in the environment.
    if config.SKIP_CREW_VALIDATOR:
        check = {"relevant": True, "valid": True, "raw": ["crew validator skipped"]}
    else:
        check = analyze_and_validate(question, final_context)

    return {
        "success":                   check["relevant"] and check["valid"],
        "best_chunk":                final_context,
        "top_sentences":             top_sentences,
        "rerank_score":              score,
        "funnel_meta":               funnel_meta,
        "validation":                check,
        "thoughts":                  trace.thoughts,
        "queries_used":              trace.queries_used,
        "answer_found":              answer_found,
        "answerability_reason":      answerability_reason,
        "query_expansion_triggered": query_expansion_triggered,
        "_top3_chunks":              funnel_meta.get("_all_chunks", []),
        "_top3_sources":             funnel_meta.get("_all_sources", []),
    }
