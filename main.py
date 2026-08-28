"""
OmniAgentAI — Entry Point
==========================
Orchestration is now handled entirely by the LangGraph StateGraph
defined in graph.py.  This file provides:

  answer_question(question) — thin wrapper around run_pipeline()
                               kept for backward compatibility with
                               any callers that imported from main.py.

  __main__ block            — unchanged CLI interface; prints the full
                               structured execution report exactly as
                               before.

Architecture (see graph.py for the full node/edge diagram):

                               User Query
                                    │
                                    ▼
                          LangGraph StateGraph
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
               detect_intent             memory_check
                       └────────────┬────────────┘
                                    ▼
                               Router Agent
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
             traditional_rag                    web_rag
                     │                     (ReAct + MCP)
                     │   (escalate on           │
                     │    gate failure)          │
                     └──────────────┬────────────┘
                                    ▼
                             evidence_gate
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
                  generate                   no_evidence
                     │                             │
                     ▼                           END
              sac_reward
                     │
              memory_write
                     │
                    END
"""
import sys
import json

from graph import run_pipeline


# ---------------------------------------------------------------------------
# Public API — backward-compatible wrapper
# ---------------------------------------------------------------------------

def answer_question(question: str) -> dict:
    """
    Run the full hybrid RAG pipeline for *question* and return the result dict.

    This is now a thin wrapper around graph.run_pipeline().  The return value
    has exactly the same keys as the old hand-written implementation so any
    existing callers continue to work without changes.
    """
    return run_pipeline(question)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "Who invented quantum mechanics?"
        print(f"No prompt provided. Defaulting to sample query: '{q}'\n")

    out = answer_question(q)
    fmeta = out.get("funnel_meta", {})

    print("=" * 70)
    print("HIERARCHICAL MULTI-STAGE HYBRID RAG EXECUTION RESULTS")
    print("=" * 70)
    print(f"Query:                   {q}")

    intent_info = out.get("intent", {})

    # Direct-answer routes (currency, weather, finance, travel) skip RAG entirely
    if out.get("route") in {"currency", "weather", "finance", "travel"}:
        print(f"Routing Decision:        {out.get('route')}")
        if out.get("generation_blocked") or out.get("error"):
            print(f"\n--- Error ---\n  {out.get('error') or out.get('reason')}")
        else:
            print("\nFINAL ANSWER:")
            ans_out = out.get("final_answer") or "(no answer)"
            try:
                print(ans_out)
            except UnicodeEncodeError:
                print(ans_out.encode("ascii", "replace").decode("ascii"))
        print("=" * 70)
        import sys; sys.exit(0)
    # intent may be an IntentResult dataclass or a plain dict depending on
    # whether generation was reached or the no_evidence path fired.
    if hasattr(intent_info, "intent_type"):
        itype = intent_info.intent_type
        iconf = intent_info.confidence
        ireason = intent_info.reasoning
    else:
        itype   = intent_info.get("type")   if isinstance(intent_info, dict) else str(intent_info)
        iconf   = intent_info.get("confidence", "") if isinstance(intent_info, dict) else ""
        ireason = intent_info.get("reasoning", "")  if isinstance(intent_info, dict) else ""

    print(f"Intent Detection:        {itype} (Confidence: {iconf})")
    print(f"  Reasoning:             {ireason}")
    analysis = out.get("problem_analysis", {})
    strategy = out.get("strategy", {})
    if analysis:
        print(
            f"Problem Pattern:         {analysis.get('domain')}/"
            f"{analysis.get('subdomain')} -> {analysis.get('pattern')}"
        )
        print(f"Strategy Selection:      {strategy.get('strategy', 'unknown')}")
    # funnel_meta keys differ between the DQN path and the hybrid combiner path.
    # Read both and prefer whichever is populated.
    pool_size   = (fmeta.get("initial_chunks_count")
                   or fmeta.get("combined_pool_size", 0))
    emb_scores  = (fmeta.get("top5_embedding_scores")
                   or fmeta.get("embedding_scores"))
    ce_scores   = (fmeta.get("top3_cross_encoder_scores")
                   or fmeta.get("cross_encoder_scores"))
    dqn_idx     = fmeta.get("dqn_selected_index")
    gate_passed = fmeta.get("evidence_gate_passed")

    # Hybrid combiner path: show trad/web split
    trad_count = fmeta.get("trad_rag_count")
    web_count  = fmeta.get("web_rag_count")
    pool_label = (
        f"{pool_size} chunks  (trad={trad_count}, web={web_count})"
        if trad_count is not None
        else f"{pool_size} chunks"
    )

    print(f"Routing Decision:        {out.get('route')}")
    print(f"Initial Extracted Pool:  {pool_label}")
    print(f"Embedding Sim Filter:    Filtered to Top-5 (Scores: {emb_scores})")
    print(f"QA Cross-Encoder Rerank: Reranked to Top-3 (Scores: {ce_scores})")
    print(
        f"Rich DQN Selector:       Index {dqn_idx} selected "
        f"(Evidence Gate Passed: {gate_passed})"
    )

    print("\n--- Rich DQN State Vector (Chosen Chunk) ---")
    if fmeta.get("dqn_rich_states"):
        idx = fmeta.get("dqn_selected_index", 0)
        if isinstance(idx, int) and idx < len(fmeta["dqn_rich_states"]):
            print(json.dumps(fmeta["dqn_rich_states"][idx], indent=2))

    print("\n--- Fine-Grained Sentence Chunking & Similarity Scores ---")
    sent_scores = fmeta.get("sentence_scores_list", [])
    extracted   = out.get("extracted_sentences", [])
    for item in sent_scores[:5]:
        is_top = item.get("sentence") in extracted
        mark   = " [SELECTED]" if is_top else ""
        print(f"  Sentence {item.get('index')} -> {item.get('score')}{mark}")
        s_text = item.get("sentence", "")
        try:
            print(f'    "{s_text}"')
        except UnicodeEncodeError:
            print(f'    "{s_text.encode("ascii", "replace").decode("ascii")}"')

    print("\n--- Answerability Agent ---")
    print(f"  Answer Found:            {out.get('answer_found', True)}")
    print(f"  Query Expansion Run:     {out.get('query_expansion_triggered', False)}")
    ans_reason = out.get("answerability_reason", "")
    if ans_reason:
        print(f"  Reason:                  {ans_reason}")

    # generation_blocked path: no verification dimensions
    if out.get("generation_blocked"):
        print("\n--- Evidence Gate: BLOCKED ---")
        print(f"  Reason: {out.get('reason', '')}")
    else:
        print("\n--- 4D Verification Agent Dimensions ---")
        print(json.dumps(out.get("verification_dimensions", {}), indent=2))
        print("-" * 70)
        print(f"Verifier Composite Score: {out.get('final_score')}")
        print(f"SAC Policy Reward R(s,a): {out.get('sac_reward')}")

    print("FINAL ANSWER:")
    ans_out = out.get("final_answer") or out.get("error") or "(no answer)"
    try:
        print(ans_out)
    except UnicodeEncodeError:
        print(ans_out.encode("ascii", "replace").decode("ascii"))
    print("=" * 70)
