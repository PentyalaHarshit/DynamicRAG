"""
LangGraph StateGraph Orchestration for OmniAgentAI Hybrid RAG
=============================================================

Unified RAG Pipeline: Combines traditional RAG and web RAG chunks
into a single pool, then filters through: 
  Embedding (→Top-5) → Cross-Encoder (→Top-3) → Answerability Gate

Full graph topology
-------------------

                      User Query
                           │
                           ▼
              ┌────── detect_intent ──────┐
              │                           │
              ▼                           ▼
        memory_check              (parallel)
              │
              ▼
          [router]
              │
     ┌────────┴────────────────────┐
     │                             │
     ▼                             ▼
traditional_rag              web_rag
     │                         │
     └────────┬────────────────┘
              │
              ▼
       hybrid_combine
      (Merge + Embedding)
              │
              ▼
        Top-5 Embedding
              │
              ▼
      cross_encoder_rerank
              │
              ▼
         Top-3 Chunks
              │
              ▼
      answerability_check
              │
         ┌────┴────┐
         │         │
        PASS      FAIL
         │         │
         ▼         ▼
      generate  query_expansion
         │         │
         ▼         ▼
     sac_reward  web_rag (retry)
         │
         ▼
    memory_write ──► END

All nodes read from and write back to AgentState — a TypedDict that
threads through the entire graph.

Usage
-----
    from graph import run_pipeline

    result = run_pipeline("How long did Chola dynasty rule?")
    print(result["final_answer"])
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

# ── Pipeline modules (unchanged — same imports as main.py) ──────────────
from intent_detector import detect_intent as _detect_intent, IntentResult
from vector_store import TraditionalRAG
from reranker import multi_stage_funnel
from web_rag import run_web_rag
from verifier import generate_with_self_correction
from sac_learning import log_sac_transition
from hybrid_combiner import unified_hybrid_funnel, extract_top3_sentences
from llm_client import call_llm
from weather_api import get_weather


# ============================================================
# Lazily shared TraditionalRAG instance
# ============================================================
_trad_rag: Optional[TraditionalRAG] = None


def _get_trad_rag() -> TraditionalRAG:
    """
    Construct the vector store only when the routed query actually needs it.
    This avoids loading embedding models during graph import or direct-LLM
    requests.
    """
    global _trad_rag
    if _trad_rag is None:
        _trad_rag = TraditionalRAG()
    return _trad_rag


# ============================================================
# AgentState
# ============================================================

class AgentState(TypedDict, total=False):
    # ── Input ───────────────────────────────────────────────
    question: str

    # ── Intent detection ────────────────────────────────────
    intent: IntentResult          # full IntentResult dataclass
    intent_type: str              # shortcut: intent.intent_type
    needs_web: bool

    # ── Router / memory check ───────────────────────────────
    trad_rag_confident: bool      # True -> vector DB has answer above threshold
    trad_rag_best_chunk: Any      # RetrievedChunk | None

    # ── Traditional RAG retrieval ───────────────────────────
    raw_chunks: List[str]
    sources: List[str]
    trad_chunks: List[str]
    trad_sources: List[str]

    # ── Web RAG ─────────────────────────────────────────────
    web_result: Dict[str, Any]    # full dict from run_web_rag()
    web_chunks: List[str]
    web_sources: List[str]
    web_top3_chunks: List[str]
    web_top3_sources: List[str]

    # ── Shared post-retrieval state ──────────────────────────
    context: str                  # final LLM context string
    top_sentences: List[str]
    funnel_meta: Dict[str, Any]
    route: str                    # "traditional_rag" | "memory" | "web_rag" | etc.
    route_meta: Dict[str, Any]
    direct_answer: str
    weather_data: Dict[str, Any]

    # ── Evidence gate ───────────────────────────────────────
    generation_blocked: bool
    evidence_gate_passed: bool
    answerability_failed: bool
    answerability_reason: str
    query_expansion_triggered: bool
    answer_found: bool

    # ── Generation / verification ───────────────────────────
    gen_result: Dict[str, Any]    # full dict from generate_with_self_correction()
    final_answer: str
    final_score: float
    verification_dimensions: Dict[str, bool]
    passed: bool

    # ── SAC reward ──────────────────────────────────────────
    sac_reward: float

    # ── Error / terminal ────────────────────────────────────
    error: Optional[str]


# ============================================================
# Nodes
# ============================================================

def node_detect_intent(state: AgentState) -> AgentState:
    """
    Node 1 — Intent Detection.
    Runs heuristic fast-path then LLM classifier to determine intent type
    and whether the query needs live web data.
    """
    question = state["question"]
    intent = _detect_intent(question)
    print(f"[Graph] Intent: {intent.intent_type} | needs_web={intent.needs_web} | conf={intent.confidence}")
    return {
        "intent":      intent,
        "intent_type": intent.intent_type,
        "needs_web":   intent.needs_web,
    }


def node_memory_check(state: AgentState) -> AgentState:
    """
    Node 2 — Memory / Confidence Check.
    Asks the vector DB whether it has a high-confidence answer.
    Result gates the router decision.
    """
    question = state["question"]
    if state.get("intent_type") in {"MATH", "CODING", "REASONING"}:
        print("[Graph] Memory check: skipped for direct LLM intent")
        return {
            "trad_rag_confident": False,
            "trad_rag_best_chunk": None,
        }

    confident, best_chunk = _get_trad_rag().has_confident_answer(question)
    print(f"[Graph] Memory check: confident={confident}")
    return {
        "trad_rag_confident":  confident,
        "trad_rag_best_chunk": best_chunk,
    }


def node_traditional_rag(state: AgentState) -> AgentState:
    """
    Node 3 — Traditional RAG (Raw Chunk Extraction).
    Retrieves top-20 chunks from vector DB WITHOUT filtering.
    Filtering will happen in the unified hybrid_combine node.
    
    Returns raw chunks to be merged with web RAG chunks.
    """
    question = state["question"]

    top_chunks_objs = _get_trad_rag().query(question, top_k=20)
    trad_chunks = [c.text for c in top_chunks_objs]
    trad_sources = [c.source for c in top_chunks_objs]

    print(f"[Graph] Traditional RAG: Retrieved {len(trad_chunks)} raw chunks")

    return {
        "trad_chunks": trad_chunks,
        "trad_sources": trad_sources,
    }


def node_web_rag(state: AgentState) -> AgentState:
    """
    Node 4 — Web RAG (Raw Chunk Extraction).
    Runs web search to retrieve raw chunks without filtering yet.
    Stores all chunks and top-3 chunks from the funnel for later hybrid combine.
    
    Returns chunks to be merged with traditional RAG chunks.
    """
    question    = state["question"]
    intent_type = state.get("intent_type", "FACTOID")

    web_result = run_web_rag(question, intent_type=intent_type)

    # Extract chunks from web_result
    # When answerability fails, the top-3 chunks are stored at top level of web_result
    top3_chunks = web_result.get("_top3_chunks", [])
    top3_sources = web_result.get("_top3_sources", ["web"] * len(top3_chunks))
    
    # Get funnel_meta which contains Phase 1 metadata
    funnel_meta = web_result.get("funnel_meta", {})
    
    # For raw pool, try to reconstruct from funnel_meta which has chunk pool info
    all_chunks = funnel_meta.get("_all_chunks", top3_chunks)
    all_sources = funnel_meta.get("_all_sources", top3_sources)
    
    # If no chunks found, use top-3 as fallback
    if not all_chunks and top3_chunks:
        all_chunks = top3_chunks
        all_sources = top3_sources
    
    blocked = bool(web_result.get("generation_blocked", False))
    answer_found = bool(web_result.get("answer_found", False))
    
    print(f"[Graph] Web RAG: answer_found={answer_found}, top3_chunks={len(top3_chunks)}, all_chunks={len(all_chunks)}, blocked={blocked}")

    return {
        "web_chunks": all_chunks,  # All chunks for hybrid combine
        "web_sources": all_sources,
        "web_top3_chunks": top3_chunks,  # Top-3 for fallback
        "web_top3_sources": top3_sources,
        "web_result": web_result,
        "generation_blocked": blocked,
        "answer_found": answer_found,
    }


def node_direct_llm(state: AgentState) -> AgentState:
    """
    Node 4b — Direct LLM.
    Used when retrieval adds latency without adding evidence: math, coding,
    and open reasoning. The graph bypasses embedding, reranking, DQN, SAC,
    and web calls for these intents.
    """
    question = state["question"]
    intent = state.get("intent")
    system = (
        "You are a helpful assistant. Answer the user directly and concisely. "
        "For math, show the essential calculation. For coding, provide correct code "
        "and a short explanation when useful."
    )

    print("[Graph] Direct LLM route: generating without retrieval...")
    answer = call_llm(system=system, prompt=question)

    return {
        "route": "direct_llm",
        "route_meta": {
            "intent": intent.intent_type if intent else state.get("intent_type", "unknown"),
            "reason": "Retrieval not required for this intent.",
        },
        "direct_answer": answer,
        "final_answer": answer,
        "final_score": 1.0 if answer else 0.0,
        "verification_dimensions": {
            "retrieved_context_has_answer": True,
            "answer_contains_entity": bool(answer),
            "user_question_answered": bool(answer),
            "hallucination": False,
        },
        "passed": bool(answer),
        "answer_found": bool(answer),
        "generation_blocked": False,
        "funnel_meta": {},
        "top_sentences": [],
    }


def node_weather(state: AgentState) -> AgentState:
    """Fetch live weather data for the location named in the query."""
    question = state["question"]
    location = question
    match = re.search(
        r"(?:weather|forecast|temperature|rain|snow|wind)\s+(?:in|at|for)\s+(.+)$",
        question,
        re.IGNORECASE,
    )
    if match:
        location = match.group(1)
    else:
        location = re.sub(
            r"\b(what is the|what's the|how is the|current|today's|today|weather|"
            r"forecast|temperature|rain|snow|wind)\b",
            "",
            question,
            flags=re.IGNORECASE,
        ).strip(" ?.,")

    try:
        weather = get_weather(location)
        answer = (
            f"Current weather in {weather['location']}: {weather['condition']}, "
            f"{weather['temperature_c']}°C (feels like {weather['feels_like_c']}°C). "
            f"Humidity is {weather['humidity_percent']}% and wind speed is "
            f"{weather['wind_kmh']} km/h."
        )
        return {
            "route": "weather",
            "weather_data": weather,
            "direct_answer": answer,
            "final_answer": answer,
            "final_score": 1.0,
            "verification_dimensions": {
                "retrieved_context_has_answer": True,
                "answer_contains_entity": True,
                "user_question_answered": True,
                "hallucination": False,
            },
            "passed": True,
            "answer_found": True,
            "generation_blocked": False,
            "funnel_meta": {},
            "top_sentences": [],
        }
    except Exception as exc:
        answer = f"I could not retrieve weather for {location}: {exc}"
        return {
            "route": "weather",
            "direct_answer": answer,
            "final_answer": answer,
            "final_score": 0.0,
            "passed": False,
            "answer_found": False,
            "generation_blocked": False,
            "error": answer,
            "funnel_meta": {},
            "top_sentences": [],
        }


def node_hybrid_combine(state: AgentState) -> AgentState:
    """
    Node 5 — Hybrid Combine.
    Merges chunks from traditional RAG and web RAG into a unified pool,
    then filters to top-3 chunks using a unified pipeline.
    
    This ensures richer context by combining evidence from both sources,
    not just picking the single best chunk.
    """
    question = state["question"]
    intent_type = state.get("intent_type", "FACTOID")
    
    # Get chunks from traditional RAG node output
    trad_chunks = state.get("trad_chunks", [])
    trad_sources = state.get("trad_sources", [])
    
    # Get chunks from web RAG node output
    web_chunks = state.get("web_chunks", [])
    web_sources = state.get("web_sources", [])

    # Recover from the full web result as a compatibility path for states
    # produced before the unified chunk fields were added.
    if not web_chunks:
        web_result = state.get("web_result", {})
        web_funnel_meta = web_result.get("funnel_meta", {})
        web_chunks = web_funnel_meta.get("_all_chunks", web_result.get("_top3_chunks", []))
        web_sources = web_funnel_meta.get("_all_sources", web_result.get("_top3_sources", []))

    # Preserve the legacy traditional-RAG fields when present.
    if not trad_chunks:
        trad_chunks = state.get("raw_chunks", [])
        trad_sources = state.get("sources", [])
    
    print(f"[Graph] Hybrid Combine: Received trad={len(trad_chunks)}, web={len(web_chunks)}")
    
    # If we only have traditional RAG chunks and no web chunks, we're in traditional_rag path
    # In that case, fetch web chunks to enrich
    if trad_chunks and not web_chunks:
        print("[Graph] Hybrid Combine: Traditional RAG alone. Enriching with web chunks...")
        try:
            web_result_new = run_web_rag(question, intent_type=intent_type)
            web_chunks = web_result_new.get("_top3_chunks", [])
            web_sources = web_result_new.get("_top3_sources", ["web"] * len(web_chunks))
        except Exception as e:
            print(f"[Graph] Hybrid Combine: Failed to fetch web chunks: {e}")
    
    # If we only have web RAG chunks and no traditional chunks, we're in web_rag path
    # In that case, fetch traditional chunks to enrich  
    if web_chunks and not trad_chunks:
        print("[Graph] Hybrid Combine: Web RAG alone. Enriching with traditional RAG chunks...")
        try:
            top_chunks_objs = _get_trad_rag().query(question, top_k=10)
            trad_chunks = [c.text for c in top_chunks_objs]
            trad_sources = [c.source for c in top_chunks_objs]
        except Exception as e:
            print(f"[Graph] Hybrid Combine: Failed to fetch traditional RAG chunks: {e}")
    
    # If we have no chunks at all, return blocked state
    if not trad_chunks and not web_chunks:
        print("[Graph] Hybrid Combine: No chunks available from any source")
        return {
            "context": "",
            "top_sentences": [],
            "funnel_meta": {
                "combined_pool_size": 0,
                "trad_rag_count": 0,
                "web_rag_count": 0,
                "hybrid_chunks": [],
                "sources_used": [],
                "evidence_gate_passed": False,
            },
            "generation_blocked": True,
            "evidence_gate_passed": False,
        }
    
    # Run unified hybrid funnel
    top_hybrid_chunks, context, hybrid_meta = unified_hybrid_funnel(
        question=question,
        trad_chunks=trad_chunks,
        trad_sources=trad_sources,
        web_chunks=web_chunks,
        web_sources=web_sources,
        top_final=3,  # Always top-3 for richer context
    )
    
    if not top_hybrid_chunks:
        print("[Graph] Hybrid Combine: No chunks passed the filtering pipeline")
        return {
            "context": "",
            "top_sentences": [],
            "funnel_meta": hybrid_meta,
            "generation_blocked": True,
            "evidence_gate_passed": False,
        }
    
    # Extract fine-grained sentences from all top-3 chunks
    top_sentences = extract_top3_sentences(question, top_hybrid_chunks)
    
    print(f"[Graph] Hybrid Combine: Combined {hybrid_meta.get('combined_pool_size')} chunks → Top-3")
    print(f"  Trad: {hybrid_meta.get('trad_rag_count')}, Web: {hybrid_meta.get('web_rag_count')}")
    print(f"  Sources: {hybrid_meta.get('sources_used')}")
    print(f"  Context length: {len(context)} chars | Top sentences: {len(top_sentences)}")
    
    return {
        "context": context,
        "top_sentences": top_sentences,
        "funnel_meta": hybrid_meta,
        "evidence_gate_passed": hybrid_meta.get("evidence_gate_passed", True),
        "generation_blocked": False,
    }


def node_evidence_gate(state: AgentState) -> AgentState:
    """
    Node 6 — Evidence Gate.
    A read-only routing node: consolidates the generation_blocked and
    evidence_gate_passed flags into a single boolean that the conditional
    edge uses to decide PASS -> generate or FAIL -> no_evidence.

    This is where the two-gate (topic + evidence) DQN result and the
    answerability agent decision both converge.
    """
    blocked       = state.get("generation_blocked", False)
    funnel_meta   = state.get("funnel_meta", {})
    gate_passed   = funnel_meta.get("evidence_gate_passed", True)

    # Combined: blocked explicitly OR DQN evidence gate failed
    evidence_gate_passed = (not blocked) and gate_passed

    if not evidence_gate_passed:
        print(
            f"[Graph] Evidence Gate: FAIL "
            f"(generation_blocked={blocked}, evidence_gate_passed={gate_passed})"
        )
    else:
        print("[Graph] Evidence Gate: PASS")

    return {"evidence_gate_passed": evidence_gate_passed}


def node_generate(state: AgentState) -> AgentState:
    """
    Node 7 — LLM Generation + Verification + Self-Correction.
    Calls generate_with_self_correction which internally runs:
      generate_answer -> verify_answer -> retry loop (up to MAX_RETRIES)
    Returns the best answer and 4D verification dimensions.
    """
    question = state["question"]
    context  = state.get("context", "")

    print("[Graph] Generating answer from top-3 hybrid chunks...")
    gen_result = generate_with_self_correction(question, context)
    gen_result = generate_with_self_correction(question, context)

    intent = state.get("intent")
    gen_result["intent"] = {
        "type":       intent.intent_type  if intent else state.get("intent_type"),
        "confidence": intent.confidence   if intent else 0.0,
        "reasoning":  intent.reasoning    if intent else "",
    }
    gen_result["route"]              = state.get("route", "unknown")
    gen_result["route_meta"]         = state.get("route_meta", {})
    gen_result["funnel_meta"]        = state.get("funnel_meta", {})
    gen_result["extracted_sentences"]= state.get("top_sentences", [])

    web_result = state.get("web_result", {})
    route      = state.get("route", "")
    gen_result["answer_found"]              = state.get("answer_found", True)
    gen_result["query_expansion_triggered"] = state.get("query_expansion_triggered", False)
    gen_result["answerability_reason"]      = (
        web_result.get("answerability_reason", "")
        if "web_rag" in route
        else state.get("answerability_reason", "")
    )

    return {
        "gen_result":              gen_result,
        "final_answer":            gen_result.get("final_answer", ""),
        "final_score":             gen_result.get("final_score", 0.0),
        "verification_dimensions": gen_result.get("verification_dimensions", {}),
        "passed":                  gen_result.get("passed", False),
    }


def node_sac_reward(state: AgentState) -> AgentState:
    """
    Node 8 — SAC Reward Learning.
    Computes the continuous reward signal and logs the (s, a, r, s')
    transition tuple to disk for offline SAC policy updates.
    """
    funnel_meta   = state.get("funnel_meta", {})
    top_sentences = state.get("top_sentences", [])

    dqn_state: Dict[str, Any] = {}
    rich_states  = funnel_meta.get("dqn_rich_states")
    selected_idx = funnel_meta.get("dqn_selected_index", 0)
    if rich_states and len(rich_states) > selected_idx:
        dqn_state = rich_states[selected_idx]

    sac_reward = log_sac_transition(
        query=state["question"],
        dqn_state=dqn_state,
        action_index=selected_idx,
        selected_sentences=top_sentences,
        final_answer=state.get("final_answer", ""),
        verification_dimensions=state.get("verification_dimensions", {}),
        verifier_score=state.get("final_score", 0.0),
        answer_found=state.get("answer_found", True),
        query_expansion_triggered=state.get("query_expansion_triggered", False),
    )

    print(f"[Graph] SAC reward: {sac_reward:.4f}")
    return {"sac_reward": sac_reward}


def node_memory_write(state: AgentState) -> AgentState:
    """
    Node 9 — Memory Write-Back.
    Writes verified (question, answer) pairs back into the vector DB
    so future paraphrases of the same question are answered from memory.
    Only runs when the answer passed verification.
    """
    route = state.get("route", "")
    if route != "memory" and state.get("passed", False):
        _get_trad_rag().add_qa_memory(
            question=state["question"],
            answer=state.get("final_answer", ""),
            route=route,
            verified=True,
        )
        print("[Graph] Memory write-back: answer stored in vector DB.")
    else:
        print("[Graph] Memory write-back: skipped (not passed or memory route).")

    return {}


def node_no_evidence(state: AgentState) -> AgentState:
    """
    Node 10 — No Evidence / Hard Failure.
    Reached when the evidence gate fails after all retrieval attempts.
    Returns a structured refusal so the caller gets a clean dict instead
    of an exception or a hallucinated answer.
    """
    web_result = state.get("web_result", {})
    reason     = web_result.get("reason") or state.get("answerability_reason", "Evidence gate failed.")

    final_answer = (
        "I could not find a reliable answer to your question. "
        "The retrieved documents did not contain the required evidence. "
        "Please try rephrasing your question or check a primary source."
    )

    print(f"[Graph] No-evidence terminal. Reason: {reason}")
    return {
        "error":                     None,
        "answer_found":              False,
        "generation_blocked":        True,
        "final_answer":              final_answer,
        "reason":                    reason,
        "answerability_reason":      state.get("answerability_reason", ""),
        "query_expansion_triggered": state.get("query_expansion_triggered", False),
        "queries_used":              web_result.get("queries_used", []),
        "thoughts":                  web_result.get("thoughts", []),
        "funnel_meta":               state.get("funnel_meta", {}),
        "passed":                    False,
        "sac_reward":                0.0,
    }


# ============================================================
# Conditional edge functions
# ============================================================

def route_after_intent_and_memory(state: AgentState) -> str:
    """
    Router decision after intent detection + memory check.
    - needs_web=True          -> web_rag  (CURRENT_FACT always goes to web)
    - MATH/CODING/REASONING   -> direct_llm
    - confident=True          -> traditional_rag
    - neither                 -> web_rag  (no local knowledge)
    """
    if state.get("intent_type") == "WEATHER":
        print("[Graph] Router -> weather (live weather API)")
        return "weather"
    if state.get("needs_web"):
        print("[Graph] Router -> web_rag (intent requires live data)")
        return "web_rag"
    if state.get("intent_type") in {"MATH", "CODING", "REASONING"}:
        print("[Graph] Router -> direct_llm (retrieval not required)")
        return "direct_llm"
    if state.get("trad_rag_confident"):
        print("[Graph] Router -> traditional_rag (vector DB confident)")
        return "traditional_rag"
    print("[Graph] Router -> web_rag (no confident local answer)")
    return "web_rag"


def route_after_traditional_rag(state: AgentState) -> str:
    """
    After Traditional RAG funnel:
    - answerability_failed=True  -> escalate to web_rag
    - else                       -> hybrid_combine
    """
    if state.get("answerability_failed"):
        print("[Graph] Trad RAG answerability failed -> escalating to web_rag")
        return "web_rag"
    return "hybrid_combine"


def route_after_web_rag(state: AgentState) -> str:
    """
    After Web RAG:
    Always proceeds to evidence_gate — the gate node decides pass/fail.
    """
    return "evidence_gate"


def route_evidence_gate(state: AgentState) -> str:
    """
    Evidence gate decision:
    - PASS -> generate
    - FAIL -> no_evidence
    """
    if state.get("evidence_gate_passed", False):
        return "generate"
    return "no_evidence"


def route_after_sac(state: AgentState) -> str:
    """
    After SAC reward:
    - passed=True  -> memory_write
    - passed=False -> END (nothing worth storing)
    """
    if state.get("passed", False):
        return "memory_write"
    return END


# ============================================================
# Graph construction
# ============================================================

def build_graph() -> StateGraph:
    """
    Builds and returns the compiled LangGraph StateGraph.

    Node execution order (happy path):
      detect_intent
           ↓
      memory_check               (parallel, same turn)
           ↓
      [router]
           ↓
      traditional_rag ──(fail)──► web_rag
           ↓                          ↓
      hybrid_combine ◄────────────────┘
           ↓
      evidence_gate
           ↓ PASS
        generate
           ↓
       sac_reward
           ↓
      memory_write -> END
           ↓ (no_evidence path)
       no_evidence -> END
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ───────────────────────────────────────────────────
    graph.add_node("detect_intent",    node_detect_intent)
    graph.add_node("memory_check",     node_memory_check)
    graph.add_node("traditional_rag",  node_traditional_rag)
    graph.add_node("web_rag",          node_web_rag)
    graph.add_node("direct_llm",       node_direct_llm)
    graph.add_node("weather",           node_weather)
    graph.add_node("hybrid_combine",   node_hybrid_combine)
    graph.add_node("evidence_gate",    node_evidence_gate)
    graph.add_node("generate",         node_generate)
    graph.add_node("sac_reward",       node_sac_reward)
    graph.add_node("memory_write",     node_memory_write)
    graph.add_node("no_evidence",      node_no_evidence)

    # ── Entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("detect_intent")

    # ── Fixed edges ────────────────────────────────────────────────────────
    # detect_intent -> memory_check always (both run before the router)
    graph.add_edge("detect_intent", "memory_check")

    # traditional_rag and web_rag both -> hybrid_combine (combines both sources)
    graph.add_edge("traditional_rag", "hybrid_combine")
    graph.add_edge("web_rag", "hybrid_combine")

    # hybrid_combine -> evidence_gate
    graph.add_edge("hybrid_combine", "evidence_gate")

    # generate always -> sac_reward
    graph.add_edge("generate", "sac_reward")

    # memory_write and no_evidence both terminate
    graph.add_edge("memory_write", END)
    graph.add_edge("no_evidence",  END)
    graph.add_edge("direct_llm",    END)
    graph.add_edge("weather",       END)

    # ── Conditional edges ─────────────────────────────────────────────────
    # Router: after memory_check decide traditional_rag or web_rag
    graph.add_conditional_edges(
        "memory_check",
        route_after_intent_and_memory,
        {
            "traditional_rag": "traditional_rag",
            "web_rag":          "web_rag",
            "direct_llm":       "direct_llm",
            "weather":          "weather",
        },
    )

    # Evidence gate: pass -> generate, fail -> no_evidence
    graph.add_conditional_edges(
        "evidence_gate",
        route_evidence_gate,
        {
            "generate":     "generate",
            "no_evidence":  "no_evidence",
        },
    )

    # After SAC: store memory if answer passed verification
    graph.add_conditional_edges(
        "sac_reward",
        route_after_sac,
        {
            "memory_write": "memory_write",
            END:            END,
        },
    )

    return graph.compile()


# ============================================================
# Public API
# ============================================================

# Compiled graph — import this directly for async / streaming use
pipeline = build_graph()


def run_pipeline(question: str) -> Dict[str, Any]:
    """
    Run the full OmniAgentAI pipeline for a question.

    Args:
        question: The user's natural language query.

    Returns:
        The final AgentState dict with all pipeline outputs:
            final_answer, final_score, verification_dimensions,
            funnel_meta, sac_reward, route, intent, etc.
    """
    initial_state: AgentState = {"question": question}
    final_state = pipeline.invoke(initial_state)
    return dict(final_state)
