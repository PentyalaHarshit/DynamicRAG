"""
Minimal ReAct loop: Thought -> Action -> Observation.

Speed policy
------------
For simple factual intents (BIOGRAPHY, FACTOID, HISTORICAL_FACT, DEFINITION,
COMPARISON, CURRENT_FACT, CURRENCY, WEATHER, FINANCE) the question itself is
already a good search query.  Calling Ollama just to reformat "Who is Thomas
Edison?" into "Thomas Edison biography" adds 8+ seconds and provides zero
retrieval benefit.

The LLM call is therefore skipped for these intents and the raw question
(lightly cleaned) is used directly.  The LLM path is preserved only for
REASONING and CODING queries where rephrasing genuinely helps.
"""
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from agents.search_tool import google_search, _wikipedia_search, SearchResult
from llm_client import call_llm, was_fallback

REACT_SYSTEM_PROMPT = """You are a research agent. Given a user's question, decide what to \
search for on the web to answer it well. Respond ONLY in JSON:
{"thought": "...", "action": "search", "action_input": "search query string"}
If you already have enough search results (given in the conversation), respond:
{"thought": "...", "action": "finish"}
"""

# Intents for which the raw question is a perfectly good search query.
# Bypassing the LLM for these saves 8–30 seconds per query.
_DIRECT_SEARCH_INTENTS = frozenset({
    "BIOGRAPHY", "FACTOID", "HISTORICAL_FACT", "DEFINITION",
    "COMPARISON", "CURRENT_FACT", "CURRENCY", "WEATHER", "FINANCE", "TRAVEL",
})


def _clean_query(question: str) -> str:
    """Strip trailing punctuation/whitespace to make a clean search query."""
    return question.strip().rstrip("?!.,;:")


@dataclass
class ReActTrace:
    thoughts: List[str] = field(default_factory=list)
    queries_used: List[str] = field(default_factory=list)
    results: List[SearchResult] = field(default_factory=list)


def run_react_search(
    question: str,
    max_steps: int = 1,
    intent_type: Optional[str] = None,
) -> ReActTrace:
    """
    Runs a ReAct-style search loop.

    When intent_type is in _DIRECT_SEARCH_INTENTS the LLM step is skipped
    entirely — the question is used directly as the search query, saving
    one full Ollama round-trip (~8–30 s).

    Args:
        question:    The original user query.
        max_steps:   Maximum LLM-formulated search steps (ignored for direct intents).
        intent_type: Intent from intent_detector.  Pass this in for maximum speed.
    """
    trace = ReActTrace()

    # ── Coding path: target GeeksforGeeks, LeetCode, Codeforces, StackOverflow ─────
    if intent_type == "CODING":
        base_q = _clean_query(question)
        query = f"{base_q} geeksforgeeks leetcode codeforces solution"
        trace.thoughts.append("Targeting GeeksforGeeks, LeetCode, and Codeforces solutions.")
        trace.queries_used.append(query)
        trace.results = google_search(query)
        print(f"[ReAct] Coding platform search (intent=CODING): '{query}'")

        if not trace.results:
            fallback_results = _fallback_search(query, num_results=10)
            trace.results.extend(fallback_results)
            if fallback_results:
                print(f"[ReAct] Live coding web search fallback: {len(fallback_results)} results retrieved.")

        return trace

    # ── Fast path: skip LLM, search directly ────────────────────────────
    if intent_type in _DIRECT_SEARCH_INTENTS:
        query = _clean_query(question)
        trace.thoughts.append("Direct search (no LLM query formulation needed).")
        trace.queries_used.append(query)
        trace.results = google_search(query)
        print(f"[ReAct] Direct search (intent={intent_type}): '{query}'")

        if not trace.results:
            print("[ReAct] google_search returned 0 results — trying live web search fallback...")
            fallback_results = _fallback_search(question, num_results=10)
            trace.results.extend(fallback_results)
            if fallback_results:
                print(f"[ReAct] Live web search fallback: {len(fallback_results)} results retrieved.")

        return trace

    # ── Standard path: LLM formulates the search query ──────────────────
    history = f"User question: {question}"

    for _ in range(max_steps):
        raw = call_llm(system=REACT_SYSTEM_PROMPT, prompt=history, timeout=8)
        llm_down = was_fallback()

        try:
            step = json.loads(raw)
        except Exception:
            step = {"thought": "fallback", "action": "search", "action_input": question}
            llm_down = True

        trace.thoughts.append(step.get("thought", ""))

        if step.get("action") == "finish" or not step.get("action_input"):
            break

        query = step["action_input"]
        trace.queries_used.append(query)
        results = google_search(query)
        trace.results.extend(results)

        if llm_down:
            break

        history += (
            f"\nAction: search('{query}')"
            f"\nObservation: {len(results)} results found."
        )

        if len(trace.results) >= 5:
            break

    # Guaranteed fallback: if nothing was returned, try Wikipedia directly
    if not trace.results:
        print("[ReAct] google_search returned 0 results — trying Wikipedia direct search...")
        wiki_results = _wikipedia_search(question, num_results=10)
        trace.results.extend(wiki_results)
        if not trace.queries_used:
            trace.queries_used.append(question)
        if wiki_results:
            print(f"[ReAct] Wikipedia fallback: {len(wiki_results)} results retrieved.")

    return trace
