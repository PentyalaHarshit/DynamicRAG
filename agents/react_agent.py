"""
Minimal ReAct loop: Thought -> Action -> Observation, repeated until the agent
decides it has enough search results to hand off to the RAG/rerank stage.
Kept intentionally small - this is not meant to be a general-purpose agent
framework, just enough reasoning to turn a user question into 1-2 good
search queries instead of firing the raw question at Google verbatim.
"""
import json
from dataclasses import dataclass, field
from typing import List

from agents.search_tool import google_search, _wikipedia_search, SearchResult
from llm_client import call_llm, was_fallback

REACT_SYSTEM_PROMPT = """You are a research agent. Given a user's question, decide what to \
search for on the web to answer it well. Respond ONLY in JSON:
{"thought": "...", "action": "search", "action_input": "search query string"}
If you already have enough search results (given in the conversation), respond:
{"thought": "...", "action": "finish"}
"""


@dataclass
class ReActTrace:
    thoughts: List[str] = field(default_factory=list)
    queries_used: List[str] = field(default_factory=list)
    results: List[SearchResult] = field(default_factory=list)


def run_react_search(question: str, max_steps: int = 3) -> ReActTrace:
    trace = ReActTrace()
    history = f"User question: {question}"

    for _ in range(max_steps):
        raw = call_llm(system=REACT_SYSTEM_PROMPT, prompt=history)

        # If LLM is down/fallback, skip further LLM calls and search directly
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

        # If LLM unavailable, break immediately after first search — no need to iterate
        if llm_down:
            break

        history += f"\nAction: search('{query}')\nObservation: {len(results)} results found."

        if len(trace.results) >= 5:
            break

    # Guaranteed fallback: if nothing was returned, try Wikipedia directly
    if not trace.results:
        print(f"[ReAct] google_search returned 0 results — trying Wikipedia direct search...")
        wiki_results = _wikipedia_search(question, num_results=10)
        trace.results.extend(wiki_results)
        if not trace.queries_used:
            trace.queries_used.append(question)
        if wiki_results:
            print(f"[ReAct] Wikipedia fallback: {len(wiki_results)} results retrieved.")

    return trace
