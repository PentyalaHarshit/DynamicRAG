"""
Fast Relevance & Credibility Validator:
Evaluates retrieved content for relevance and credibility.
Replaces slow multi-turn CrewAI agent loops with a single fast LLM validation prompt
to reduce request latency from minutes to seconds on local LLM hardware.
"""
import json
from llm_client import call_llm

VALIDATOR_SYSTEM_PROMPT = """You are a research validator checking retrieved context.
Given a user question and a retrieved chunk, evaluate:
1. Is the chunk relevant to the question?
2. Is the content valid and plausible?

Respond ONLY in JSON:
{"relevant": true, "valid": true, "reason": "brief summary"}
"""


def analyze_and_validate(question: str, chunk: str) -> dict:
    """Fast validation pass on retrieved chunk."""
    try:
        prompt = f"Question: {question}\n\nRetrieved Chunk:\n{chunk}"
        raw = call_llm(system=VALIDATOR_SYSTEM_PROMPT, prompt=prompt)
        parsed = json.loads(raw)
        return {
            "relevant": bool(parsed.get("relevant", True)),
            "valid": bool(parsed.get("valid", True)),
            "raw": [parsed.get("reason", "valid context")],
        }
    except Exception:
        # Fallback fast pass if JSON decoding or LLM times out
        is_relevant = any(word in chunk.lower() for word in question.lower().split() if len(word) > 3)
        return {
            "relevant": is_relevant,
            "valid": True,
            "raw": ["Heuristic fast validation pass"],
        }
