"""
Query Expansion Module
======================
When the Answerability Agent (running on Top-3 after cross-encoder) determines
that no chunk contains the required answer entity, this module generates 2-4
targeted queries and performs new web searches.

For CURRENT_FACT queries the current year is always embedded in the generated
queries so search engines return up-to-date results.

Example:
  Input:  "Who is president of Brazil?"  (intent_type=CURRENT_FACT)
  Output: [
    "current president of Brazil 2026",
    "Brazil president 2026",
    "incumbent president Brazil",
    "president of Brazil name 2026",
  ]
"""
import re
import time
from typing import List

from agents.search_tool import google_search, SearchResult


_YEAR = time.strftime("%Y")

# Regex to extract the role + subject from WHO questions
_ROLE_SUBJECT_RE = re.compile(
    r"who\s+(?:is|was|are|were)"
    r"(?:\s+the)?(?:\s+current|\s+incumbent|\s+present|\s+former)?\s+"
    r"(president|prime\s+minister|minister|ceo|head|leader|chancellor|governor|"
    r"secretary|director|chairman|speaker|king|queen|emperor|founder|"
    r"inventor|discoverer|creator|scientist|physicist)"
    r"\s+(?:of\s+)?(.+)",
    re.IGNORECASE,
)


def _heuristic_expansions(query: str, intent_type: str = "FACTOID") -> List[str]:
    """
    Generates 2-4 targeted search queries from the original query.
    Produces clean, high-recall search terms for role and entity queries.
    """
    q = query.strip().rstrip("?")

    # Try to extract role + subject for WHO-type questions
    m = _ROLE_SUBJECT_RE.search(query)
    if m:
        role    = m.group(1).strip()
        subject = m.group(2).strip().rstrip("?").strip()
        expansions = [
            f"current {role} of {subject}",
            f"incumbent {role} {subject}",
            f"who is current {role} of {subject}",
            f"{role} of {subject} name",
        ]
    else:
        expansions = [
            f"current {q}",
            f"incumbent {q} name",
            f"who is current {q}",
            f"{q}",
        ]

    # Deduplicate, preserving insertion order
    seen: set = set()
    unique: List[str] = []
    for e in expansions:
        key = e.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(e.strip())

    return unique[:4]


def expand_and_search(
    query: str,
    intent_type: str = "FACTOID",
    max_new_chunks: int = 10,
) -> List[str]:
    """
    Generates expanded queries, performs new web searches, and returns a flat
    list of text snippets as the new candidate chunk pool.

    Args:
        query:          Original user question.
        intent_type:    From intent_detector — used to embed the year for
                        CURRENT_FACT queries (e.g. "president of Brazil 2026").
        max_new_chunks: Maximum number of snippets to return.

    Returns:
        List of text snippet strings used as the new Top-20 pool for Phase 1.
    """
    expanded_queries = _heuristic_expansions(query, intent_type)
    print(f"[Query Expansion] intent_type={intent_type} | Expanded queries: {expanded_queries}")

    new_results: List[SearchResult] = []
    for eq in expanded_queries:
        try:
            results = google_search(eq, num_results=5)
            new_results.extend(results)
            print(f"[Query Expansion] '{eq}' -> {len(results)} results")
        except Exception as e:
            print(f"[Query Expansion] Search failed for '{eq}': {e}")

    # Return non-empty snippets up to max_new_chunks
    snippets = [r.snippet for r in new_results if r.snippet and r.snippet.strip()]
    return snippets[:max_new_chunks]
