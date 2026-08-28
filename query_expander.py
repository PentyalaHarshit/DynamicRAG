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

# ---------------------------------------------------------------------------
# Currency-specific expansion
# ---------------------------------------------------------------------------

# Canonical ISO codes for currencies commonly referenced by name in queries
_CURRENCY_CODE_MAP = {
    "dollar": "USD", "dollars": "USD", "usd": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
    "yen": "JPY", "jpy": "JPY",
    "rupee": "INR", "rupees": "INR", "inr": "INR",
    "yuan": "CNY", "cny": "CNY",
    "won": "KRW", "krw": "KRW",
    "franc": "CHF", "francs": "CHF", "chf": "CHF",
    "peso": "MXN", "pesos": "MXN", "mxn": "MXN",
    "ruble": "RUB", "rubles": "RUB", "rub": "RUB",
    "dirham": "AED", "aed": "AED",
    "riyal": "SAR", "sar": "SAR",
    "baht": "THB", "thb": "THB",
}

# Extracts  FROM  and  TO  currency tokens from a conversion query.
# Handles: "100 USD in INR", " USD to INR", "dollars to rupees", etc.
_CURRENCY_EXP_RE = re.compile(
    r"""
    (?:\d[\d,\.]*)?\s*                        # optional amount
    (?P<from>[A-Za-z]{2,8})\s+                # from currency (name or code)
    (?:in|to|into|=|->)\s+                    # separator
    (?P<to>[A-Za-z]{2,8})                     # to currency
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _currency_expansions(query: str) -> list[str]:
    """
    Generates targeted search queries for a currency conversion query.
    Always produces at least one clean query even when the dollar sign
    was stripped by the shell (e.g. query arrives as ' USD in INR?').
    """
    m = _CURRENCY_EXP_RE.search(query)
    if not m:
        # Fallback: generic exchange rate query
        q = re.sub(r"[^a-zA-Z0-9 ]", "", query).strip()
        return [
            f"{q} exchange rate today",
            f"live {q} rate {_YEAR}",
        ]

    raw_from = m.group("from").strip().lower()
    raw_to   = m.group("to").strip().lower()

    # Resolve to ISO codes when possible, fall back to the raw token
    from_code = _CURRENCY_CODE_MAP.get(raw_from, raw_from.upper())
    to_code   = _CURRENCY_CODE_MAP.get(raw_to,   raw_to.upper())

    # Extract numeric amount if present
    amt_match = re.search(r"(\d[\d,\.]*)", query)
    amount = amt_match.group(1).replace(",", "") if amt_match else "1"

    return [
        f"{from_code} to {to_code} exchange rate today",
        f"{amount} {from_code} in {to_code} today",
        f"live {from_code} {to_code} rate {_YEAR}",
        f"current exchange rate {from_code} {to_code}",
    ]

# Regex to extract the role + subject from WHO questions
_ROLE_SUBJECT_RE = re.compile(
    r"who\s+(?:is|was|are|were)"
    r"(?:\s+the)?(?:\s+current|\s+incumbent|\s+present|\s+former)?\s+"
    r"(president|prime\s+minister|minister|ceo|head|leader|chancellor|governor|"
    r"secretary|director|chairman|speaker|king|queen|emperor|founder|"
    r"inventor|discoverer|creator|scientist|physicist|programmer|coder|developer)"
    r"\s+(?:of\s+)?(.+)",
    re.IGNORECASE,
)


def _heuristic_expansions(query: str, intent_type: str = "FACTOID") -> List[str]:
    """
    Generates 2-4 targeted search queries from the original query.
    Produces clean, high-recall search terms for role and entity queries.
    """
    q = query.strip().rstrip("?")

    # Scientific / Math / Derivation queries
    if intent_type in {"SCIENTIFIC_REASONING", "MATH", "REASONING", "CODING"} or any(w in q.lower() for w in ("derive", "derivation", "proof", "prove", "equation")):
        expansions = [
            f"{q}",
            f"{q} proof equations",
            f"{q} derivation step by step",
        ]
    # Try to extract role + subject for WHO-type questions
    elif _ROLE_SUBJECT_RE.search(query):
        m = _ROLE_SUBJECT_RE.search(query)
        role    = m.group(1).strip()
        subject = m.group(2).strip().rstrip("?").strip()
        expansions = [
            f"current {role} of {subject}",
            f"incumbent {role} {subject}",
            f"who is current {role} of {subject}",
            f"{role} of {subject} name",
        ]
    elif intent_type == "CURRENT_FACT":
        expansions = [
            f"current {q}",
            f"incumbent {q} name",
            f"{q} {_YEAR}",
        ]
    else:
        expansions = [
            f"{q}",
            f"{q} summary",
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
    # Currency queries get specialised search terms that hit live rate pages
    if intent_type == "CURRENCY":
        expanded_queries = _currency_expansions(query)
    else:
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
