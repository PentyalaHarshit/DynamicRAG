"""Deterministic problem understanding and pattern extraction."""
from __future__ import annotations

from typing import Any, Dict, List
import re


_PATTERNS = (
    ("scientific_research", "research", "scientific_sources", re.compile(
        r"\b(latest|recent|new|current)\s+(?:physics|mathematics|scientific)\s+"
        r"(research|work|findings?)\b|\b(research|research paper|textbook|"
        r"scientific sources?|papers?)\b.*\b(physics|mathematics|quantum|gravity)\b",
        re.IGNORECASE)),
    ("news_research", "news", "current_events", re.compile(
        r"\b(latest|recent|today|current)\s+(news|updates?|events?)\b",
        re.IGNORECASE)),
    # ── General Relativity — must come BEFORE special_relativity so that
    #    "Schwarzschild" and "Einstein field equations" are caught first. ───
    ("general_relativity", "physics", "general_relativity", re.compile(
        r"\b(schwarzschild|kerr|reissner|nordstrom|friedmann|"
        r"einstein\s+field\s+equation|ricci\s+(tensor|scalar)|"
        r"riemann\s+(tensor|curvature)|stress.energy\s+tensor|"
        r"christoffel|geodesic\s+equation|spacetime\s+(metric|curvature)|"
        r"metric\s+tensor|covariant\s+derivative|general\s+relativity|"
        r"gravitational\s+(wave|lensing|redshift|collapse)|"
        r"event\s+horizon|singularity|black\s+hole\s+(metric|solution)|"
        r"vacuum\s+solution|static\s+spherically\s+symmetric)\b",
        re.IGNORECASE)),
    # ── Theoretical derivation / proof request (any domain) ──────────────
    # Catches: "derive X", "prove X", "show that X", "derivation of X"
    # This fires for any high-complexity derivation not already caught above.
    ("theoretical_derivation", "physics", "theoretical_physics", re.compile(
        r"\b(derive\s+(?:the\s+)?|derivation\s+of\s+(?:the\s+)?|"
        r"proof\s+of\s+(?:the\s+)?|prove\s+(?:that\s+)?|"
        r"show\s+that\s+(?:the\s+)?)\b",
        re.IGNORECASE)),
    ("perturbed_harmonic_oscillator", "physics", "quantum_mechanics", re.compile(
        r"(?=.*\b(ground[- ]state|harmonic oscillator|relativistic correction)\b)"
        r"(?=.*\b(lambda|λ)\b)(?=.*\b(x|p)\b)",
        re.IGNORECASE)),
    ("special_relativity", "physics", "relativity", re.compile(
        r"\b(spacecraft|spaceship|relativity|relativistic|lorentz|time dilation)\b|"
        r"\d+(?:\.\d+)?\s*c\b", re.IGNORECASE)),
    ("symbolic_calculus", "mathematics", "calculus", re.compile(
        r"\b(derivative|differentiate|integrat(?:e|ion)|integral|limit)\b|"
        r"[∫]\s*[a-zA-Z0-9]",
        re.IGNORECASE)),
    ("percentage_word_problem", "mathematics", "arithmetic", re.compile(
        r"\b(percentage|percent)\b|\d+(?:\.\d+)?\s*%", re.IGNORECASE)),
    ("coding_algorithm", "computer_science", "programming", re.compile(
        r"\b(code|python|javascript|js|typescript|java|c\+\+|cpp|c#|sql|script|"
        r"function|algorithm|class|method|syntax|debug|write\s+a\s+program|"
        r"programming|write\s+code|leetcode|geeksforgeeks|codeforces|hackerrank|"
        r"codechef|interviewbit|solution|two\s+sum|quicksort|mergesort)\b",
        re.IGNORECASE)),
    ("numeric_aggregate", "mathematics", "arithmetic", re.compile(
        r"\b(average|mean|sum)\b", re.IGNORECASE)),
    ("fitness_routine", "health_fitness", "workout_routine", re.compile(
        r"\b(workout|workout\s+plan|workout\s+routine|exercise|fitness|"
        r"gym\s+routine|training\s+plan|diet\s+plan)\b",
        re.IGNORECASE)),
    ("finance_credit", "finance", "banking", re.compile(
        r"\b(credit\s+card|debit\s+card|bank\s+account|interest\s+rate|"
        r"mortgage|loan|credit\s+score)\b",
        re.IGNORECASE)),
    ("travel_flight", "travel", "flights", re.compile(
        r"\b(flight|flights|airfare|plane ticket|airline|airlines|fly from|dallas to hyderabad|cheapest flight)\b",
        re.IGNORECASE)),
    ("quantum_mechanics", "physics", "quantum_mechanics", re.compile(
        r"\b(quantum|Schrodinger|Schrödinger|Hamiltonian|wavefunction|perturbation)\b",
        re.IGNORECASE)),
)


def detect_question_pattern(question: str) -> str:
    """
    Stage 1: Question Pattern / Operation Detector.
    Returns one of: COUNT, IDENTITY, DEFINITION, PROCEDURE, COMPARISON,
                    SELECTION_RANKING, TIME_CURRENT, CALCULATION, EXPLANATION,
                    HISTORICAL_EVENT, LIST, GENERAL
    """
    q = question.strip().lower()

    # 1. COUNT
    if any(k in q for k in (
        "how many", "how much", "number of", "count of", "total number", "how often",
        "how frequently", "how many times", "how many people", "how many years",
        "how many matches", "how many battles", "count"
    )):
        return "COUNT"

    # 2. TIME / CURRENT (must fire before DEFINITION so 'temperature right now' is TIME_CURRENT)
    if any(k in q for k in (
        "right now", "currently", "current", "today", "tonight", "this week",
        "this month", "recent", "recently", "latest", "news", "what happened today",
        "what is happening", "as of today", "at present"
    )):
        return "TIME_CURRENT"

    # 3. CALCULATION
    if any(k in q for k in (
        "calculate", "compute", "solve", "evaluate", "determine", "derive",
        "differentiate", "integrate", "simplify", "factor", "solve for x",
        "find the derivative", "find the integral"
    )) or re.search(r'\b\d+\s*[\+\-\*/\^]\s*\d+\b', q):
        return "CALCULATION"

    # 4. IDENTITY / WHO
    if any(q.startswith(k) for k in (
        "who is", "who was", "who are", "who were", "who founded", "who created",
        "who invented", "who developed", "who discovered", "who won", "who currently"
    )) or "who is the current" in q:
        return "IDENTITY"

    # 5. COMPARISON (must fire before DEFINITION/RANKING for 'which is better, RAG or X')
    if " vs " in q or "versus" in q or "difference between" in q or "compare" in q or "comparison of" in q or "better than" in q or "faster than" in q or "which is better" in q or re.search(r'\bbetter,?\s+.*\s+or\b', q):
        return "COMPARISON"

    # 6. SELECTION / RANKING
    if any(k in q for k in (
        "who is the best", "what is the best", "top ", "highest", "lowest", "largest",
        "smallest", "most ", "least ", "maximum", "minimum", "ranking", "best"
    )) or q.startswith("which "):
        return "SELECTION_RANKING"

    # 7. PROCEDURE / METHOD
    if any(q.startswith(k) for k in (
        "how can i", "how do i", "how to", "how does", "steps to", "way to",
        "method to", "procedure for", "process of"
    )):
        return "PROCEDURE"

    # 8. DEFINITION / MEANING
    if any(q.startswith(k) for k in ("what is ", "what are ", "define ", "definition of ")) or ("what does" in q and "mean" in q) or "what is meant by" in q:
        return "DEFINITION"

    # 9. EXPLANATION / REASONING
    if q.startswith("why ") or "explain why" in q or "how does it work" in q or "reason why" in q or "cause of" in q:
        return "EXPLANATION"

    # 10. HISTORICAL_EVENT
    if any(q.startswith(k) for k in ("when did", "when was", "when were", "history of")) or any(k in q for k in ("war", "battle", "conflict", "invasion", "independence", "revolution", "who ruled")):
        return "HISTORICAL_EVENT"

    # 11. LIST
    if q.startswith("list ") or "examples of" in q or "types of" in q:
        return "LIST"

    return "GENERAL"


def analyze_problem(question: str) -> Dict[str, Any]:
    """Extract a compact, explainable problem representation before routing."""
    normalized = question.strip()
    op_pattern = detect_question_pattern(normalized)

    # Derivation/proof complexity signals
    is_derivation = bool(re.search(
        r"\b(derive|derivation|prove|proof|show\s+that|demonstrate)\b",
        normalized, re.IGNORECASE,
    ))

    for pattern, domain, subdomain, matcher in _PATTERNS:
        if matcher.search(normalized):
            features: List[str] = []
            if re.search(r"\d+(?:\.\d+)?\s*c\b", normalized, re.IGNORECASE):
                features.append("light_speed_fraction")
            if re.search(r"\d+(?:\.\d+)?\s*%", normalized):
                features.append("percentage_values")
            if re.search(r"\b(second|third|first)\s+derivative", normalized, re.IGNORECASE):
                features.append("derivative_order")
            if re.search(r"\b(topics?|papers?|textbooks?|research|source)", normalized, re.IGNORECASE):
                features.append("research_requested")
            if is_derivation:
                features.append("derivation")

            complexity = (
                "very_high"
                if (is_derivation or subdomain in {"general_relativity", "theoretical_physics", "quantum_mechanics"})
                else "medium"
            )

            needs_research = domain in {"physics"} or is_derivation

            return {
                "operation_pattern": op_pattern,
                "domain":            domain,
                "subdomain":         subdomain,
                "pattern":           pattern,
                "features":          features,
                "complexity":        complexity,
                "needs_research":    needs_research,
                "confidence":        0.95,
                "reason":            f"Matched deterministic pattern: {pattern} ({op_pattern}).",
            }

    return {
        "operation_pattern": op_pattern,
        "domain":            "general",
        "subdomain":         "unknown",
        "pattern":           "general_question",
        "features":          [],
        "complexity":        "medium",
        "needs_research":    False,
        "confidence":        0.50,
        "reason":            f"No specialized deterministic pattern matched ({op_pattern}).",
    }
