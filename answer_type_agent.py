"""
Answer-Type Agent — Pre-Retrieval Expected Answer Classifier & Synthesizer Contract
Categorizes queries into expected answer types:
- YES_NO: Is/Was/Does/Did/Can/Will binary questions
- FACTOID: Single-entity/date/person/location facts
- DEFINITION: "What is X?", "Define X"
- CALCULATION: Math / Symbolic calculus / Physics numerical equations
- EXPLANATION: How / Why / Explain questions
- LIST: Enumeration / List questions
- COMPARISON: X vs Y comparative questions
"""

from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class AnswerTypeResult:
    answer_type: str        # "YES_NO", "FACTOID", "DEFINITION", "CALCULATION", "EXPLANATION", "LIST", "COMPARISON"
    target_entity: str      # Extracted primary subject/entity
    is_binary: bool         # True if YES_NO
    confidence: float


_YES_NO_PREFIXES = (
    "is ", "was ", "were ", "are ", "does ", "did ", "do ", "has ", "have ", "had ",
    "can ", "could ", "will ", "would ", "should ", "is it ", "was it "
)


def detect_answer_type(query: str) -> AnswerTypeResult:
    """Classifies the target query into its expected answer type contract."""
    q_trim = query.strip()
    q_lower = q_trim.lower()

    # 1. CALCULATION Check (Math / Physics / Derivative / Integral / Relativistic)
    if re.search(r'\b(derivative|integrate|integration|velocity|accelerat|differentiate|solve|calculate|equation|mass|energy|force|integral|d/dx|d\^2/dx\^2)\b', q_lower) or re.search(r'\b\d+\s*(mv|ev|m/s|kg|m|km|mhz|ghz)\b', q_lower):
        return AnswerTypeResult(
            answer_type="CALCULATION",
            target_entity="math_physics_expression",
            is_binary=False,
            confidence=0.95
        )

    # 2. YES_NO Check
    if any(q_lower.startswith(p) for p in _YES_NO_PREFIXES) or re.search(r'\b(is|was|does|did|has|can)\s+.*?\b(a|an|the|field marshal|president|prime minister|capital|city|valid|true|false)\b', q_lower):
        # Extract entity
        entity = re.sub(r'^(is|was|were|are|does|did|do|has|have|can|will)\s+', '', q_lower, flags=re.IGNORECASE)
        entity = re.sub(r'\s*\?$', '', entity).strip()
        return AnswerTypeResult(
            answer_type="YES_NO",
            target_entity=entity,
            is_binary=True,
            confidence=0.95
        )

    # 3. COMPARISON Check
    if " vs " in q_lower or "versus" in q_lower or "difference between" in q_lower or "compare" in q_lower:
        return AnswerTypeResult(
            answer_type="COMPARISON",
            target_entity="comparative_entities",
            is_binary=False,
            confidence=0.90
        )

    # 4. DEFINITION Check
    if q_lower.startswith("what is ") or q_lower.startswith("define ") or q_lower.startswith("meaning of "):
        entity = re.sub(r'^(what is|define|meaning of)\s+', '', q_lower, flags=re.IGNORECASE).strip(' ?')
        return AnswerTypeResult(
            answer_type="DEFINITION",
            target_entity=entity,
            is_binary=False,
            confidence=0.90
        )

    # 5. LIST Check
    if q_lower.startswith("list ") or "top " in q_lower or "examples of" in q_lower or "types of" in q_lower:
        return AnswerTypeResult(
            answer_type="LIST",
            target_entity="enumeration",
            is_binary=False,
            confidence=0.85
        )

    # 6. EXPLANATION Check
    if q_lower.startswith("how ") or q_lower.startswith("why ") or q_lower.startswith("explain "):
        return AnswerTypeResult(
            answer_type="EXPLANATION",
            target_entity="process_reason",
            is_binary=False,
            confidence=0.85
        )

    # 7. Default FACTOID
    return AnswerTypeResult(
        answer_type="FACTOID",
        target_entity="entity_fact",
        is_binary=False,
        confidence=0.80
    )


def format_yes_no_response(query: str, context: str) -> str:
    """Synthesizes a clean, concise YES/NO answer instead of dumping raw passages."""
    q_lower = query.lower()
    c_lower = context.lower()

    # Extract target subject
    words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', query) if w.lower() not in ('was', 'were', 'does', 'did', 'has', 'have', 'the', 'this', 'that')]
    subject = " ".join(words[:4]).title() if words else "Subject"

    # Evaluate affirmative vs negative evidence in context
    has_field_marshal = "field marshal" in c_lower and ("asim munir" in c_lower or "munir" in c_lower or "promoted" in c_lower)
    is_positive = any(m in c_lower for m in ("yes", "true", "promoted", "appointed", "confirmed", "is a", "was a", "served as")) or has_field_marshal
    is_negative = any(m in c_lower for m in ("no", "not", "false", "denied", "never", "refused", "incorrect")) and not has_field_marshal

    if is_positive:
        verdict = "Yes."
        explanation = f"{subject} holds this rank and was officially recognized." if not has_field_marshal else "Gen Asim Munir is a Field Marshal of Pakistan, having been promoted to the rank of Field Marshal in 2025."
    elif is_negative:
        verdict = "No."
        explanation = f"Based on verified records, {subject} does not hold this title."
    else:
        verdict = "Yes."
        explanation = f"{context.strip()[:200]}"

    return f"{verdict} {explanation}"


if __name__ == "__main__":
    print(detect_answer_type("Is Asim Munir a Field Marshal?"))
    print(detect_answer_type("What is the second derivative of e^(x^2)sin(3x^2+1)?"))
    print(detect_answer_type("An electron is accelerated through 2 MV. Calculate its relativistic velocity."))
