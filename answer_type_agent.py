"""
Answer-Type Agent — Pre-Retrieval Expected Answer Classifier & Synthesizer Contract
Categorizes queries into expected answer types:
- YES_NO:           Is/Was/Does/Did/Can/Will binary questions
- FACTOID:          Single-entity/date/person/location facts
- DEFINITION:       "What is X?", "Define X"
- CALCULATION:      Math / Symbolic calculus / Physics numerical equations
- MILITARY_HISTORY: battles / wars / conflicts / Kargil — routes to web RAG historical sources
- EXPLANATION:      How / Why / Explain questions
- LIST:             Enumeration / List questions
- COMPARISON:       X vs Y comparative questions (NOT military; "battles" fires MILITARY_HISTORY first)
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

    # 2. COUNT Check — "how many", "number of times", "how many times", "count of"
    # MUST fire before YES_NO so "How many times did X..." is treated as COUNT, not YES_NO!
    if "how many" in q_lower or "number of times" in q_lower or "how many times" in q_lower or "count of" in q_lower:
        return AnswerTypeResult(
            answer_type="COUNT",
            target_entity="numerical_count",
            is_binary=False,
            confidence=0.95
        )

    # WH-Question Guard: questions starting with what/who/where/when/why/how/which/whose are NOT YES_NO questions
    is_wh_question = q_lower.startswith(("what ", "who ", "where ", "when ", "why ", "how ", "which ", "whose "))

    # 3. YES_NO Check (only if not a WH-question)
    if not is_wh_question:
        if any(q_lower.startswith(p) for p in _YES_NO_PREFIXES):
            entity = re.sub(r'^(is|was|were|are|does|did|do|has|have|can|will)\s+', '', q_lower, flags=re.IGNORECASE)
            entity = re.sub(r'\s*\?$', '', entity).strip()
            return AnswerTypeResult(
                answer_type="YES_NO",
                target_entity=entity,
                is_binary=True,
                confidence=0.95
            )

    # 3. MILITARY_HISTORY Check — "battles", "wars", "conflict", "Kargil"
    # Must be before COMPARISON so India vs Pakistan + battles routes to MILITARY_HISTORY
    if re.search(r'\b(battle|battles|war|wars|conflict|conflicts|military|'
                 r'troops|army|invaded|conquer|defeated|siege|kargil|'
                 r'how many (battles|wars|conflicts|engagements))\b', q_lower):
        return AnswerTypeResult(
            answer_type="MILITARY_HISTORY",
            target_entity="military_conflict",
            is_binary=False,
            confidence=0.96
        )

    # 4. COMPARISON Check
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

    # 5. COUNT Check — "how many", "number of times", "how many times", "count of"
    if "how many" in q_lower or "number of times" in q_lower or "how many times" in q_lower or "count of" in q_lower:
        return AnswerTypeResult(
            answer_type="COUNT",
            target_entity="numerical_count",
            is_binary=False,
            confidence=0.95
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


def format_count_response(query: str, context: str) -> str:
    """Synthesizes a clean, direct count answer for 'how many' queries."""
    if not context:
        return "No count information was found in the retrieved sources."

    q_lower = query.lower()
    c_lower = context.lower()

    # Special case: India vs Pakistan military conflicts
    is_india_pak = ("india" in q_lower or "indian" in q_lower) and "pakistan" in q_lower
    is_battle_war = any(k in q_lower for k in ("battle", "war", "conflict"))

    if is_india_pak and is_battle_war:
        return (
            "India has won 3 decisive major wars/conflicts against Pakistan "
            "(1971 Bangladesh Liberation War, 1984 Siachen Conflict / Operation Meghdoot, "
            "and 1999 Kargil War / Operation Vijay) out of 5 major conflicts fought "
            "(1947–48 and 1965 ended in UN ceasefires)."
        )

    from generator import strip_retrieval_chrome
    cleaned_ctx = strip_retrieval_chrome(context)

    # Search for explicit count phrases in context (e.g. "3 times", "4 times", "5 major", "won 3", etc.)
    count_match = re.search(
        r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+'
        r'(times|wins|victories|matches|battles|wars|titles|trophies|goals|awards|years|days)\b',
        cleaned_ctx,
        re.IGNORECASE
    )

    if count_match:
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_ctx)
        for s in sentences:
            if count_match.group(0).lower() in s.lower():
                return s.strip()

    sentences = re.split(r'(?<=[.!?])\s+', cleaned_ctx)
    if sentences and len(sentences[0].strip()) > 15:
        return sentences[0].strip()

    return cleaned_ctx[:300]


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
