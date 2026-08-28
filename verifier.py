"""
Multi-Dimensional Verification Agent:
Evaluates generated answers across 4 factual grounding dimensions:
  1. retrieved_context_has_answer: Does the context actually contain the answer entity?
  2. answer_contains_entity: Does the generated answer contain the required name/value?
  3. user_question_answered: Is the question fully resolved (not evasive)?
  4. hallucination: Does the answer add claims not grounded in the context?

Calculates a calibrated composite verification score with corrected weighting.
"I am unable to answer" type responses are scored 0.0.
"""
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, Any

import config
from generator import generate_answer
from llm_client import call_llm, was_fallback

VERIFIER_SYSTEM_PROMPT = """You are a strict factual grounding verifier for an AI-generated answer.
Given a Context, Question, and Answer, evaluate ONLY on these 4 dimensions:

1. "retrieved_context_has_answer": Does the context text actually contain the answer entity
   (a person name, number, date, or specific mathematical derivation steps/solution)?
   FALSE if context only discusses the topic without providing the specific answer or derivation steps.

2. "answer_contains_entity": Does the generated answer contain a specific entity or mathematical derivation
   that directly answers the question?
   FALSE if answer says "I cannot", "I don't know", "I am unable to", or gives a generic description.

3. "user_question_answered": Is the user's question fully and specifically resolved?
   FALSE if the answer is evasive, generic, or says it cannot be answered.
   CRITICAL FOR DERIVATION QUERIES: If the question asks to "derive", "prove", or "show" a mathematical/physics result,
   set user_question_answered=false if the answer only provides a qualitative/high-level summary without the step-by-step mathematical derivation.

4. "hallucination": Does the answer make claims NOT present in the context?
   TRUE if the answer introduces specific names/facts/dates/equations that are absent from the context.

CRITICAL: "I am unable to answer from the provided context" -> answer_contains_entity=false, user_question_answered=false.

Respond ONLY in JSON:
{
  "retrieved_context_has_answer": false,
  "answer_contains_entity": false,
  "user_question_answered": false,
  "hallucination": false,
  "feedback": "Context does not contain derivation steps. Answer is an incomplete summary."
}
"""

# Detect evasive/refused answers without needing LLM
_EVASIVE_RE = re.compile(
    r'\b(i (am |)unable|cannot answer|i don\'?t know|i do not know|'
    r'not (found|mentioned|provided|available) in (the |)context|'
    r'no (information|detail|data|mention) (is |)(found|available|provided)|'
    r'the (context|provided text) does not (contain|mention|include|specify))\b',
    re.IGNORECASE,
)


@dataclass
class VerificationResult:
    score: float
    dimensions: Dict[str, bool]
    feedback: str
    coverage_score: float = 1.0          # fraction of required concepts covered (0.0-1.0)
    uncovered_concepts: list = None      # concepts missing from the answer

    def __post_init__(self):
        if self.uncovered_concepts is None:
            self.uncovered_concepts = []

    @property
    def hallucination(self) -> bool:
        return self.dimensions.get("hallucination", False)


def _is_derivation_query(question: str) -> bool:
    """Detect if question requests a mathematical/scientific derivation or proof."""
    return bool(re.search(
        r'\b(derive|derivation|prove|proof|show\s+that|demonstrate)\b',
        question, re.IGNORECASE
    ))


def _count_derivation_milestones(text: str) -> int:
    """
    Count how many mathematical derivation milestones (0-5) are present in text.
    Derivations require explicit mathematical steps or formulas, not just general background phrases.
    """
    if not text:
        return 0
    t_lower = text.lower()
    milestones = 0
    # 1. Metric Ansatz / Line Element setup (requires formula/variables)
    if re.search(r'\b(ds\^?2|dt\^?2|dr\^?2|d\\omega|e\^?\{?2\\nu\}?|e\^?\{?2\\lambda\}?|g_\{?00\}?|g_\{?rr\}?|metric ansatz|line element)\b', t_lower):
        milestones += 1
    # 2. Vacuum Field Equations Calculation (requires equation or components)
    if re.search(r'\b(g_{\\mu\\nu}\s*=\s*0|r_{\\mu\\nu}\s*=\s*0|r_\{?00\}?\s*=|r_\{?11\}?\s*=|ricci tensor\s*=\s*0|vacuum field equation)\b', t_lower):
        milestones += 1
    # 3. Differential Equations & Integration steps
    if re.search(r'\b(e\^?\{?\\nu\s*\+\s*\\lambda\}?\s*=\s*1|\\nu\'\s*\+\s*\\lambda\'\s*=\s*0|d\\nu/dr|d\\lambda/dr|integration constant|constant of integration)\b', t_lower):
        milestones += 1
    # 4. Boundary Conditions & Newtonian Limit
    if re.search(r'\b(asymptot|infinity|r\s*\\to\s*\\infty|minkowski|weak-field|newtonian limit|phi\s*=\s*-gm/r|\\phi\s*=\s*-gm/r|2gm/c\^?2?)\b', t_lower):
        milestones += 1
    # 5. Final Explicit Metric / Solution Formula
    if re.search(r'\b(boxed|ds\^?2\s*=\s*-\s*\(1|1\s*-\s*\\frac\{2gm\}|1\s*-\s*\\frac\{2gm\}\{c\^?2?\s*r\}|1\s*-\s*\\frac\{r_s\}\{r\}|1\s*-\s*2gm/r)\b', t_lower):
        milestones += 1
    return milestones


def _heuristic_verify(
    question: str,
    context: str,
    answer: str,
    requirements=None,
) -> VerificationResult:
    """
    Entity-based heuristic verification used when Ollama is unavailable.
    Checks whether the answer contains entities from the context that
    match the expected entity type for the question.
    Includes milestone verification for scientific derivation queries.
    """
    from answerability_agent import (
        _expected_entity_type,
        _extract_persons,
        _extract_dates,
        _extract_numbers,
        _extract_locations,
    )

    # ── Scientific Derivation Check ──────────────────────────────────────
    if _is_derivation_query(question):
        ans_milestones = _count_derivation_milestones(answer)
        ctx_milestones = _count_derivation_milestones(context)
        is_derivation_complete = ans_milestones >= 2

        dims = {
            "retrieved_context_has_answer": ctx_milestones >= 2,
            "answer_contains_entity": ans_milestones >= 1,
            "user_question_answered": is_derivation_complete,
            "hallucination": False,
            "incomplete_derivation": not is_derivation_complete,
        }

        score = 1.0 if is_derivation_complete else (0.20 if ans_milestones == 1 else 0.0)
        feedback = (
            f"Scientific Derivation Verifier: answer met {ans_milestones}/5 derivation milestones "
            f"(context met {ctx_milestones}/5). "
            + ("Derivation complete." if is_derivation_complete else "INCOMPLETE DERIVATION: Only qualitative summary provided without mathematical steps.")
        )
        return VerificationResult(score=round(score, 3), dimensions=dims, feedback=feedback)

    entity_type = _expected_entity_type(question)
    answer_lower = answer.lower()

    # Check if the answer contains a specific entity (not evasive)
    has_entity = False
    ctx_has_answer = False

    if entity_type == "PERSON":
        answer_persons = _extract_persons(answer)
        ctx_persons = _extract_persons(context)
        has_entity = len(answer_persons) > 0
        ctx_has_answer = len(ctx_persons) > 0 or has_entity
    elif entity_type == "DATE":
        answer_dates = _extract_dates(answer)
        ctx_dates = _extract_dates(context)
        has_entity = len(answer_dates) > 0
        ctx_has_answer = len(ctx_dates) > 0
    elif entity_type == "NUMBER":
        answer_numbers = _extract_numbers(answer)
        ctx_numbers = _extract_numbers(context)
        has_entity = len(answer_numbers) > 0
        ctx_has_answer = len(ctx_numbers) > 0
    elif entity_type == "LOCATION":
        answer_locations = _extract_locations(answer)
        ctx_locations = _extract_locations(context)
        has_entity = len(answer_locations) > 0
        ctx_has_answer = len(ctx_locations) > 0
    else:
        # Unknown entity type — give benefit of the doubt if answer is not evasive
        has_entity = not _EVASIVE_RE.search(answer)
        ctx_has_answer = True

    # Check if answer is evasive
    is_evasive = _EVASIVE_RE.search(answer) is not None
    question_answered = has_entity and not is_evasive

    # Hallucination check: two-part test for PERSON queries.
    hallucination = False
    if entity_type == "PERSON" and has_entity:
        from answerability_agent import _important_question_terms
        answer_persons = _extract_persons(answer)

        # Part A: primary answer person must be grounded in context or query
        if answer_persons:
            primary_grounded = any(p in context or p in question for p in answer_persons[:2])
            if not primary_grounded and context.strip():
                hallucination = True

        # Part B: the answer must address the actual question subject
        if not hallucination and question:
            q_terms = _important_question_terms(question)
            q_name_terms = {t.lower() for t in q_terms if len(t) > 3}
            if q_name_terms:
                answer_lower = answer.lower()
                if not any(t in answer_lower for t in q_name_terms):
                    hallucination = True

    dims = {
        "retrieved_context_has_answer": ctx_has_answer,
        "answer_contains_entity": has_entity,
        "user_question_answered": question_answered,
        "hallucination": hallucination,
    }

    # ── 5th dimension: Concept Coverage ──────────────────────────────────
    coverage_score = 1.0
    uncovered_concepts: list = []
    if requirements is not None:
        from info_requirements import check_concept_coverage
        cov = check_concept_coverage(answer, requirements)
        coverage_score = cov["coverage_score"]
        uncovered_concepts = cov["uncovered_concepts"]
        dims["concept_coverage"] = coverage_score >= requirements.coverage_threshold

    # Score formula: 4 original dims (0.80 total weight) + coverage (0.20)
    score = (
        (0.32 * float(question_answered))
        + (0.28 * float(has_entity))
        + (0.20 * float(ctx_has_answer))
        + (0.20 * coverage_score)
    )
    if hallucination:
        score = max(0.0, score - 0.50)

    feedback = (
        f"Heuristic verification (LLM unavailable): "
        f"entity_type={entity_type}, has_entity={has_entity}, "
        f"ctx_has_answer={ctx_has_answer}, question_answered={question_answered}, "
        f"coverage={coverage_score:.2f} ({len(requirements.concepts) - len(uncovered_concepts)}/{len(requirements.concepts)} concepts)"
        if requirements else
        f"Heuristic verification (LLM unavailable): "
        f"entity_type={entity_type}, has_entity={has_entity}, "
        f"ctx_has_answer={ctx_has_answer}, question_answered={question_answered}"
    )

    return VerificationResult(
        score=round(score, 3),
        dimensions=dims,
        feedback=feedback,
        coverage_score=coverage_score,
        uncovered_concepts=uncovered_concepts,
    )


def verify_answer(
    question: str,
    context: str,
    answer: str,
    requirements=None,
) -> VerificationResult:
    """
    Evaluates answer quality across 4 factual grounding dimensions.
    Evasive/refused answers are fast-pathed to score 0.0 without LLM call.
    When Ollama is unavailable, uses heuristic entity-based verification.
    """
    # ── Invariant: "[LLM unavailable]" sentinel must never score > 0.0 ──────
    # The fallback synthesis returns this exact string when no LLM and no
    # extractable entity exist.  It is NOT a real answer — it must score 0.0.
    _SENTINEL_RE = re.compile(
        r'\[LLM unavailable|no response generated|LLM unavailable\s*—',
        re.IGNORECASE,
    )
    if _SENTINEL_RE.search(answer):
        dims = {
            "retrieved_context_has_answer": False,
            "answer_contains_entity":       False,
            "user_question_answered":       False,
            "hallucination":                False,
        }
        return VerificationResult(
            score=0.0,
            dimensions=dims,
            feedback=(
                "Answer is the LLM-unavailable sentinel string — "
                "no real answer was generated. Score forced to 0.0."
            ),
        )

    # Fast-path: detect evasive answers without burning an LLM call
    if _EVASIVE_RE.search(answer):
        dims = {
            "retrieved_context_has_answer": False,
            "answer_contains_entity": False,
            "user_question_answered": False,
            "hallucination": False,
        }
        return VerificationResult(
            score=0.0,
            dimensions=dims,
            feedback="Answer is evasive/refused: answer_contains_entity and user_question_answered are false.",
        )

    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}"
    try:
        raw = call_llm(system=VERIFIER_SYSTEM_PROMPT, prompt=prompt)
        # If the LLM call fell back, use heuristic verification instead
        if was_fallback():
            print("[Verifier] LLM unavailable - using heuristic entity verification.")
            return _heuristic_verify(question, context, answer, requirements)
        parsed = json.loads(raw)
    except Exception:
        # LLM response wasn't valid JSON — try heuristic verification
        print("[Verifier] Could not parse LLM response - using heuristic verification.")
        return _heuristic_verify(question, context, answer, requirements)

    ctx_has  = bool(parsed.get("retrieved_context_has_answer", False))
    ent      = bool(parsed.get("answer_contains_entity",       False))
    ans_q    = bool(parsed.get("user_question_answered",       False))
    hal      = bool(parsed.get("hallucination",                False))

    # Derivation completeness override
    incomplete_derivation = False
    if _is_derivation_query(question):
        ans_milestones = _count_derivation_milestones(answer)
        if ans_milestones < 2:
            ans_q = False
            incomplete_derivation = True

    # ── 5th dimension: Concept Coverage (LLM path) ───────────────────────
    coverage_score = 1.0
    uncovered_concepts: list = []
    if requirements is not None:
        from info_requirements import check_concept_coverage
        cov = check_concept_coverage(answer, requirements)
        coverage_score = cov["coverage_score"]
        uncovered_concepts = cov["uncovered_concepts"]
        dims["concept_coverage"] = coverage_score >= requirements.coverage_threshold
        if coverage_score < requirements.coverage_threshold:
            print(
                f"[Verifier] Coverage BELOW threshold: "
                f"{coverage_score:.2f} < {requirements.coverage_threshold:.2f} | "
                f"uncovered={uncovered_concepts}"
            )

    # Calibrated weighted score (coverage gets 0.20 weight)
    score = (
        (0.32 * float(ans_q))
        + (0.28 * float(ent))
        + (0.20 * float(ctx_has))
        + (0.20 * coverage_score)
    )
    if incomplete_derivation:
        score = min(score, 0.20)
    if hal:
        score = max(0.0, score - 0.50)

    dims = {
        "retrieved_context_has_answer": ctx_has,
        "answer_contains_entity":       ent,
        "user_question_answered":       ans_q,
        "hallucination":                hal,
        "incomplete_derivation":        incomplete_derivation,
    }

    return VerificationResult(
        score=round(score, 3),
        dimensions=dims,
        feedback=parsed.get("feedback", ""),
        coverage_score=coverage_score,
        uncovered_concepts=uncovered_concepts,
    )


def _log_episode(record: dict):
    record["timestamp"] = time.time()
    with open(config.EPISODE_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def generate_with_self_correction(question: str, context: str) -> dict:
    """
    Bounded generate -> verify -> retry loop.
    Returns the best answer found plus full 4D verification dimensions.
    """
    feedback = ""
    best = {"answer": "", "score": -1.0, "dimensions": {}}
    attempts = []

    llm_was_down = False

    for attempt in range(1, config.MAX_SELF_CORRECTION_RETRIES + 1):
        answer = generate_answer(question, context, correction_feedback=feedback)

        # Check if the generation used fallback (Ollama unavailable)
        gen_used_fallback = was_fallback()
        if gen_used_fallback:
            llm_was_down = True

        result = verify_answer(question, context, answer)

        attempts.append({
            "attempt":    attempt,
            "answer":     answer,
            "score":      result.score,
            "dimensions": result.dimensions,
            "feedback":   result.feedback,
        })

        if result.score > best["score"]:
            best = {"answer": answer, "score": result.score, "dimensions": result.dimensions}

        passed = (result.score >= config.VERIFIER_PASS_THRESHOLD) and not result.hallucination
        if passed:
            break

        # If the LLM is completely unavailable, don't waste time retrying
        # — the fallback answer is deterministic and won't change
        if llm_was_down:
            print("[Verifier] LLM unavailable - skipping further retries (fallback is deterministic).")
            break

        feedback = result.feedback

    final_record = {
        "question":                question,
        "context":                 context,
        "attempts":                attempts,
        "final_answer":            best["answer"],
        "final_score":             best["score"],
        "verification_dimensions": best["dimensions"],
        "passed":                  best["score"] >= config.VERIFIER_PASS_THRESHOLD,
    }
    _log_episode(final_record)
    return final_record
