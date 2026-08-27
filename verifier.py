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
   (a person name, number, date, or organisation that directly answers the question)?
   FALSE if context only discusses the topic without providing the specific answer.

2. "answer_contains_entity": Does the generated answer contain a specific entity
   (a person name, date, or number) that directly answers the question?
   FALSE if answer says "I cannot", "I don't know", "I am unable to", or gives a generic description.

3. "user_question_answered": Is the user's question fully and specifically resolved?
   FALSE if the answer is evasive, generic, or says it cannot be answered.

4. "hallucination": Does the answer make claims NOT present in the context?
   TRUE if the answer introduces specific names/facts/dates that are absent from the context.

CRITICAL: "I am unable to answer from the provided context" -> answer_contains_entity=false, user_question_answered=false.

Respond ONLY in JSON:
{
  "retrieved_context_has_answer": false,
  "answer_contains_entity": false,
  "user_question_answered": false,
  "hallucination": false,
  "feedback": "Context does not contain the person name. Answer is evasive."
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

    @property
    def hallucination(self) -> bool:
        return self.dimensions.get("hallucination", False)


def _heuristic_verify(question: str, context: str, answer: str) -> VerificationResult:
    """
    Entity-based heuristic verification used when Ollama is unavailable.
    Checks whether the answer contains entities from the context that
    match the expected entity type for the question.
    """
    from answerability_agent import (
        _expected_entity_type,
        _extract_persons,
        _extract_dates,
        _extract_numbers,
        _extract_locations,
    )

    entity_type = _expected_entity_type(question)
    answer_lower = answer.lower()

    # Check if the answer contains a specific entity (not evasive)
    has_entity = False
    ctx_has_answer = False

    if entity_type == "PERSON":
        answer_persons = _extract_persons(answer)
        ctx_persons = _extract_persons(context)
        has_entity = len(answer_persons) > 0
        ctx_has_answer = len(ctx_persons) > 0
        # Check if the answer entity appears in the context
        if has_entity and ctx_persons:
            has_entity = any(p in context for p in answer_persons)
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

    # Hallucination check: does the answer mention entities NOT in the context?
    hallucination = False
    if entity_type == "PERSON" and has_entity:
        answer_persons = _extract_persons(answer)
        for p in answer_persons:
            if p not in context:
                hallucination = True
                break

    dims = {
        "retrieved_context_has_answer": ctx_has_answer,
        "answer_contains_entity": has_entity,
        "user_question_answered": question_answered,
        "hallucination": hallucination,
    }

    score = (0.40 * float(question_answered)) + (0.35 * float(has_entity)) + (0.25 * float(ctx_has_answer))
    if hallucination:
        score = max(0.0, score - 0.50)

    feedback = (
        f"Heuristic verification (LLM unavailable): "
        f"entity_type={entity_type}, has_entity={has_entity}, "
        f"ctx_has_answer={ctx_has_answer}, question_answered={question_answered}"
    )

    return VerificationResult(
        score=round(score, 3),
        dimensions=dims,
        feedback=feedback,
    )


def verify_answer(question: str, context: str, answer: str) -> VerificationResult:
    """
    Evaluates answer quality across 4 factual grounding dimensions.
    Evasive/refused answers are fast-pathed to score 0.0 without LLM call.
    When Ollama is unavailable, uses heuristic entity-based verification.
    """
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
            return _heuristic_verify(question, context, answer)
        parsed = json.loads(raw)
    except Exception:
        # LLM response wasn't valid JSON — try heuristic verification
        print("[Verifier] Could not parse LLM response - using heuristic verification.")
        return _heuristic_verify(question, context, answer)

    ctx_has  = bool(parsed.get("retrieved_context_has_answer", False))
    ent      = bool(parsed.get("answer_contains_entity",       False))
    ans_q    = bool(parsed.get("user_question_answered",       False))
    hal      = bool(parsed.get("hallucination",                False))

    # Calibrated weighted score
    # Highest weight on whether the question is actually answered with an entity
    score = (0.40 * float(ans_q)) + (0.35 * float(ent)) + (0.25 * float(ctx_has))
    if hal:
        score = max(0.0, score - 0.50)   # Hard penalty for hallucination

    dims = {
        "retrieved_context_has_answer": ctx_has,
        "answer_contains_entity":       ent,
        "user_question_answered":       ans_q,
        "hallucination":                hal,
    }

    return VerificationResult(
        score=round(score, 3),
        dimensions=dims,
        feedback=parsed.get("feedback", ""),
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
