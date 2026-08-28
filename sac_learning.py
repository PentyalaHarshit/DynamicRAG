"""
SAC Reward Learning Module:
Computes continuous reward signals R(s, a) from 4D Verification Agent outputs.

Corrected Reward Structure:
  +2.0  answer_correct (question answered with a real entity)
  -0.5  context_relevant_but_answer_missing (DQN picked a relevant chunk but
         the answer entity was not in it — soft penalty for retrieval gap)
  -2.0  hallucination (answer fabricated facts not in context — hard penalty)
   0.0  otherwise (generic fallback)

Logs state-action-reward transition tuples (s, a, r, s') to disk for offline
Soft Actor-Critic (SAC) reinforcement learning model updates.
"""
import json
import time
import os
from typing import Dict, Any, List

import config

SAC_EPISODE_LOG_PATH = os.path.join(os.path.dirname(config.EPISODE_LOG_PATH), "sac_episodes.jsonl")


def compute_sac_reward(
    verification_dimensions: Dict[str, bool],
    verifier_score: float,
) -> float:
    """
    Computes a continuous SAC policy reward signal using corrected reward structure.

    Reward ladder:
      +2.0  answer is correct (entity found, question answered, no hallucination, complete derivation)
      -0.7  incomplete derivation or user question unanswered despite relevant topic
      -0.8  retrieval completely missed / zero evidence
      -2.0  hallucination detected (fabricated facts)
    """
    ans_q     = bool(verification_dimensions.get("user_question_answered",       False))
    ent       = bool(verification_dimensions.get("answer_contains_entity",       False))
    ctx_has   = bool(verification_dimensions.get("retrieved_context_has_answer", False))
    hal       = bool(verification_dimensions.get("hallucination",                False))
    incomp_der= bool(verification_dimensions.get("incomplete_derivation",        False))

    if hal:
        # Hard penalty: model made up an answer not grounded in context
        return -2.0

    if incomp_der:
        # Incomplete scientific derivation: answer was only a high-level overview
        return -0.7

    if ans_q and ent and not hal:
        if ctx_has:
            # Full success: question answered with real entity/derivation grounded in context
            return +2.0
        else:
            # Ungrounded success: answer generated from LLM memory without retrieved context evidence
            return +0.5

    if ctx_has and not ans_q:
        # Context was retrieved but question was not fully answered
        return -0.7

    if ctx_has and not ent:
        # Soft penalty: chunk was topically relevant but missing the specific answer entity
        return -0.5

    if not ctx_has and not ent:
        # Retrieval completely missed / zero evidence
        return -0.8

    # Default fallback calculation
    return round(float(verifier_score) - 0.7, 4)


def log_sac_transition(
    query: str,
    dqn_state: Dict[str, Any],
    action_index: int,
    selected_sentences: List[str],
    final_answer: str,
    verification_dimensions: Dict[str, bool],
    verifier_score: float,
    answer_found: bool = True,
    query_expansion_triggered: bool = False,
    strategy: str = "",
    failure_type: str = "none",
    attempt_count: int = 1,
    reward_components: Dict[str, float] | None = None,
) -> float:
    """
    Logs (s, a, r, s') state-action-reward-next_state transition tuple for offline SAC learning.
    Returns the computed reward value.
    """
    reward = compute_sac_reward(verification_dimensions, verifier_score)

    transition = {
        "timestamp":                  time.time(),
        "query":                      query,
        "state_s":                    dqn_state,
        "action_a":                   action_index,
        "reward_r":                   reward,
        "selected_sentences":         selected_sentences,
        "final_answer":               final_answer,
        "verification_dimensions":    verification_dimensions,
        "answer_found":               answer_found,
        "query_expansion_triggered":  query_expansion_triggered,
        "strategy":                   strategy,
        "failure_type":                failure_type,
        "attempt_count":               attempt_count,
        "reward_components":           reward_components or {},
        "next_state_s_prime": {
            "terminal":  True,
            "passed":    verifier_score >= config.VERIFIER_PASS_THRESHOLD,
        },
    }

    os.makedirs(os.path.dirname(SAC_EPISODE_LOG_PATH), exist_ok=True)
    with open(SAC_EPISODE_LOG_PATH, "a") as f:
        f.write(json.dumps(transition) + "\n")

    return reward
