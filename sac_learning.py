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
      +2.0  answer is correct (entity found, question answered, no hallucination)
      -0.5  context was topically relevant but the specific entity was missing
      -2.0  hallucination detected (fabricated facts)
       0.0  fallback / ambiguous outcome
    """
    ans_q   = bool(verification_dimensions.get("user_question_answered",       False))
    ent     = bool(verification_dimensions.get("answer_contains_entity",       False))
    ctx_has = bool(verification_dimensions.get("retrieved_context_has_answer", False))
    hal     = bool(verification_dimensions.get("hallucination",                False))

    if hal:
        # Hard penalty: model made up an answer not grounded in context
        return -2.0

    if ans_q and ent and not hal:
        # Full success: question answered with a real entity from the context
        return +2.0

    if ctx_has and not ent:
        # Soft penalty: chunk was topically relevant but missing the specific answer entity
        # This signals the retrieval stage to try harder (query expansion)
        return -0.5

    if not ctx_has and not ent:
        # Retrieval completely missed: stronger soft penalty than entity-missing
        return -0.8

    # Partial credit: some dimensions passed
    return round(float(verifier_score) - 0.5, 4)


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
        "next_state_s_prime": {
            "terminal":  True,
            "passed":    verifier_score >= config.VERIFIER_PASS_THRESHOLD,
        },
    }

    os.makedirs(os.path.dirname(SAC_EPISODE_LOG_PATH), exist_ok=True)
    with open(SAC_EPISODE_LOG_PATH, "a") as f:
        f.write(json.dumps(transition) + "\n")

    return reward
