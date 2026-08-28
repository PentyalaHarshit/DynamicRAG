"""Pattern-aware strategy selection with a deterministic RL-compatible interface."""
from __future__ import annotations

from typing import Any, Dict


_STRATEGIES = {
    "perturbed_harmonic_oscillator": {"strategy": "symbolic_perturbation_solver", "action_index": 0},
    "special_relativity": {"strategy": "relativistic_solver", "action_index": 0},
    "symbolic_calculus": {"strategy": "symbolic_solver", "action_index": 0},
    "percentage_word_problem": {"strategy": "numeric_solver", "action_index": 1},
    "numeric_aggregate": {"strategy": "numeric_solver", "action_index": 1},
    "differential_equation": {"strategy": "symbolic_solver", "action_index": 0},
    "quantum_mechanics": {"strategy": "research_then_solver", "action_index": 2},
}


def select_strategy(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Select a solver action from the pattern state."""
    pattern = str(analysis.get("pattern") or "")
    selected = _STRATEGIES.get(pattern, {
        "strategy": "research_then_llm",
        "action_index": 2,
    })
    return {
        "strategy": selected["strategy"],
        "action_index": selected["action_index"],
        "policy": "deterministic_baseline",
        "candidates": ["symbolic_solver", "numeric_solver", "research_then_llm"],
    }


def select_next_strategy(current: Dict[str, Any], failed: list[str]) -> Dict[str, Any]:
    """Choose the next candidate while avoiding strategies that already failed."""
    candidates = current.get("candidates", [])
    for index, candidate in enumerate(candidates):
        if candidate not in failed and candidate != current.get("strategy"):
            return {
                **current,
                "strategy": candidate,
                "action_index": index,
                "policy": "deterministic_baseline_retry",
            }
    return {**current, "strategy": "research_then_llm", "policy": "retry_budget_exhausted"}
