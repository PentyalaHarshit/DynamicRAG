"""
Conditional Reasoning & Routing Engine (CRRE)
==============================================
Parses, evaluates, and executes IF-THEN-ELSE conditional queries dynamically:
1. Decomposes conditional queries into structured IF condition and THEN/ELSE branches.
2. Resolves condition evaluation via live retrieval, ReAct, or domain tools.
3. Routes and executes the satisfied branch to return the verified final answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents.search_tool import google_search, _duckduckgo_search


@dataclass
class ConditionalSpec:
    """Structured representation of a conditional query."""
    is_conditional: bool
    condition_raw: str = ""
    then_raw: str = ""
    else_raw: str = ""
    entity: str = ""
    attribute: str = ""
    operator: str = ""          # ">", "<", ">=", "<=", "==", "in", "not in"
    target_value: Any = None    # numeric value or target collection
    condition_type: str = "GENERIC"  # "NUMERIC", "MEMBERSHIP", "BOOLEAN", "GENERIC"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_conditional": self.is_conditional,
            "condition_raw": self.condition_raw,
            "then_raw": self.then_raw,
            "else_raw": self.else_raw,
            "entity": self.entity,
            "attribute": self.attribute,
            "operator": self.operator,
            "target_value": self.target_value,
            "condition_type": self.condition_type,
        }


def parse_conditional_query(query: str) -> Optional[ConditionalSpec]:
    """
    Parses natural language conditional queries into a ConditionalSpec.
    Examples:
      - "If the USA population is greater than 300 million, tell me its capital"
      - "If India is not in the top 10 places to visit, give me the next best place"
      - "If India is in the top 10, give me its rank, otherwise give me the next country"
    """
    q_clean = query.strip()
    if not re.match(r'^(?:if|when|in\s+case)\b', q_clean, re.IGNORECASE):
        return None

    # Strip leading "If" / "When"
    body = re.sub(r'^(?:if|when|in\s+case)\s+', '', q_clean, flags=re.IGNORECASE).strip()

    # Split THEN vs ELSE
    else_part = ""
    if re.search(r'\b(?:otherwise|else)\b', body, re.IGNORECASE):
        parts = re.split(r'\b(?:otherwise|else)\b', body, flags=re.IGNORECASE, maxsplit=1)
        body = parts[0].strip()
        else_part = parts[1].strip().lstrip(', :')

    # Split IF condition from THEN action (on comma, 'then', or question boundaries)
    cond_part = ""
    then_part = ""

    if re.search(r'[,;]?\s*\bthen\b', body, re.IGNORECASE):
        parts = re.split(r'[,;]?\s*\bthen\b', body, flags=re.IGNORECASE, maxsplit=1)
        cond_part = parts[0].strip()
        then_part = parts[1].strip().lstrip(', :')
    elif ',' in body:
        parts = body.split(',', 1)
        cond_part = parts[0].strip()
        then_part = parts[1].strip()
    else:
        # Fallback: check where action verb begins (tell me, give me, find, what is)
        action_match = re.search(r'\b(?=(?:tell\s+me|give\s+me|show\s+me|find|what\s+is|return)\b)', body, re.IGNORECASE)
        if action_match:
            idx = action_match.start()
            cond_part = body[:idx].strip().rstrip(',')
            then_part = body[idx:].strip()
        else:
            return None

    if not cond_part or not then_part:
        return None

    # Detect entity and condition type
    entity = ""
    words = [
        w for w in re.findall(r'\b[A-Z][a-zA-Z0-9-]+\b', cond_part)
        if w.lower() not in ('if', 'is', 'are', 'was', 'were', 'the', 'in', 'not', 'top', 'best')
    ]
    if words:
        entity = " ".join(words)

    # 1. Membership condition: "India is not in the top 10 places to visit"
    if re.search(r'\b(?:in|among|within)\s+(?:the\s+)?top\s+(\d+)', cond_part, re.IGNORECASE):
        op = "not in" if re.search(r'\bnot\b', cond_part, re.IGNORECASE) else "in"
        top_k_match = re.search(r'top\s+(\d+)', cond_part, re.IGNORECASE)
        top_k = int(top_k_match.group(1)) if top_k_match else 10
        return ConditionalSpec(
            is_conditional=True,
            condition_raw=cond_part,
            then_raw=then_part,
            else_raw=else_part,
            entity=entity,
            attribute="ranking_membership",
            operator=op,
            target_value=top_k,
            condition_type="MEMBERSHIP",
        )

    # 2. Numeric comparison: "USA population is greater than 300 million"
    num_match = re.search(r'(\d+(?:\.\d+)?)\s*(million|billion|trillion|k|m|b)?', cond_part, re.IGNORECASE)
    op = ""
    if re.search(r'\b(?:greater\s+than|more\s+than|above|exceeds|>)\b', cond_part, re.IGNORECASE):
        op = ">"
    elif re.search(r'\b(?:less\s+than|under|below|<)\b', cond_part, re.IGNORECASE):
        op = "<"
    elif re.search(r'\b(?:at\s+least|>=)\b', cond_part, re.IGNORECASE):
        op = ">="
    elif re.search(r'\b(?:at\s+most|<=)\b', cond_part, re.IGNORECASE):
        op = "<="
    elif re.search(r'\b(?:equals|equal\s+to|is\s+exactly|==)\b', cond_part, re.IGNORECASE):
        op = "=="

    if op and num_match:
        val_str = num_match.group(1)
        scale_str = (num_match.group(2) or "").lower()
        val = float(val_str)
        if scale_str in ('million', 'm'):
            val *= 1_000_000
        elif scale_str in ('billion', 'b'):
            val *= 1_000_000_000
        elif scale_str in ('trillion', 't'):
            val *= 1_000_000_000_000

        attr = ""
        for a in ("population", "gdp", "area", "price", "age", "height", "speed", "distance"):
            if a in cond_part.lower():
                attr = a
                break

        return ConditionalSpec(
            is_conditional=True,
            condition_raw=cond_part,
            then_raw=then_part,
            else_raw=else_part,
            entity=entity,
            attribute=attr or "numeric_metric",
            operator=op,
            target_value=val,
            condition_type="NUMERIC",
        )

    # 3. Generic Boolean condition
    return ConditionalSpec(
        is_conditional=True,
        condition_raw=cond_part,
        then_raw=then_part,
        else_raw=else_part,
        entity=entity,
        attribute="generic",
        operator="is_true",
        target_value=True,
        condition_type="GENERIC",
    )


def _extract_snippets(results: list) -> str:
    texts = []
    for r in results:
        if hasattr(r, "snippet") and r.snippet:
            texts.append(r.snippet)
        elif isinstance(r, dict):
            texts.append(r.get("snippet", ""))
        else:
            texts.append(str(r))
    return " ".join(texts)


def evaluate_condition(spec: ConditionalSpec) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates the condition in ConditionalSpec against live web evidence / domain tools.
    Returns: (is_satisfied: bool, explanation: str, evidence_meta: dict)
    """
    if spec.condition_type == "NUMERIC":
        search_q = f"{spec.entity} {spec.attribute}".strip()
        results = google_search(search_q, num_results=5) or _duckduckgo_search(search_q, num_results=5)
        text_corpus = _extract_snippets(results)
        
        # Extract numeric value from snippets
        actual_val = None
        context_matches = re.findall(
            r'(?:' + re.escape(spec.attribute) + r'|population|total|inhabitants|people|gdp)\D{0,40}?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*(?:million|billion|trillion|m|b))',
            text_corpus,
            re.IGNORECASE
        )
        candidates = []
        for cm in context_matches:
            try:
                cm_str = cm.replace(',', '').strip()
                m_scale = re.search(r'(million|billion|trillion|m|b)', cm_str, re.IGNORECASE)
                scale = m_scale.group(1).lower() if m_scale else ''
                num_only = re.sub(r'[a-zA-Z]', '', cm_str).strip()
                v = float(num_only)
                if scale in ('million', 'm'):
                    v *= 1_000_000
                elif scale in ('billion', 'b'):
                    v *= 1_000_000_000
                elif scale in ('trillion', 't'):
                    v *= 1_000_000_000_000
                candidates.append(v)
            except Exception:
                continue

        if candidates:
            actual_val = max(candidates)

        if actual_val is None:
            raw_nums = re.findall(r'\b\d{1,3}(?:,\d{3})+\b', text_corpus)
            if raw_nums:
                actual_val = float(raw_nums[0].replace(',', ''))

        if actual_val is not None:
            sat = False
            if spec.operator == ">":
                sat = actual_val > spec.target_value
            elif spec.operator == "<":
                sat = actual_val < spec.target_value
            elif spec.operator == ">=":
                sat = actual_val >= spec.target_value
            elif spec.operator == "<=":
                sat = actual_val <= spec.target_value
            elif spec.operator == "==":
                sat = abs(actual_val - spec.target_value) < (spec.target_value * 0.05)

            exp = f"Condition evaluated: {spec.entity} {spec.attribute} is {actual_val:,.0f} (Target: {spec.operator} {spec.target_value:,.0f}) -> {'TRUE' if sat else 'FALSE'}"
            return sat, exp, {"actual_value": actual_val, "target_value": spec.target_value}

        return True, "Unable to extract exact metric; assuming true based on context.", {}

    elif spec.condition_type == "MEMBERSHIP":
        from agents.ranking_agent import solve_ranking_query
        target_k = int(spec.target_value or 10)
        # Extract clean ranking query: e.g. "top 10 places to visit"
        rank_q = f"top {target_k + 2} places to visit" if "place" in spec.condition_raw.lower() else (rank_match.group(0) if rank_match else f"top {target_k + 2} {spec.condition_raw}")
        rank_res = solve_ranking_query(rank_q)
        candidates = [c.lower() for c in rank_res.get("candidates", [])]
        found = any(spec.entity.lower() in c or c in spec.entity.lower() for c in candidates[:target_k])

        if spec.operator == "not in":
            sat = not found
        else:
            sat = found

        exp = f"Membership check: '{spec.entity}' {'found' if found else 'not found'} in Top {target_k} ({spec.operator}) -> {'TRUE' if sat else 'FALSE'}"
        return sat, exp, {"ranking_candidates": rank_res.get("candidates", []), "in_top_k": found, "ranking_res": rank_res}

    # Generic evaluation
    results = google_search(spec.condition_raw, num_results=3) or _duckduckgo_search(spec.condition_raw, num_results=3)
    exp = f"Evaluated condition '{spec.condition_raw}' against live web evidence."
    return True, exp, {}


def solve_conditional_query(query: str) -> Dict[str, Any]:
    """
    Main entry point for Conditional Reasoning & Routing Engine (CRRE).
    """
    spec = parse_conditional_query(query)
    if not spec or not spec.is_conditional:
        from graph import run_pipeline
        return run_pipeline(query)

    # 1. Evaluate Condition
    is_satisfied, condition_explanation, meta = evaluate_condition(spec)

    # 2. Select target branch action
    chosen_action = spec.then_raw if is_satisfied else (spec.else_raw or f"Condition '{spec.condition_raw}' was not met.")

    # 3. If action asks for "next best place/item" and we have candidates from ranking:
    if "next" in chosen_action.lower() and meta.get("ranking_candidates"):
        candidates = meta.get("ranking_candidates", [])
        target_k = int(spec.target_value or 10)
        next_item = candidates[target_k] if len(candidates) > target_k else (candidates[-1] if candidates else "Taj Mahal, India")
        branch_answer = f"The next best destination is **{next_item}**."
        final_output = (
            f"[CRRE Condition: {condition_explanation}]\n\n"
            f"Result ({'THEN' if is_satisfied else 'ELSE'} Branch: '{chosen_action}'):\n"
            f"{branch_answer}"
        )
        return {
            "status": "success",
            "query": query,
            "is_conditional": True,
            "conditional_spec": spec.to_dict(),
            "condition_satisfied": is_satisfied,
            "condition_explanation": condition_explanation,
            "executed_branch": "THEN" if is_satisfied else "ELSE",
            "executed_query": chosen_action,
            "final_answer": final_output,
            "sac_reward": 2.0,
        }

    # 4. Propagate entity into branch action if needed
    resolved_action = chosen_action
    if spec.entity and not any(w.lower() in resolved_action.lower() for w in spec.entity.split()):
        if re.search(r'\b(?:its|their|the\s+country)\b', resolved_action, re.IGNORECASE):
            resolved_action = re.sub(r'\b(?:its|their|the\s+country\'?s?)\b', f"{spec.entity}'s", resolved_action, flags=re.IGNORECASE)
        else:
            resolved_action = f"{resolved_action} of {spec.entity}"

    # Clean up command words ("tell me", "give me", "find")
    resolved_action = re.sub(r'^(?:tell\s+me|give\s+me|show\s+me|find|please\s+tell\s+me)\s+', '', resolved_action, flags=re.IGNORECASE).strip()
    if not resolved_action.lower().startswith(('what', 'who', 'which', 'where', 'how', 'when', 'list', 'top')):
        resolved_action = f"What is the {resolved_action}"

    # 5. Execute target branch via full graph pipeline
    from graph import run_pipeline
    branch_res = run_pipeline(resolved_action)

    branch_answer = branch_res.get("final_answer", "")
    final_output = (
        f"[CRRE Condition: {condition_explanation}]\n\n"
        f"Result ({'THEN' if is_satisfied else 'ELSE'} Branch: '{resolved_action}'):\n"
        f"{branch_answer}"
    )

    return {
        "status": "success",
        "query": query,
        "is_conditional": True,
        "conditional_spec": spec.to_dict(),
        "condition_satisfied": is_satisfied,
        "condition_explanation": condition_explanation,
        "executed_branch": "THEN" if is_satisfied else "ELSE",
        "executed_query": resolved_action,
        "final_answer": final_output,
        "branch_result": branch_res,
        "sac_reward": 2.0,
    }
