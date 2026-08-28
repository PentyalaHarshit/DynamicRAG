"""
Information Requirements Extractor & Information Graph
======================================================
Decomposes queries into structured Information Requirement nodes (1 to N sub-queries)
and manages semantic coverage tracking, evidence reuse, and multi-query aggregation.

Supports:
1. Arbitrary N-query decomposition (multi-sentence, conjunctions, list clauses)
2. Semantic deduplication & evidence reuse
3. Concept coverage verification (for verifier and DQN reward)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

# Re-use concept extraction from adaptive_retriever
from adaptive_retriever import _extract_concepts, _MULTI_CONCEPT_CONNECTORS


# ---------------------------------------------------------------------------
# Data structures: Information Requirement Graph
# ---------------------------------------------------------------------------

@dataclass
class Requirement:
    """Structured Information Requirement node representing a single sub-query."""
    id: str                             # e.g. "Q1", "Q2", "Q3"
    query_text: str                     # e.g. "What is the capital of USA?"
    entity: str                         # e.g. "USA"
    attribute: str                      # e.g. "capital", "population", "president"
    required_concepts: List[str] = field(default_factory=list)
    is_satisfied: bool = False
    observation: str = ""
    source_chunk: str = ""

    def __str__(self) -> str:
        ent_attr = f"{self.entity}.{self.attribute}" if (self.entity or self.attribute) else self.query_text[:30]
        return f"{self.id}({ent_attr})"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entity": self.entity,
            "attribute": self.attribute,
            "query_text": self.query_text,
            "is_satisfied": self.is_satisfied,
            "observation": self.observation,
        }


@dataclass
class InfoRequirements:
    """
    Full information requirement profile and dependency graph for a query.
    Produced once per query and threaded through the pipeline.
    """
    concepts: List[str]                            # required sub-topics
    complexity: str                                # SIMPLE | MODERATE | COMPLEX
    requires_multi_retrieval: bool                 # True when >= 2 sub-queries or >= 3 concepts
    coverage_threshold: float                      # min fraction of concepts that must appear in answer
    query_type: str                                # mirrors operation_pattern
    structured_requirements: List[Requirement] = field(default_factory=list)

    def __str__(self) -> str:
        req_str = f" | sub_queries={[str(r) for r in self.structured_requirements]}" if self.structured_requirements else ""
        return (
            f"complexity={self.complexity} | concepts({len(self.concepts)})={self.concepts} | "
            f"coverage_threshold={self.coverage_threshold:.2f} | "
            f"multi_retrieval={self.requires_multi_retrieval}{req_str}"
        )


# ---------------------------------------------------------------------------
# General N-Query Decomposition Engine
# ---------------------------------------------------------------------------

_DEPTH_THRESHOLD = {
    "CONCISE":       0.40,
    "BALANCED":      0.60,
    "DETAILED":      0.70,
    "VERY_DETAILED": 0.75,
}

_PATTERN_COMPLEXITY = {
    "COUNT":            "SIMPLE",
    "IDENTITY":         "SIMPLE",
    "FACTOID":          "SIMPLE",
    "DEFINITION":       "MODERATE",
    "PROCEDURE":        "MODERATE",
    "COMPARISON":       "MODERATE",
    "EXPLANATION":      "COMPLEX",
    "HISTORICAL_EVENT": "MODERATE",
    "SELECTION_RANKING": "SIMPLE",
    "TIME_CURRENT":     "SIMPLE",
    "CALCULATION":      "SIMPLE",
    "GENERAL":          "MODERATE",
}

_COMPLEXITY_LEVELS = ["SIMPLE", "MODERATE", "COMPLEX"]


def _max_complexity(a: str, b: str) -> str:
    return a if _COMPLEXITY_LEVELS.index(a) >= _COMPLEXITY_LEVELS.index(b) else b


def _complexity_from_concept_count(n: int) -> str:
    if n >= 4:
        return "COMPLEX"
    if n >= 2:
        return "MODERATE"
    return "SIMPLE"


def _extract_structured_requirements(question: str) -> List[Requirement]:
    """
    Decomposes any multi-sentence or multi-clause question into 1 to N distinct,
    independent sub-query requirement nodes (e.g. Q1, Q2, Q3, Q4...).
    """
    normalized = question.strip()
    # Normalize punctuation, conjunctions, and clause boundaries into delimiter '|'
    norm = re.sub(r'[,;?\n]+', '|', normalized)
    norm = re.sub(r'\b(?:and\s+also|as\s+well\s+as|along\s+with|and)\b', '|', norm, flags=re.IGNORECASE)
    norm = re.sub(r'\b(?=(?:what|who|which|where|when|why|how|explain|describe|tell|name)\b)', '|', norm, flags=re.IGNORECASE)
    norm = re.sub(r'\b(?=(?:its|their)\s+(?:capital|population|currency|president|prime\s+minister|gdp|founder|area|economy|leader|flag|language)\b)', '|', norm, flags=re.IGNORECASE)

    raw_clauses: List[str] = []
    for part in norm.split('|'):
        p_clean = part.strip().rstrip('?., ')
        if not p_clean or p_clean.lower() in ('and', 'its', 'their', 'the', 'is', 'what is', 'what'):
            continue
        if re.match(r'^(?:is|are|was|were)\s+', p_clean, re.IGNORECASE):
            p_clean = 'What ' + p_clean
        if len(p_clean) >= 2:
            raw_clauses.append(p_clean)

    if not raw_clauses:
        raw_clauses = [normalized]

    # Entity propagation across sequential sub-queries
    reqs: List[Requirement] = []
    current_entity = ""
    req_idx = 1
    seen_queries = set()

    for clause in raw_clauses:
        # Detect proper noun entity in this clause (excluding attribute keywords)
        words = [
            w for w in re.findall(r'\b[A-Z][a-zA-Z0-9-]+\b', clause)
            if w.lower() not in (
                'what', 'who', 'which', 'where', 'when', 'why', 'how', 'is', 'are',
                'was', 'were', 'the', 'a', 'an', 'and', 'or', 'if', 'tell', 'explain',
                'describe', 'find', 'calculate', 'in', 'of', 'for', 'with', 'state', 'area',
                'capital', 'population', 'president', 'economy', 'gdp', 'founder', 'currency',
                'its', 'their', 'nation'
            )
        ]
        if words:
            current_entity = " ".join(words)

        resolved_q = clause
        if current_entity:
            # Replace pronouns with entity
            if re.search(r'\b(?:its|their|the\s+country|the\s+nation)\b', resolved_q, re.IGNORECASE):
                resolved_q = re.sub(
                    r'\b(?:its|their|the\s+country\'?s?|the\s+nation\'?s?)\b',
                    f"{current_entity}'s",
                    resolved_q,
                    flags=re.IGNORECASE,
                )
            elif not any(w.lower() in resolved_q.lower() for w in current_entity.split()):
                if not re.search(r'\b(?:in|of|for)\s+' + re.escape(current_entity) + r'\b', resolved_q, re.IGNORECASE):
                    resolved_q = f"{resolved_q} of {current_entity}"

        # Detect target attribute focus
        attr = ""
        attr_candidates = [
            "capital", "population", "currency", "president", "prime minister", "gdp", "founder",
            "economy", "area", "largest state", "first law", "second law",
            "third law", "formula", "distance", "speed", "velocity", "acceleration"
        ]
        for ac in attr_candidates:
            if ac in resolved_q.lower():
                attr = ac
                break

        q_key = (current_entity.lower(), attr.lower()) if (current_entity and attr) else re.sub(r'[^a-z0-9]', '', resolved_q.lower())
        if q_key in seen_queries:
            continue
        seen_queries.add(q_key)

        reqs.append(
            Requirement(
                id=f"Q{req_idx}",
                query_text=resolved_q,
                entity=current_entity,
                attribute=attr,
                required_concepts=[attr] if attr else [w for w in resolved_q.split() if len(w) > 3][:3],
            )
        )
        req_idx += 1

    return reqs


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def extract_requirements(
    question: str,
    answer_style=None,
    operation_pattern: Optional[str] = None,
) -> InfoRequirements:
    """
    Derives the full information requirements and sub-query dependency graph for a query.
    """
    concepts = _extract_concepts(question)
    structured_reqs = _extract_structured_requirements(question)

    complexity = _complexity_from_concept_count(max(len(concepts), len(structured_reqs)))
    pat = (operation_pattern or "GENERAL").upper()
    pattern_complexity = _PATTERN_COMPLEXITY.get(pat, "MODERATE")
    complexity = _max_complexity(complexity, pattern_complexity)

    if _MULTI_CONCEPT_CONNECTORS.search(question) or len(structured_reqs) >= 2:
        complexity = _max_complexity(complexity, "MODERATE")

    depth = getattr(answer_style, "depth", "BALANCED") if answer_style else "BALANCED"
    coverage_threshold = _DEPTH_THRESHOLD.get(depth, 0.60)
    if complexity == "COMPLEX" and coverage_threshold < 0.75:
        coverage_threshold = 0.75

    requires_multi_retrieval = len(concepts) >= 3 or len(structured_reqs) >= 2

    req = InfoRequirements(
        concepts=concepts,
        complexity=complexity,
        requires_multi_retrieval=requires_multi_retrieval,
        coverage_threshold=coverage_threshold,
        query_type=pat,
        structured_requirements=structured_reqs,
    )

    print(f"[InfoRequirements] {req}")
    return req


# ---------------------------------------------------------------------------
# Coverage check (used by verifier and retrieval loop)
# ---------------------------------------------------------------------------

def check_concept_coverage(
    answer: str,
    requirements: InfoRequirements,
) -> dict:
    """
    Checks what fraction of required concepts appear in the generated answer.
    """
    if not requirements.concepts:
        return {
            "coverage_score": 1.0,
            "covered_concepts": [],
            "uncovered_concepts": [],
        }

    answer_lower = answer.lower()
    covered = []
    uncovered = []

    for concept in requirements.concepts:
        variants = [concept, concept.replace("-", " "), concept.replace(" ", "-")]
        if any(v in answer_lower for v in variants):
            covered.append(concept)
        else:
            uncovered.append(concept)

    total = len(requirements.concepts)
    coverage_score = len(covered) / total if total > 0 else 1.0

    return {
        "coverage_score": round(coverage_score, 3),
        "covered_concepts": covered,
        "uncovered_concepts": uncovered,
    }
