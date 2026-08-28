"""
Adaptive Query-Driven Retrieval Granularity
============================================

Derives the optimal retrieval granularity from *query meaning*, not query length.

Decision inputs:
  1. answer_style.depth  -- from answer_style_detector (CONCISE/BALANCED/DETAILED/VERY_DETAILED)
  2. operation_pattern   -- from problem_analyzer (FACTOID/DEFINITION/EXPLANATION/COUNT/...)
  3. concept_count       -- number of distinct sub-topics detected dynamically in the query

Granularity levels:
  SENTENCE      -- 30-80 words/unit  -- tight factoid, COUNT, simple identity
  PARAGRAPH     -- 150-300 words     -- focused DEFINITION, single concept
  SECTION       -- 400-700 words     -- DETAILED explanation, multi-aspect
  MULTI_SECTION -- 900-1500 words    -- COMPREHENSIVE, multi-concept research

No hardcoded vocabulary or keyword-to-chunk-size mapping.
The decision emerges dynamically from semantic intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# -- Granularity constants ----------------------------------------------------

SENTENCE      = "SENTENCE"
PARAGRAPH     = "PARAGRAPH"
SECTION       = "SECTION"
MULTI_SECTION = "MULTI_SECTION"

# (chunk_words, max_chunks, top_emb, top_ce)
_GRANULARITY_CONFIG = {
    SENTENCE:      (80,   3, 5, 3),
    PARAGRAPH:     (250,  3, 5, 3),
    SECTION:       (600,  4, 6, 4),
    MULTI_SECTION: (1200, 5, 8, 5),
}


# -- RetrievalSpec ------------------------------------------------------------

@dataclass
class RetrievalSpec:
    """
    Fully describes how the retrieval pipeline should fetch and slice documents
    for this specific query.
    """
    granularity: str          # SENTENCE | PARAGRAPH | SECTION | MULTI_SECTION
    chunk_words: int          # target words per retrieved unit
    max_chunks:  int          # maximum chunks to carry to the LLM
    top_emb:     int          # embedding filter top-k
    top_ce:      int          # cross-encoder rerank top-k
    concepts:    List[str] = field(default_factory=list)
    fusion_mode: str = "SINGLE"

    def __str__(self) -> str:
        return (
            f"granularity={self.granularity} | chunk_words={self.chunk_words} | "
            f"max_chunks={self.max_chunks} | concepts={self.concepts}"
        )


# -- Dynamic Domain-Agnostic Concept Extractor --------------------------------

_STOPWORDS = {
    "what", "is", "are", "was", "were", "how", "why", "when", "where", "who", "which",
    "explain", "describe", "detail", "detailed", "depth", "comprehensive",
    "overview", "give", "tell", "show", "define", "definition", "meaning",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "about", "and", "or", "as", "well", "such", "including", "along",
    "does", "do", "did", "can", "could", "would", "should", "works", "work",
    "differ", "difference", "between", "versus", "comparison", "compare",
    "step", "steps", "way", "ways", "method", "methods", "process", "system",
    "top", "best", "strongest", "most", "world", "explained", "tier",
}

_MULTI_CONCEPT_CONNECTORS = re.compile(
    r"\b(and|also|as well as|including|along with|"
    r"from .{1,30} to|how .{1,30} led to|evolution of|"
    r"both .{1,20} and|not only .{1,20} but)\b",
    re.IGNORECASE,
)


def _extract_concepts(question: str) -> List[str]:
    """
    Dynamic, domain-agnostic concept & sub-topic extractor.
    Extracts key concepts from ANY query (science, history, AI, finance, sports, etc.)
    without hardcoded vocabulary lists.
    """
    if not question:
        return []

    found: list[str] = []

    # 1. Quoted terms or phrases: "..." or '...'
    quotes = re.findall(r'["\']([^"\']+)["\']', question)
    for q in quotes:
        q_clean = q.strip().lower()
        if len(q_clean) >= 3 and q_clean not in found:
            found.append(q_clean)

    # 2. Comma / colon / semi-colon separated list items
    # E.g. "tokenization, embeddings, transformer architecture, self-attention"
    parts = re.split(r'[:,;]\s*|\s+(?:and|including|such as|like)\s+', question, flags=re.IGNORECASE)
    if len(parts) > 1:
        for p in parts:
            p_clean = re.sub(r'^(?:explain|describe|what is|how|about)\s+', '', p.strip(), flags=re.IGNORECASE)
            p_words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9-]+\b', p_clean) if w.lower() not in _STOPWORDS]
            if 1 <= len(p_words) <= 4:
                phrase = " ".join(p_words)
                if len(phrase) >= 3 and phrase not in found:
                    found.append(phrase)

    # 3. Capitalised multi-word proper nouns / technical terms
    # E.g. "Battle of Panipat", "Byte Pair Encoding", "Quantum Mechanics"
    caps = re.findall(r'\b(?:[A-Z][a-z0-9-]+(?:\s+(?:of|and|in|the)\s+)?)+[A-Z][a-z0-9-]+\b', question)
    for cap in caps:
        c_clean = cap.strip().lower()
        c_words = [w for w in c_clean.split() if w not in _STOPWORDS]
        if c_words and len(c_clean) >= 4 and c_clean not in found:
            found.append(c_clean)

    # 4. Cohesive non-stopword sequence
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9-]+\b', question) if w.lower() not in _STOPWORDS]
    if len(words) >= 2:
        full_phrase = " ".join(words)
        if len(full_phrase) >= 5 and full_phrase not in found:
            found.append(full_phrase)
    elif len(words) == 1:
        if len(words[0]) >= 3 and words[0] not in found:
            found.append(words[0])

    # 5. Deduplicate subsets: if "special forces" is present, discard plain "forces"
    deduped: list[str] = []
    for c in found:
        # Keep if not a strict substring of another longer concept in the list
        if not any(c != other and c in other for other in found):
            deduped.append(c)

    return deduped if deduped else found[:10]


# -- Core decision function ---------------------------------------------------

_DEPTH_GRANULARITY = {
    "CONCISE":       SENTENCE,
    "BALANCED":      PARAGRAPH,
    "DETAILED":      SECTION,
    "VERY_DETAILED": MULTI_SECTION,
}

_PATTERN_MIN_GRANULARITY = {
    "COUNT":            SENTENCE,
    "IDENTITY":         SENTENCE,
    "FACTOID":          SENTENCE,
    "DEFINITION":       PARAGRAPH,
    "PROCEDURE":        SECTION,
    "COMPARISON":       SECTION,
    "EXPLANATION":      SECTION,
    "HISTORICAL_EVENT": SECTION,
    "SELECTION_RANKING": PARAGRAPH,
    "TIME_CURRENT":     PARAGRAPH,
    "CALCULATION":      PARAGRAPH,
    "GENERAL":          PARAGRAPH,
}

_LEVELS = [SENTENCE, PARAGRAPH, SECTION, MULTI_SECTION]


def _max_granularity(a: str, b: str) -> str:
    return a if _LEVELS.index(a) >= _LEVELS.index(b) else b


def _concept_bump(granularity: str, concept_count: int) -> str:
    if concept_count >= 4:
        return _max_granularity(granularity, MULTI_SECTION)
    if concept_count >= 2:
        return _max_granularity(granularity, SECTION)
    return granularity


def _connector_bump(granularity: str, question: str) -> str:
    if _MULTI_CONCEPT_CONNECTORS.search(question):
        return _max_granularity(granularity, SECTION)
    return granularity


def derive_retrieval_spec(
    question: str,
    answer_style=None,
    operation_pattern: Optional[str] = None,
) -> RetrievalSpec:
    """
    Derives the optimal RetrievalSpec for this query dynamically from:
      - answer_style.depth
      - operation_pattern
      - detected dynamic concept count

    Args:
        question:          Raw user question.
        answer_style:      AnswerStyle dataclass instance (or None).
        operation_pattern: Pattern string e.g. "DEFINITION", "COUNT" (or None).

    Returns:
        RetrievalSpec with granularity, chunk_words, max_chunks, top_emb, top_ce.
    """
    # 1. Depth -> base granularity
    depth = getattr(answer_style, "depth", "BALANCED") if answer_style else "BALANCED"
    granularity = _DEPTH_GRANULARITY.get(depth, PARAGRAPH)

    # 2. Pattern minimum (never downgrade)
    pat = (operation_pattern or "GENERAL").upper()
    pattern_min = _PATTERN_MIN_GRANULARITY.get(pat, PARAGRAPH)
    granularity = _max_granularity(granularity, pattern_min)

    # 3. Dynamic concept count bump
    concepts = _extract_concepts(question)
    granularity = _concept_bump(granularity, len(concepts))

    # 4. Multi-concept connector bump
    granularity = _connector_bump(granularity, question)

    # 5. Resolve numeric parameters
    chunk_words, max_chunks, top_emb, top_ce = _GRANULARITY_CONFIG[granularity]

    spec = RetrievalSpec(
        granularity=granularity,
        chunk_words=chunk_words,
        max_chunks=max_chunks,
        top_emb=top_emb,
        top_ce=top_ce,
        concepts=concepts,
        fusion_mode="SINGLE",
    )

    print(
        f"[AdaptiveRetriever] {spec} | "
        f"depth={depth} | pattern={pat} | concept_count={len(concepts)}"
    )
    return spec
