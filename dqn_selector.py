"""
Rich DQN Chunk Selection Module with Two-Gate Evidence Architecture:

Retrieved Context
        │
        ▼
Topic Relevance Gate  (keyword overlap >= threshold)
        │  FAIL -> chunk rejected, try next best
        ▼
Answer Evidence Gate  (typed entity found in chunk)
        │  FAIL -> signal answerability failure upstream
        ▼
DQN Selection
        │
        ▼
LLM Generation

10-dimensional rich state vector:
  0. embedding_similarity  - Cosine similarity score (0.0 – 1.0)
  1. cross_encoder_score   - CrossEncoder model score, normalised (0.0 – 1.0)
  2. retrieval_rank        - 1.0 / rank_position
  3. chunk_length          - normalised char length, capped at 500
  4. topic_match           - keyword overlap between query and chunk (0.0 – 1.0)
                             Pure overlap — no discovery-term bonus.
                             Used exclusively by Topic Relevance Gate.
  5. source_score          - domain credibility (0.0 – 1.0)
  6. answer_entity_found   - 1.0 if the TYPED answer entity is present; 0.0 otherwise
                             (PERSON for WHO, DATE for WHEN, NUMBER for HOW MANY, etc.)
  7. person_entity_found   - 1.0 if any PERSON name is in the chunk; 0.0 otherwise
                             Always computed regardless of query type.
  8. answerability_score   - composite:
                               0.95 typed entity found
                               0.5  entity type undetermined (NONE)
                               0.1  expected entity absent
  9. date_found            - 1.0 if any date/year string is present in the chunk.
                             Distinct from answer_entity_found; provides an explicit
                             temporal evidence signal for HISTORICAL_FACT / DATE queries.

question_entity_type is also exposed in the state dict (string, not in vector):
  e.g. "PERSON" for WHO queries, "DATE" for WHEN, "NUMBER" for HOW MANY,
       "LOCATION" for WHERE, "NONE" if type could not be determined.

W2 weight rationale (hidden_dim x 1):
  Cross-encoder score (0.27) + answerability_score (0.23) dominate.
  answer_entity_found (0.19) and person_entity_found (0.15) are strong secondary
  signals.  date_found (0.06) gives explicit temporal evidence credit.
  topic_match (0.07) and source_score (0.09) are supporting.
  embedding_similarity (0.09), rank (0.04), length (0.03) are weak priors.
  (All 10 weights sum to ~1.22 before hidden-layer mixing.)
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import IntEnum
import numpy as np

from answerability_agent import _extract_persons, _extract_entities, _expected_entity_type, _extract_dates


# ---------------------------------------------------------------------------
# Retrieval Action Space (for multi-concept queries)
# ---------------------------------------------------------------------------

class RetrievalAction(IntEnum):
    """
    Actions the DQN retrieval agent can choose when coverage is insufficient.
    Only active when requires_multi_retrieval=True.
    """
    RETRIEVE_PARAGRAPH = 0   # tight paragraph retrieval (default for simple queries)
    EXPAND_CONTEXT     = 1   # include surrounding paragraphs for more context
    RETRIEVE_SECTION   = 2   # full section-level retrieval
    RETRIEVE_RELATED   = 3   # fetch a related concept section (cross-topic)
    STOP               = 4   # evidence sufficient, proceed to generation


_ACTION_NAMES = {
    RetrievalAction.RETRIEVE_PARAGRAPH: "retrieve_paragraph",
    RetrievalAction.EXPAND_CONTEXT:     "expand_context",
    RetrievalAction.RETRIEVE_SECTION:   "retrieve_section",
    RetrievalAction.RETRIEVE_RELATED:   "retrieve_related",
    RetrievalAction.STOP:               "stop",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RichState:
    embedding_similarity: float
    cross_encoder_score: float   # normalised to [0, 1]
    retrieval_rank: int
    chunk_length: int
    topic_match: float           # keyword overlap + entity bonus
    source_score: float
    question_entity_type: str    # PERSON / DATE / NUMBER / LOCATION / NONE
    answer_entity_found: float   # 1.0 / 0.0
    person_entity_found: float   # 1.0 / 0.0
    answerability_score: float   # 0.95 / 0.5 / 0.1
    date_found: float            # 1.0 if any date/year present in chunk
    concept_coverage: float = 0.0   # dim 10: fraction of required concepts covered so far
    uncovered_ratio: float = 1.0    # dim 11: fraction of concepts still missing

    def to_vector(self) -> List[float]:
        # question_entity_type is a string — excluded from the numeric vector
        return [
            float(self.embedding_similarity),        # dim 0
            float(self.cross_encoder_score),         # dim 1
            1.0 / float(self.retrieval_rank),        # dim 2
            min(1.0, self.chunk_length / 500.0),     # dim 3
            float(self.topic_match),                 # dim 4
            float(self.source_score),                # dim 5
            float(self.answer_entity_found),         # dim 6
            float(self.person_entity_found),         # dim 7
            float(self.answerability_score),         # dim 8
            float(self.date_found),                  # dim 9
            float(self.concept_coverage),            # dim 10 (NEW: coverage ratio)
            float(self.uncovered_ratio),             # dim 11 (NEW: uncovered ratio)
        ]


@dataclass
class RichDQNResult:
    selected_chunk: str
    selected_index: int
    q_values: List[float]
    rich_states: List[Dict[str, Any]]
    confidence: float
    evidence_gate_passed: bool
    topic_gate_passed: bool       # NEW: indicates whether Topic Relevance Gate passed


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class RichDQNChunkSelector:
    # Topic Relevance Gate: chunk must overlap this fraction of query keywords
    TOPIC_THRESHOLD: float = 0.15
    # Answer Evidence Gate: chunk must reach this answerability_score
    EVIDENCE_THRESHOLD: float = 0.5

    def __init__(self, state_dim: int = 12, hidden_dim: int = 16):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        np.random.seed(42)
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)

        # W2 shape: (hidden_dim, 1)
        # Dims 0-9: original 10 dims. Dims 10-11: coverage signals.
        # concept_coverage (0.12): reward signal for multi-concept completeness.
        # uncovered_ratio (0.08): penalty signal for missing concepts.
        self.W2 = np.array(
            [0.08, 0.24, 0.04, 0.03, 0.06, 0.08, 0.17, 0.13, 0.20, 0.05, 0.12, 0.08]
            + [0.0] * (hidden_dim - 12)
        ).reshape(hidden_dim, 1)
        self.b2 = np.array([0.05])

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def build_rich_states(
        self,
        query: str,
        chunks: List[str],
        embedding_scores: List[float],
        cross_encoder_scores: List[float],
        sources: Optional[List[str]] = None,
        intent_type: str = "FACTOID",
    ) -> List[RichState]:
        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        num_chunks = len(chunks)
        sources = sources or ["web"] * num_chunks

        # Determine expected answer entity type once for the whole batch
        expected_type = _expected_entity_type(query)
        type_unknown = (expected_type == "NONE")
        # Normalise cross-encoder scores to [0, 1]
        ce_arr = np.array(cross_encoder_scores, dtype=float)
        ce_min, ce_max = ce_arr.min(), ce_arr.max()
        ce_norm = (
            ((ce_arr - ce_min) / (ce_max - ce_min)).tolist()
            if ce_max > ce_min
            else [0.5] * num_chunks
        )

        states: List[RichState] = []
        for i, (chunk, emb_s, _ce_raw, ce_s_norm, src) in enumerate(
            zip(chunks, embedding_scores, cross_encoder_scores, ce_norm, sources)
        ):
            # ── Topic Match: keyword overlap + entity-presence bonus ──────
            # Pure word overlap undersells chunks that contain the answer entity.
            # Example: "who is president of India" has 3 content words (president,
            # india, who).  A chunk containing "Incumbent Droupadi Murmu, President
            # of India" matches all three -> overlap = 1.0 already.  But a description
            # chunk "The President of India is the constitutional head..." matches
            # president + india = 0.67 with no entity.  Adding the entity bonus
            # correctly separates these two cases.
            #
            # Formula:
            #   base   = overlap of query content-words in chunk
            #   bonus  = +0.30 if the typed answer entity is found in this chunk
            #             (so a chunk with the actual answer scores at least 0.30
            #              even if keyword overlap is low, and up to 1.0)
            #   topic_match = min(1.0, base + bonus)
            chunk_words = set(w.lower() for w in chunk.split())
            base_overlap = (
                sum(1 for w in query_words if w in chunk_words)
                / max(1, len(query_words))
            )
            # entity bonus: reuse the already-computed answer_entity_found flag
            # (computed below) — we need to compute entities first, then bonus.
            # We handle this by computing entities before topic_match is finalised.

            # ── Source credibility ─────────────────────────────────────────
            src_lower = src.lower()
            src_score = (
                0.96 if ("wikipedia" in src_lower or "kb" in src_lower or "memory" in src_lower)
                else 0.78
            )

            # ── Person entity (always extracted) ──────────────────────────
            persons_in_chunk = _extract_persons(chunk)
            person_entity_found = 1.0 if persons_in_chunk else 0.0

            # ── Typed answer entity ────────────────────────────────────────
            if type_unknown:
                # Can't determine what to look for — don't block on this
                answer_entity_found = 1.0
            else:
                typed_entities = _extract_entities(chunk, expected_type)
                answer_entity_found = 1.0 if typed_entities else 0.0

            # ── Topic Match: finalise with entity-presence bonus ───────────
            # A chunk that actually contains the answer entity is, by definition,
            # more topically relevant than one that merely mentions the subject.
            # Bonus = 0.30 when the typed entity is confirmed present.
            # type_unknown -> no typed entity check -> no bonus (base only).
            entity_bonus = 0.30 if (not type_unknown and answer_entity_found == 1.0) else 0.0
            topic_match = min(1.0, base_overlap + entity_bonus)

            # ── Answerability score (composite) ───────────────────────────
            if type_unknown:
                answerability_score = 0.5    # neutral: type undetermined
            elif answer_entity_found == 1.0:
                answerability_score = 0.95   # typed entity confirmed in chunk
            else:
                answerability_score = 0.1    # expected entity absent

            # ── Date found (always extracted, independent of expected entity type) ─────
            # date_found is a standalone dim: gives explicit temporal evidence credit
            # even when the expected entity type is not DATE (e.g. a PERSON chunk that
            # also contains a year gets credit for temporal grounding).
            dates_in_chunk = _extract_dates(chunk)
            date_found = 1.0 if dates_in_chunk else 0.0

            print(
                f"[DQN State] chunk {i}: "
                f"topic_match={topic_match:.2f} | "
                f"answer_entity_found={answer_entity_found:.1f} | "
                f"person_entity_found={person_entity_found:.1f} | "
                f"date_found={date_found:.1f} | "
                f"answerability_score={answerability_score:.2f}"
            )

            states.append(RichState(
                embedding_similarity=float(emb_s),
                cross_encoder_score=float(ce_s_norm),
                retrieval_rank=i + 1,
                chunk_length=len(chunk),
                topic_match=float(topic_match),
                source_score=src_score,
                question_entity_type=expected_type,
                answer_entity_found=answer_entity_found,
                person_entity_found=person_entity_found,
                answerability_score=answerability_score,
                date_found=date_found,
            ))

        return states

    # ------------------------------------------------------------------
    # DQN forward pass
    # ------------------------------------------------------------------

    def _compute_combined_scores(
        self,
        rich_states: List[RichState],
        cross_encoder_scores: List[float],
    ) -> np.ndarray:
        """Returns combined DQN + cross-encoder scores for each chunk."""
        state_vectors = np.array(
            [s.to_vector() for s in rich_states], dtype=np.float32
        )
        hidden = np.maximum(0, state_vectors @ self.W1 + self.b1)
        q_vals = (hidden @ self.W2).flatten() + self.b2[0]

        ce_arr = np.array(cross_encoder_scores, dtype=float)
        ce_min, ce_max = ce_arr.min(), ce_arr.max()
        ce_norm = (
            (ce_arr - ce_min) / (ce_max - ce_min)
            if ce_max > ce_min
            else np.full_like(ce_arr, 0.5)
        )
        return q_vals + ce_norm

    # ------------------------------------------------------------------
    # Two-gate selection
    # ------------------------------------------------------------------

    def select_top1(
        self,
        query: str,
        chunks: List[str],
        embedding_scores: List[float],
        cross_encoder_scores: List[float],
        sources: Optional[List[str]] = None,
        evidence_threshold: float = None,   # kept for API compat; ignored (use class const)
        intent_type: str = "FACTOID",
    ) -> RichDQNResult:
        if not chunks:
            raise ValueError("No candidate chunks provided for Rich DQN selection.")

        rich_states = self.build_rich_states(
            query, chunks, embedding_scores, cross_encoder_scores, sources, intent_type
        )
        combined_scores = self._compute_combined_scores(rich_states, cross_encoder_scores)

        # ── Gate 1: Topic Relevance ────────────────────────────────────────
        # Require minimum keyword overlap so we never pass a completely off-topic
        # chunk to the LLM.  Chunks that fail are deprioritised (not hard-blocked
        # here — we still pick the best available if ALL fail).
        topic_passing = [
            i for i, s in enumerate(rich_states)
            if s.topic_match >= self.TOPIC_THRESHOLD
        ]
        if topic_passing:
            topic_gate_passed = True
            candidate_pool = topic_passing
            print(
                f"[DQN Topic Gate] {len(topic_passing)}/{len(chunks)} chunks passed "
                f"topic_match >= {self.TOPIC_THRESHOLD}"
            )
        else:
            # All chunks failed topic gate — use all as fallback, flag it
            topic_gate_passed = False
            candidate_pool = list(range(len(chunks)))
            print(
                f"[DQN Topic Gate] FAILED: no chunk reached topic_match >= "
                f"{self.TOPIC_THRESHOLD}. Using all chunks as fallback."
            )

        # ── Gate 2: Answer Evidence ────────────────────────────────────────
        # Within the topic-passing pool, require the typed answer entity to be
        # present (answerability_score >= 0.5).
        evidence_passing = [
            i for i in candidate_pool
            if rich_states[i].answerability_score >= self.EVIDENCE_THRESHOLD
        ]
        if evidence_passing:
            evidence_gate_passed = True
            final_pool = evidence_passing
            print(
                f"[DQN Evidence Gate] {len(evidence_passing)} chunks passed "
                f"answerability_score >= {self.EVIDENCE_THRESHOLD}"
            )
        else:
            # No chunk has the required entity — gate failed, pick best from topic pool
            evidence_gate_passed = False
            final_pool = candidate_pool
            print(
                f"[DQN Evidence Gate] FAILED: no chunk in topic pool has "
                f"answerability_score >= {self.EVIDENCE_THRESHOLD}. "
                f"Signalling evidence failure."
            )

        # ── DQN picks highest combined score within the surviving pool ─────
        best_idx = max(final_pool, key=lambda i: combined_scores[i])

        state_dicts = [
            {
                "embedding_similarity": round(s.embedding_similarity, 3),
                "cross_encoder_score":  round(s.cross_encoder_score, 3),
                "retrieval_rank":        s.retrieval_rank,
                "chunk_length":          s.chunk_length,
                "topic_match":           round(s.topic_match, 3),
                "source_score":          round(s.source_score, 3),
                "question_entity_type":  s.question_entity_type,
                "answer_entity_found":   round(s.answer_entity_found, 3),
                "person_entity_found":   round(s.person_entity_found, 3),
                "date_found":            round(s.date_found, 3),
                "answerability_score":   round(s.answerability_score, 3),
                "concept_coverage":      round(s.concept_coverage, 3),
                "uncovered_ratio":       round(s.uncovered_ratio, 3),
            }
            for s in rich_states
        ]

        return RichDQNResult(
            selected_chunk=chunks[best_idx],
            selected_index=best_idx,
            q_values=[round(float(q), 3) for q in combined_scores],
            rich_states=state_dicts,
            confidence=float(combined_scores[best_idx]),
            evidence_gate_passed=evidence_gate_passed,
            topic_gate_passed=topic_gate_passed,
        )


# ---------------------------------------------------------------------------
# Module-level singleton + public API
# ---------------------------------------------------------------------------

_rich_dqn_selector = RichDQNChunkSelector()


def select_retrieval_action(
    coverage_score: float,
    uncovered_concepts: list,
    requires_multi_retrieval: bool,
    coverage_threshold: float = 0.75,
) -> RetrievalAction:
    """
    Selects the next retrieval action based on current coverage state.
    Only meaningful for multi-concept queries (requires_multi_retrieval=True).
    For simple queries, always returns STOP (chunk selection path is used instead).

    Action selection heuristic:
      coverage >= threshold          → STOP (sufficient evidence)
      uncovered >= 4 concepts        → RETRIEVE_SECTION (need broad context)
      uncovered >= 2 concepts        → EXPAND_CONTEXT (expand current evidence)
      uncovered == 1 concept         → RETRIEVE_RELATED (fetch specific related section)
      coverage ~threshold (< 0.15)   → RETRIEVE_PARAGRAPH (targeted tight retrieval)
    """
    if not requires_multi_retrieval:
        return RetrievalAction.STOP

    if coverage_score >= coverage_threshold:
        action = RetrievalAction.STOP
    elif len(uncovered_concepts) >= 4:
        action = RetrievalAction.RETRIEVE_SECTION
    elif len(uncovered_concepts) >= 2:
        action = RetrievalAction.EXPAND_CONTEXT
    elif len(uncovered_concepts) == 1:
        action = RetrievalAction.RETRIEVE_RELATED
    else:
        action = RetrievalAction.RETRIEVE_PARAGRAPH

    print(
        f"[DQN RetrievalAction] {_ACTION_NAMES[action]} | "
        f"coverage={coverage_score:.2f} | uncovered={uncovered_concepts}"
    )
    return action


def select_top1_rich_dqn(
    query: str,
    chunks: List[str],
    embedding_scores: List[float],
    cross_encoder_scores: List[float],
    sources: Optional[List[str]] = None,
    intent_type: str = "FACTOID",
) -> RichDQNResult:
    return _rich_dqn_selector.select_top1(
        query, chunks, embedding_scores, cross_encoder_scores, sources,
        intent_type=intent_type,
    )
