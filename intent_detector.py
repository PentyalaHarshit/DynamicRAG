"""
Intent Detection (NLP) Module
==============================
Analyses incoming user queries to determine intent type, routing strategy
(Traditional RAG vs Web RAG), and confidence.

Taxonomy (9 types)
------------------
CURRENT_FACT    — who currently holds a role, live prices, incumbent data,
                  recent events.  Always needs web.
HISTORICAL_FACT — specific past events, wars, treaties, revolutions,
                  inventions with a known year, historical milestones.
                  Stable knowledge — no web needed.
FACTOID         — short factual lookups not requiring live data: capitals,
                  distances, record holders, geographical measurements.
BIOGRAPHY       — questions about a specific person's life, career, works,
                  achievements (past tense focus).
DEFINITION     — "what is X", "define X", "what does X mean".
COMPARISON     — "difference between A and B", "A vs B", "compare X and Y".
CODING         — write/explain/debug code, algorithms, data structures.
MATH           — arithmetic, equations, proofs, numerical reasoning.
REASONING      — multi-step logic, hypotheticals, causal analysis,
                 "why" questions requiring inference.

Routing guidance
-----------------
CURRENT_FACT    -> web (data changes over time)
HISTORICAL_FACT -> traditional RAG (stable historical knowledge)
FACTOID         -> traditional RAG first, web fallback
BIOGRAPHY       -> traditional RAG first, web fallback
DEFINITION     -> traditional RAG (stable knowledge)
COMPARISON     -> traditional RAG first
CODING         -> traditional RAG / LLM-only (no web needed in most cases)
MATH           -> LLM-only
REASONING      -> LLM-only

Fast-path heuristics (no LLM call needed)
-------------------------------------------
Order matters: CURRENT_FACT is tested first so "who is president of X" never
falls through to the BIOGRAPHY/FACTOID branch.
HISTORICAL_FACT is tested before BIOGRAPHY and FACTOID so that event-based
queries like "When was World War II?" are not misclassified.
"""
import json
import re
from dataclasses import dataclass
from typing import List

from llm_client import call_llm


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    intent_type: str    # one of the 8 types above
    needs_web: bool
    confidence: float
    keywords: List[str]
    reasoning: str


# ---------------------------------------------------------------------------
# Heuristic fast-path patterns
# (evaluated top-to-bottom; first match wins)
# ---------------------------------------------------------------------------

# ── CURRENT_FACT ────────────────────────────────────────────────────────────
# Catches:  "who is the president/PM/CEO/head/leader of X"
#           "who are the current …"
#           temporal markers: current, now, today, latest, recent, incumbent
#           year markers: 2024, 2025, 2026
#           price/rate queries
_CURRENT_FACT_RE = re.compile(
    r'\b('
    # role-holder queries
    r'who\s+(is|are)\s+(the\s+)?(current\s+|incumbent\s+|present\s+|new\s+)?'
    r'(president|prime\s+minister|minister|chancellor|governor|senator|'
    r'secretary|director|chairman|ceo|cfo|cto|head|leader|king|queen|'
    r'emperor|pope|patriarch|mayor|ambassador|speaker|treasurer|'
    r'commander|superintendent|commissioner|chief)'
    r'|'
    # temporal / liveness markers
    r'current(ly)?|incumbent|right\s+now|\bnow\b|today\b|'
    r'latest|recent(ly)?|as\s+of\s+(now|today|2024|2025|2026)|'
    r'\b(2024|2025|2026)\b'
    r'|'
    # price / rate queries
    r'what\s+is\s+the\s+(current\s+)?(price|rate|cost|value|exchange\s+rate)'
    r'|how\s+much\s+(is|does|do|did)\s+.{0,40}(cost|worth|price)'
    r')',
    re.IGNORECASE,
)

# ── MATH ────────────────────────────────────────────────────────────────────
_MATH_RE = re.compile(
    r'\b(calculate|compute|solve|integrate|differentiate|derivative|'
    r'equation|formula|arithmetic|algebra|geometry|trigonometry|'
    r'probability|statistics|matrix|vector|eigenvalue|proof|'
    r'what\s+is\s+\d[\d\s\+\-\*/\^]*=|'
    r'\d+\s*[\+\-\*\/\^]\s*\d+)\b',
    re.IGNORECASE,
)

# ── CODING ───────────────────────────────────────────────────────────────────
_CODING_RE = re.compile(
    r'\b(code|program|script|function|class|method|algorithm|implement|debug|'
    r'error\s+in\s+(my\s+)?(code|script|program)|'
    r'write\s+(a\s+)?(function|class|script|program|code)|'
    r'how\s+to\s+(implement|code|write|use|call|import|install)|'
    r'python|javascript|typescript|java|c\+\+|rust|golang|sql|bash|'
    r'api|library|framework|package|module|syntax|runtime|compile|'
    r'stack\s+overflow|git\s+|docker|kubernetes)\b',
    re.IGNORECASE,
)

# ── COMPARISON ───────────────────────────────────────────────────────────────
_COMPARISON_RE = re.compile(
    r'\b(difference\s+between|compare\s+|versus\s+|\bvs\.?\b|'
    r'better\s+(than|between)|which\s+(is|are)\s+(better|faster|cheaper|'
    r'more|less|worse)|pros\s+and\s+cons|advantages?\s+(of|and)|'
    r'disadvantages?\s+(of|and))\b',
    re.IGNORECASE,
)

# ── DEFINITION ───────────────────────────────────────────────────────────────
_DEFINITION_RE = re.compile(
    r'\b(what\s+is\s+(a\s+|an\s+|the\s+)?(?!price|rate|cost|value)'
    r'|define\s+|definition\s+of|what\s+does\s+\w+\s+mean|'
    r'meaning\s+of|explain\s+(what|the\s+concept)|'
    r'what\s+are\s+(the\s+)?(types|kinds|categories|examples)\s+of)\b',
    re.IGNORECASE,
)

# ── HISTORICAL_FACT ──────────────────────────────────────────────────────────
# Named historical events, wars, treaties, revolutions, discoveries with dates.
# Examples: "When was World War II?", "When did the French Revolution start?",
#           "What happened during the Cold War?", "When was the moon landing?"
# Checked BEFORE BIOGRAPHY so event-based queries don't fall into person queries.
_HISTORICAL_FACT_RE = re.compile(
    r'\b('
    # Temporal questions about named events
    r'when\s+(did|was|were|is)\s+.{0,60}(war|revolution|battle|siege|'
    r'treaty|independence|founded|established|discovered|invented|'
    r'signed|declared|ended|began|started|occurred|happened|took\s+place)'
    r'|'
    # Named historical conflicts / events
    r'(world\s+war|civil\s+war|cold\s+war|french\s+revolution|'
    r'industrial\s+revolution|renaissance|enlightenment|crusade|'
    r'holocaust|genocide|empire\s+(rise|fall)|fall\s+of|'
    r'american\s+revolution|russian\s+revolution|french\s+revolution|'
    r'partition\s+of|independence\s+of|coloniali[sz]ation|'
    r'moon\s+landing|space\s+race|manhattan\s+project)'
    r'|'
    # "What happened during/after/before [event]?"
    r'what\s+happened\s+(during|after|before|in)\s+.{0,50}'
    r'(war|revolution|battle|crisis|period|era|age|century|decade)'
    r'|'
    # Historical time-period queries
    r'\b(ancient|medieval|colonial|victorian|renaissance|cold\s+war|'
    r'post\s*war|pre\s*war|interwar)\s+(period|era|history|times?)'
    r')',
    re.IGNORECASE,
)

# ── BIOGRAPHY ────────────────────────────────────────────────────────────────
# "Who invented/founded/created/wrote/discovered X"
# "Tell me about [person]", "biography of [person]"
# Explicitly excludes current role-holders (caught by CURRENT_FACT above)
_BIOGRAPHY_RE = re.compile(
    r'\b(who\s+(invented|founded|created|wrote|authored|discovered|'
    r'designed|built|established|started|developed|composed|painted|'
    r'directed|produced|first\s+introduced)|'
    r'who\s+(is|was)\s+(?!the\s+)?(?!president\b|prime\s+minister\b|'
    r'minister\b|chancellor\b|governor\b|senator\b|secretary\b|'
    r'director\b|chairman\b|ceo\b|cfo\b|cto\b|head\b|leader\b|king\b|'
    r'queen\b|emperor\b|pope\b|mayor\b|ambassador\b|speaker\b|chief\b)'
    r'[A-Z]?[a-z]+(?:\s+[A-Z]?[a-z]+){1,4}|'
    r'biography\s+of|life\s+of|career\s+of|achievements?\s+of|'
    r'tell\s+me\s+about\s+[A-Z]|'
    r'when\s+was\s+\w+\s+born|born\s+in\b|died\s+in\b)\b',
    re.IGNORECASE,
)

# ── REASONING ────────────────────────────────────────────────────────────────
_REASONING_RE = re.compile(
    r'\b(why\s+(is|are|did|does|do|was|were)|'
    r'how\s+(does|do|did|would|could|should|can)\s+.{0,40}(work|happen|'
    r'affect|cause|lead|result|explain)|'
    r'what\s+would\s+happen\s+if|suppose\s+(that)?|'
    r'analyse|analyze|reason|logic|argument|implication|consequence|'
    r'because\s+of|as\s+a\s+result|cause\s+and\s+effect)\b',
    re.IGNORECASE,
)

# ── FACTOID (catch-all for short factual lookups) ────────────────────────────
# Matches "what is the capital of", "how many", "when did", "where is", etc.
# Placed last — anything not matched above that looks like a short fact query.
_FACTOID_RE = re.compile(
    r'\b(what\s+is\s+the\s+(capital|population|area|height|length|'
    r'distance|speed|temperature|weight|largest|smallest|highest|'
    r'deepest|longest|oldest|newest)|'
    r'how\s+(many|much|far|tall|long|old|big|large|small|deep|wide)|'
    r'when\s+(was|did|were|is)\b|'
    r'where\s+(is|was|are|were)\b|'
    r'which\s+(country|city|year|person|team|player|film|book))\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Routing table: intent_type -> (needs_web, confidence)
# ---------------------------------------------------------------------------
_ROUTING: dict = {
    "CURRENT_FACT":    (True,  0.93),
    "HISTORICAL_FACT": (False, 0.87),  # stable historical knowledge — RAG first
    "FACTOID":         (False, 0.85),  # try RAG first; router escalates if needed
    "BIOGRAPHY":       (False, 0.85),
    "DEFINITION":      (False, 0.88),
    "COMPARISON":      (False, 0.82),
    "CODING":          (False, 0.90),
    "MATH":            (False, 0.92),
    "REASONING":       (False, 0.80),
}


# ---------------------------------------------------------------------------
# LLM system prompt (updated taxonomy)
# ---------------------------------------------------------------------------
INTENT_SYSTEM_PROMPT = """You are an NLP Intent Classifier for a retrieval-augmented generation (RAG) system.

Classify the user query into EXACTLY ONE of these 9 intent types:

  CURRENT_FACT    — who currently holds a role, live/recent data, incumbent
                    office-holders, current prices or rates.
                    Examples: "Who is president of France?", "What is Bitcoin price?"
  HISTORICAL_FACT — specific named historical events, wars, revolutions, treaties,
                    historical milestones with a known date or time period.
                    Examples: "When was World War II?", "When did the French Revolution start?",
                              "What happened during the Cold War?", "When was the moon landing?"
  FACTOID         — short factual lookups not requiring live data: capitals,
                    distances, geographical measurements, static records.
                    Examples: "What is the capital of Japan?", "How tall is Everest?"
  BIOGRAPHY       — a specific person's life, career, works, or achievements.
                    Examples: "Who invented the telephone?", "Tell me about Einstein."
  DEFINITION      — define a term, concept, or phenomenon.
                    Examples: "What is machine learning?", "Define entropy."
  COMPARISON      — compare two or more things, pros/cons, A vs B.
                    Examples: "Difference between TCP and UDP", "Python vs JavaScript"
  CODING          — write, explain, or debug code; programming questions.
                    Examples: "Write a quicksort in Python", "Why does my loop break?"
  MATH            — arithmetic, algebra, calculus, proofs, numerical reasoning.
                    Examples: "Solve x^2 + 3x = 10", "Derivative of sin(x)"
  REASONING       — multi-step logic, causal analysis, hypotheticals, 'why' questions.
                    Examples: "Why did Rome fall?", "What would happen if the sun disappeared?"

Rules:
  - CURRENT_FACT takes priority over BIOGRAPHY for "who is [role]" queries.
  - HISTORICAL_FACT takes priority over FACTOID for named historical events.
  - HISTORICAL_FACT takes priority over BIOGRAPHY when the question is about an event, not a person.
  - DEFINITION takes priority over FACTOID for "what is [concept]" queries.
  - CODING takes priority over REASONING for implementation questions.
  - needs_web = true ONLY for CURRENT_FACT.

Respond ONLY with valid JSON — no markdown, no explanation:
{
  "intent_type": "CURRENT_FACT",
  "needs_web": true,
  "confidence": 0.95,
  "keywords": ["president", "France"],
  "reasoning": "Query asks for the current office-holder of a political role."
}
"""


# ---------------------------------------------------------------------------
# Heuristic fast-path (ordered: most specific first)
# ---------------------------------------------------------------------------

def _heuristic_intent(query: str) -> IntentResult | None:
    """
    Returns an IntentResult if the query matches a fast-path heuristic,
    otherwise returns None (caller falls through to LLM).

    Order is critical — CURRENT_FACT is checked before BIOGRAPHY so that
    'who is president of X' never misclassifies as BIOGRAPHY/FACTOID.
    """
    words = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 2]

    if _CURRENT_FACT_RE.search(query):
        needs_web, conf = _ROUTING["CURRENT_FACT"]
        return IntentResult(
            intent_type="CURRENT_FACT",
            needs_web=needs_web,
            confidence=conf,
            keywords=words,
            reasoning=(
                "Heuristic: query contains a current-fact marker "
                "(role-holder / temporal / price pattern)."
            ),
        )

    if _MATH_RE.search(query):
        needs_web, conf = _ROUTING["MATH"]
        return IntentResult(
            intent_type="MATH",
            needs_web=needs_web,
            confidence=conf,
            keywords=words,
            reasoning="Heuristic: query contains mathematical / numerical pattern.",
        )

    if _CODING_RE.search(query):
        needs_web, conf = _ROUTING["CODING"]
        return IntentResult(
            intent_type="CODING",
            needs_web=needs_web,
            confidence=conf,
            keywords=words,
            reasoning="Heuristic: query contains programming / code pattern.",
        )

    if _COMPARISON_RE.search(query):
        needs_web, conf = _ROUTING["COMPARISON"]
        return IntentResult(
            intent_type="COMPARISON",
            needs_web=needs_web,
            confidence=conf,
            keywords=words,
            reasoning="Heuristic: query contains comparison / versus pattern.",
        )

    if _DEFINITION_RE.search(query):
        needs_web, conf = _ROUTING["DEFINITION"]
        return IntentResult(
            intent_type="DEFINITION",
            needs_web=needs_web,
            confidence=conf,
            keywords=words,
            reasoning="Heuristic: query asks for a definition or explanation.",
        )

    if _HISTORICAL_FACT_RE.search(query):
        needs_web, conf = _ROUTING["HISTORICAL_FACT"]
        return IntentResult(
            intent_type="HISTORICAL_FACT",
            needs_web=needs_web,
            confidence=conf,
            keywords=words,
            reasoning=(
                "Heuristic: query references a named historical event, war, revolution, "
                "treaty, or dated milestone."
            ),
        )

    if _BIOGRAPHY_RE.search(query):
        needs_web, conf = _ROUTING["BIOGRAPHY"]
        return IntentResult(
            intent_type="BIOGRAPHY",
            needs_web=needs_web,
            confidence=conf,
            keywords=words,
            reasoning="Heuristic: query asks about a person's life, works, or historical role.",
        )

    if _REASONING_RE.search(query):
        needs_web, conf = _ROUTING["REASONING"]
        return IntentResult(
            intent_type="REASONING",
            needs_web=needs_web,
            confidence=conf,
            keywords=words,
            reasoning="Heuristic: query requires causal / multi-step reasoning.",
        )

    if _FACTOID_RE.search(query):
        needs_web, conf = _ROUTING["FACTOID"]
        return IntentResult(
            intent_type="FACTOID",
            needs_web=needs_web,
            confidence=conf,
            keywords=words,
            reasoning="Heuristic: query is a short factual lookup.",
        )

    return None  # no heuristic matched — call LLM


# ---------------------------------------------------------------------------
# Fallback classifier (LLM parse failure)
# ---------------------------------------------------------------------------

# Markers that signal a historical/biographical query in the keyword fallback.
# 'who' is intentionally ABSENT — "who is X" is CURRENT_FACT, not BIOGRAPHY.
_HISTORICAL_FACT_MARKERS = frozenset({
    "war", "revolution", "battle", "siege", "treaty", "independence",
    "colonialism", "empire", "medieval", "ancient", "renaissance",
    "enlightenment", "crusade", "holocaust", "genocide", "partition",
    "invasion", "occupation", "liberation", "armistice", "ceasefire",
    "rebellion", "uprising", "coup",
})

_BIOGRAPHY_MARKERS = frozenset({
    "invented", "discovered", "founded", "created", "wrote", "authored",
    "designed", "built", "established", "born", "died", "history", "origin",
    "biography", "autobiography", "life", "career",
})

_CODING_MARKERS = frozenset({
    "code", "function", "class", "algorithm", "program", "script",
    "python", "javascript", "java", "sql", "debug", "error", "syntax",
})

_MATH_MARKERS = frozenset({
    "calculate", "compute", "solve", "equation", "integral", "derivative",
    "algebra", "geometry", "proof", "matrix",
})


def _fallback_intent(query: str) -> IntentResult:
    """
    Keyword-based fallback used when the LLM call fails or returns unparseable JSON.
    Intentionally conservative — defaults to FACTOID rather than misclassifying.
    """
    words_set = set(re.findall(r'\w+', query.lower()))

    if words_set & _MATH_MARKERS:
        itype = "MATH"
    elif words_set & _CODING_MARKERS:
        itype = "CODING"
    elif words_set & _HISTORICAL_FACT_MARKERS:
        itype = "HISTORICAL_FACT"
    elif words_set & _BIOGRAPHY_MARKERS:
        itype = "BIOGRAPHY"
    else:
        itype = "FACTOID"

    needs_web, conf = _ROUTING[itype]
    keywords = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 2]
    return IntentResult(
        intent_type=itype,
        needs_web=needs_web,
        confidence=0.65,    # lower confidence — fallback path
        keywords=keywords,
        reasoning=f"Keyword fallback: matched '{itype}' markers (LLM unavailable).",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_intent(query: str) -> IntentResult:
    """
    Detects query intent.

    1. Fast heuristic patterns (no LLM, deterministic).
    2. LLM classification for ambiguous queries.
    3. Keyword fallback if LLM call fails or returns bad JSON.
    """
    # ── Step 1: heuristic fast-path ──────────────────────────────────────
    result = _heuristic_intent(query)
    if result is not None:
        return result

    # ── Step 2: LLM classification ───────────────────────────────────────
    try:
        raw = call_llm(system=INTENT_SYSTEM_PROMPT, prompt=f"User Query: {query}")
        parsed = json.loads(raw)

        itype = parsed.get("intent_type", "FACTOID")
        if itype not in _ROUTING:
            # LLM returned an unknown type — use fallback
            raise ValueError(f"Unknown intent type from LLM: {itype!r}")

        needs_web, _ = _ROUTING[itype]
        # Respect LLM's needs_web only for CURRENT_FACT; force False for others
        # to prevent unnecessary web calls on stable-knowledge queries.
        if itype != "CURRENT_FACT":
            needs_web = bool(parsed.get("needs_web", needs_web))

        return IntentResult(
            intent_type=itype,
            needs_web=needs_web,
            confidence=float(parsed.get("confidence", 0.80)),
            keywords=parsed.get("keywords", []),
            reasoning=parsed.get("reasoning", "LLM-classified intent."),
        )

    except Exception as exc:
        print(f"[Intent Detector] LLM classification failed ({exc}). Using keyword fallback.")
        return _fallback_intent(query)
