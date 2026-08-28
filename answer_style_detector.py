"""
Answer Style & Response Depth Detector Module
==============================================

Analyzes user queries to extract:
  - depth:            CONCISE | BALANCED | DETAILED | VERY_DETAILED
  - style:            STANDARD | STEP_BY_STEP | FIRST_PRINCIPLES | BEGINNER | TECHNICAL | COMPARISON | JUSTIFICATION | ANALYSIS
  - examples:         True | False
  - technical_level:  BEGINNER | INTERMEDIATE | ADVANCED
  - output_format:    PARAGRAPH | BULLET_POINTS | TABLE | JSON | CODE | DIAGRAM | TIMELINE
"""
import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


@dataclass
class AnswerStyle:
    depth: str            # CONCISE | BALANCED | DETAILED | VERY_DETAILED
    style: str            # STANDARD | STEP_BY_STEP | FIRST_PRINCIPLES | BEGINNER | TECHNICAL | COMPARISON | JUSTIFICATION | ANALYSIS
    examples: bool        # True | False
    technical_level: str  # BEGINNER | INTERMEDIATE | ADVANCED
    output_format: str    # PARAGRAPH | BULLET_POINTS | NUMBERED_LIST | TABLE | JSON | CODE | DIAGRAM | TIMELINE
    requested_count: Optional[int] = None  # e.g. 10 for "Top 10", 5 for "5 best", None for open text
    answer_mode: str = "FACT"              # LIST_ONLY | MULTI_FACT | EXPLANATION | COMPARISON | SUMMARY | FACT
    llm_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_answer_style(query: str) -> AnswerStyle:
    """Detects answer presentation style, depth, format, technical level, answer_mode, and item count from query."""
    q = query.strip().lower()

    # ── 1. Explicit Item Count & List Detection ──────────────────────────────
    requested_count = None
    count_match = re.search(
        r'\b(?:top|best|strongest|leading|most\s+popular|first)\s+(\d{1,2})\b|'
        r'\b(\d{1,2})\s+(?:best|strongest|top|leading|places|ways|steps|layers|principles|types|kinds|examples|rules|methods|reasons|advantages|disadvantages|forces|units)\b|'
        r'\blist\s+(?:the\s+)?(\d{1,2})\b',
        q,
        re.IGNORECASE,
    )
    if count_match:
        for g in count_match.groups():
            if g and g.isdigit():
                requested_count = int(g)
                break

    # ── 2. Output Format Detection ───────────────────────────────────────────
    output_format = "PARAGRAPH"
    if any(k in q for k in ("in a table", "as a table", "compare in a table", "tabular format", "table format")):
        output_format = "TABLE"
    elif requested_count is not None or re.match(r'^(?:list\b|name\b|top\s+\d+|rank\b|give\s+me\s+\d+)', q):
        output_format = "NUMBERED_LIST"
    elif any(k in q for k in ("bullet points", "in bullets", "bullet list", "list them", "give me a list")):
        output_format = "BULLET_POINTS"
    elif any(k in q for k in ("in json", "as json", "json format", "return json")):
        output_format = "JSON"
    elif any(k in q for k in ("give me code", "show the code", "code solution", "code example")):
        output_format = "CODE"
    elif any(k in q for k in ("with a diagram", "draw a diagram", "visualize", "flowchart")):
        output_format = "DIAGRAM"
    elif any(k in q for k in ("timeline", "chronology", "chronological order")):
        output_format = "TIMELINE"

    # ── 3. Examples Detection ────────────────────────────────────────────────
    examples = bool(re.search(
        r'\b(give\s+examples?|with\s+examples?|using\s+examples?|show\s+examples?|'
        r'real-world\s+examples?|practical\s+examples?|simple\s+examples?|multiple\s+examples?|'
        r'illustrate\s+with)\b', q, re.IGNORECASE
    )) or "with examples" in q or "examples" in q

    # ── 4. Depth Detection ───────────────────────────────────────────────────
    depth = "BALANCED"
    if any(k in q for k in (
        "comprehensive", "comprehensively", "explain everything", "from scratch",
        "from first principles", "extensively", "don't skip anything", "leave no details out",
        "in great detail", "in extreme detail", "full analysis", "complete analysis", "deep analysis"
    )):
        depth = "VERY_DETAILED"
    elif any(k in q for k in (
        "in detail", "in depth", "explain deeply", "explain thoroughly", "explain fully",
        "give a detailed", "detailed explanation", "more details", "tell me more", "elaborate",
        "go into detail", "go deeper", "deep dive", "detailed overview"
    )):
        depth = "DETAILED"
    elif any(k in q for k in (
        "briefly", "in brief", "short answer", "brief answer", "in short", "summarize",
        "summary", "quickly", "just tell me", "just give me", "only the answer", "tl;dr", "tldr"
    )):
        depth = "CONCISE"

    # ── 5. Style Detection ────────────────────────────────────────────────────
    style = "STANDARD"
    if any(k in q for k in (
        "step by step", "step-by-step", "walk me through", "show me the steps",
        "show all steps", "explain each step", "one step at a time", "procedure"
    )):
        style = "STEP_BY_STEP"
    elif any(k in q for k in ("first principles", "from scratch", "from ground up")):
        style = "FIRST_PRINCIPLES"
    elif any(k in q for k in (
        "in simple words", "simple explanation", "easy explanation", "like i'm a beginner",
        "for a beginner", "beginner friendly", "eli5", "in plain english", "easy to understand", "simplify it"
    )):
        style = "BEGINNER"
    elif any(k in q for k in (
        "technical explanation", "technical details", "advanced explanation", "under the hood",
        "internals", "implementation details", "architecture", "low-level", "mathematical derivation"
    )):
        style = "TECHNICAL"
    elif any(k in q for k in (
        "compare", "difference between", "versus", " vs ", "pros and cons", "advantages and disadvantages"
    )):
        style = "COMPARISON"
    elif any(k in q for k in ("justify", "reasoning", "give reasons", "proof", "prove", "show why")):
        style = "JUSTIFICATION"
    elif any(k in q for k in ("analyze", "analyse", "analysis", "evaluate", "assess")):
        style = "ANALYSIS"

    # ── 6. Technical Level Detection ─────────────────────────────────────────
    technical_level = "INTERMEDIATE"
    if style == "BEGINNER" or any(k in q for k in ("beginner", "simple", "plain english", "eli5", "easy", "for dummies")):
        technical_level = "BEGINNER"
    elif style == "TECHNICAL" or any(k in q for k in ("advanced", "expert", "technical", "low-level", "internals", "architecture", "derivation")):
        technical_level = "ADVANCED"

    # ── 7. Answer Mode & LLM Requirement Determination ───────────────────────
    # Distinguish pure factual lists vs conceptual explanations with numbers
    is_explanation_request = any(w in q for w in (
        "explain", "why", "how", "what is the difference", "deep dive", "walk through",
        "elaborate", "overview", "mechanism", "concept", "architecture of", "derive"
    ))
    
    if is_explanation_request:
        answer_mode = "EXPLANATION" if style != "COMPARISON" else "COMPARISON"
        llm_required = True
    elif output_format == "NUMBERED_LIST" or requested_count is not None:
        answer_mode = "LIST_ONLY"
        llm_required = False
    elif style in ("COMPARISON", "JUSTIFICATION", "ANALYSIS", "STEP_BY_STEP"):
        answer_mode = style
        llm_required = True
    elif depth in ("DETAILED", "VERY_DETAILED"):
        answer_mode = "EXPLANATION"
        llm_required = True
    else:
        answer_mode = "FACT"
        llm_required = False

    return AnswerStyle(
        depth=depth,
        style=style,
        examples=examples,
        technical_level=technical_level,
        output_format=output_format,
        requested_count=requested_count,
        answer_mode=answer_mode,
        llm_required=llm_required,
    )
