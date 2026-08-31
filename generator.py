import re
from llm_client import call_llm

GENERATION_SYSTEM_PROMPT = """You are an expert technical synthesizer and research assistant answering user queries.
Guidelines:
1. Do NOT merely copy or echo retrieved chunks; synthesize the evidence into a coherent, authoritative answer.
2. Preserve high technical accuracy:
   - Use 'tokens' rather than 'words' when describing LLM architecture and text processing.
   - Note that causal decoder models operate via masked self-attention, attending only to preceding tokens during generation.
3. Adapt explanation depth to the requested depth:
   - CONCISE: 1-2 brief sentences or bullet points.
   - DETAILED: structured sections with clear context and examples.
   - COMPREHENSIVE / VERY_DETAILED: full multi-section breakdown covering architecture (embeddings, self-attention, LayerNorm), training (pre-training, SFT, RLHF/DPO), inference (context window, KV cache, sampling), and limitations (hallucinations, RAG).
   - MATHEMATICAL: formal definitions, equations, and mathematical derivations.
   - BEGINNER: simple analogies, plain English, ELI5 presentation without jargon.
4. Do NOT introduce unsupported factual claims outside the evidence.
5. Do NOT include meta-commentary, Wikipedia labels, URLs, or source titles. Write clean, standalone prose.
"""

_SOURCE_TAG_RE = re.compile(r"\[(?:web_rag|trad_rag|memory)(?::[^\]]*)?\]")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"\n\s*-{3,}\s*\n")
_CITE_RE = re.compile(r"\[\d+\]")
_LATEST_SOURCES_RE = re.compile(r"(?i)\bbased on the latest retrieved sources:\s*")
_WIKI_LINE_RE = re.compile(
    r"(?im)^\s*.{0,140}?(?:[-–|]\s*)?(?:Wikipedia|Britannica)\b\s*:?\s*"
)
_WIKI_INLINE_RE = re.compile(r"(?i)\s*[-–|]\s*(?:Wikipedia|Britannica)\b")
_TITLE_COLON_RE = re.compile(r"^([^:]{1,80}):\s+(.+)$")


_WEB_NOISE_RE = re.compile(
    r"(?i)\b(click here to get an answer|pls mark me as|mark me as brainliest|"
    r"hope it will help you|explore all similar answers|what made soapy nostalgic)\b.*?(?=[.!?]|$)",
    re.IGNORECASE,
)
_WEB_SITE_RE = re.compile(
    r"(?i)\b(?:brainly\.in|quora\.com|studyx\.ai|geeksforgeeks\.org|leetcode\.com)\b\s*:?",
    re.IGNORECASE,
)


def strip_retrieval_chrome(text: str) -> str:
    """Remove source tags, URLs, Wikipedia labels, web forum noise, and title prefixes from retrieved text."""
    if not text:
        return ""

    text = re.sub(r'\[STRUCTURED LIST CONTRACT\].*?(?=\n\n[A-Z0-9]|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'(?i)\b(?:ANSWER TYPE|REQUIRED ITEMS|OUTPUT FORMAT|INSTRUCTION):\s*[^\n]+', '', text)
    text = _SEPARATOR_RE.sub("\n", text)
    text = _SOURCE_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _CITE_RE.sub("", text)
    text = _LATEST_SOURCES_RE.sub("", text)
    text = _WEB_NOISE_RE.sub("", text)
    text = _WEB_SITE_RE.sub("", text)
    text = re.sub(r'(?i)\bAnswer:\s*', '', text)
    # Strip search result pipe headers e.g. "First Opium War | Definition, Overview, China, Consequences ...: Aug 22, 2026 ·"
    text = re.sub(r'(?i)\b[A-Za-z0-9\s,\'–-]+(?:\s*\|\s*[A-Za-z0-9\s,\'–-]+)+\s*:\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\s*[·•-]\s*)?', '', text)
    text = re.sub(r'(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\s*[·•-]\s*', '', text)

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = _WIKI_LINE_RE.sub("", raw_line).strip()
        line = _WIKI_INLINE_RE.sub("", line).strip(" :-")
        match = _TITLE_COLON_RE.match(line)
        if match and (
            match.group(1).lower() in match.group(2).lower()
            or match.group(1).lower() in {"chopra (surname)", "wikipedia"}
        ):
            line = match.group(2).strip()
        if line and not re.fullmatch(r"[-–—]+", line):
            cleaned_lines.append(line)

    pieces = []
    for line in cleaned_lines:
        if pieces and not re.search(r"[.!?]$", pieces[-1]):
            pieces[-1] = pieces[-1].rstrip(",;:") + "."
        pieces.append(line)

    cleaned = " ".join(pieces)
    cleaned = re.sub(r"^[.?!/:;\s-]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_clean_sentences(text: str, max_len: int = 800) -> list:
    """
    Split retrieved text into complete prose sentences.
    Sentences are never truncated mid-way — if a raw chunk segment is longer
    than max_len chars, it is split at the last sentence-ending punctuation
    before that limit so the output always contains whole sentences.
    Source chrome (URLs, Wikipedia labels, title prefixes) is stripped first.
    """
    cleaned = strip_retrieval_chrome(text)
    if not cleaned:
        return []

    # Protect honorifics, military ranks, initials, and calendar abbreviations from false sentence splits
    protected = re.sub(r'\b([A-Z])\.', r'\1<DOT>', cleaned)
    protected = re.sub(r'\b(Lt|Gen|Maj|Col|Capt|Brig|Adm|Cmdr|Sgt|Cpl|Pvt|Dr|Mr|Mrs|Ms|Prof|Gov|Sen|Rep|St|vs|Inc|Ltd|Co|Corp|U\.S|U\.K|Dec|Jan|Feb|Nov|Oct|Aug|Sept|Jul|Jun|Apr|Mar)\.', r'\1<DOT>', protected, flags=re.IGNORECASE)
    parts = [p.replace('<DOT>', '.') for p in re.split(r'(?<=[.!?])\s+', protected)]
    sentences = []
    for part in parts:
        sentence = part.strip()
        if len(sentence) < 20:
            continue
        if re.match(r"^q:\s+", sentence, flags=re.IGNORECASE):
            continue
        if "learn more about" in sentence.lower():
            continue
        if "reading time" in sentence.lower():
            continue
        if len(sentence) > max_len:
            # Find the last sentence-ending punctuation before max_len
            # so we never return a fragment mid-sentence.
            truncated = sentence[:max_len]
            last_end = max(
                truncated.rfind(". "),
                truncated.rfind("! "),
                truncated.rfind("? "),
            )
            if last_end > 40:
                sentence = sentence[: last_end + 1].strip()
            else:
                # No punctuation found — keep the whole sentence rather than
                # returning a mid-word fragment.
                pass
        sentences.append(sentence)
    return sentences


def clean_generated_answer(question: str, answer: str) -> str:
    """
    Cleans up common LLM artifacts:
    1. Strips the question itself if the LLM echoed it at the start of the answer.
    2. Removes leading non-alphanumeric characters (e.g. '?' from '?Who is X?').
    3. Removes LLM knowledge-cutoff boilerplate.
    4. Strips out-of-context trailing paragraphs for non-science queries.
    """
    if not answer or not answer.strip():
        return answer

    # Normalise the question for comparison: strip leading punctuation/symbols
    # so "?Who is Nikola Tesla?" matches the same as "Who is Nikola Tesla?"
    q_normalised = re.sub(r'^[^a-zA-Z0-9]+', '', question.strip())
    q_lower = q_normalised.lower()
    is_who_question = q_lower.startswith(("who is ", "who are ", "who was ", "who were "))

    cleaned = answer.strip()

    # 1. Strip the verbatim question if the LLM echoed it at the start of the answer.
    #    Handles: "Who is Tesla? Nikola Tesla was..." and "?Who is Tesla? Nikola Tesla was..."
    # Escape the normalised question for use in a regex
    q_escaped = re.escape(q_normalised.rstrip("?"))
    cleaned = re.sub(
        r'^[^a-zA-Z0-9]*' + q_escaped + r'\??\s*',
        '',
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # 2. Strip knowledge cutoff disclaimers
    cleaned = re.sub(
        r'^(As of my knowledge cutoff (in \d{4}|up to \w+ \d{4}),?\s*|'
        r'As of early \d{4},?\s*|'
        r'Please note that the information provided here is based on the context available up to [^,.]*[,.]?\s*)',
        '',
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Memory hits are stored as "Q: ...\nA: ...". If a fallback or small local
    # model repeats that wrapper, unwrap it so the final answer is a paragraph.
    qa_match = re.search(r'(?:^|\s)Q:\s*.*?\s+A:\s*(.+)$', cleaned, flags=re.IGNORECASE | re.DOTALL)
    if qa_match:
        cleaned = qa_match.group(1).strip()

    cleaned = re.sub(r'^\s*A:\s*', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = _strip_embedded_memory_questions(cleaned)
    cleaned = strip_retrieval_chrome(cleaned)
    cleaned = re.sub(r"(?i)\bbased on the latest retrieved sources:\s*", "", cleaned).strip()

    # Avoid returning the old weak memory/fallback phrasing as the final style.
    if is_who_question:
        weak_match = re.fullmatch(
            r'Based on the available information,?\s+the answer is\s+(.+?)\.?',
            cleaned,
            flags=re.IGNORECASE,
        )
        if weak_match:
            person = weak_match.group(1).strip()
            cleaned = f"{person} is the person identified by the retrieved evidence."

    # 2. For "who is" questions, retain more content; for others, strip trailing unrelated paragraphs
    if not is_who_question:
        is_physics_q = any(w in q_lower for w in ["quantum", "relativity", "invent", "physic", "einstein", "planck", "bohr"])

        if not is_physics_q:
            paragraphs = cleaned.split("\n\n")
            filtered_paras = []
            for p in paragraphs:
                p_strip = p.strip()
                # If paragraph contains prompt-leaked quantum/relativity example boilerplate, skip it
                if re.search(r'\b(theory of relativity|quantum hypothesis|Max Planck|Albert Einstein|Hermann Minkowski)\b', p_strip, re.IGNORECASE):
                    continue
                if p_strip:
                    filtered_paras.append(p_strip)
            cleaned = "\n\n".join(filtered_paras)

    return cleaned.strip()


def _strip_embedded_memory_questions(text: str) -> str:
    """Remove repeated memory-question fragments from an otherwise useful answer."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    kept = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip() and not re.match(r'^Q:\s+', sentence.strip(), flags=re.IGNORECASE)
    ]
    return " ".join(kept).strip()


def generate_answer(question: str, context: str, correction_feedback: str = "") -> str:
    # Strip leading non-alphanumeric characters (e.g. '?' typed accidentally)
    # so '?Who is Nikola Tesla?' is treated identically to 'Who is Nikola Tesla?'
    question_clean = re.sub(r'^[^a-zA-Z0-9]+', '', question.strip())

    q_lower = question_clean.lower()
    is_biography = q_lower.startswith(("who is ", "who are ", "who was ", "who were "))

    from answer_style_detector import detect_answer_style
    style_spec = detect_answer_style(question_clean)

    prompt = f"Context:\n{context}\n\nQuestion: {question_clean}\n\n[Presentation Guidelines]"
    if style_spec.depth == "VERY_DETAILED":
        prompt += "\n- Depth: Provide an exhaustive, multi-paragraph deep-dive from first principles without skipping details."
    elif style_spec.depth == "DETAILED":
        prompt += "\n- Depth: Provide a thorough, detailed explanation with full context."
    elif style_spec.depth == "CONCISE":
        prompt += "\n- Depth: Provide a brief, concise summary in 1-2 sentences."

    if style_spec.style == "STEP_BY_STEP":
        prompt += "\n- Style: Walk through the explanation in clear, numbered step-by-step order."
    elif style_spec.style == "BEGINNER":
        prompt += "\n- Style: Explain in simple, plain English suitable for a beginner (ELI5), avoiding heavy jargon."
    elif style_spec.style == "TECHNICAL":
        prompt += "\n- Style: Provide an advanced technical explanation covering architecture and low-level details."
    elif style_spec.style == "COMPARISON":
        prompt += "\n- Style: Provide a clear comparative analysis highlighting key differences and pros/cons."

    if style_spec.examples:
        prompt += "\n- Examples: Include concrete, real-world practical examples to illustrate."

    if style_spec.output_format == "TABLE":
        prompt += "\n- Format: Present the main findings in a clean Markdown table."
    elif style_spec.output_format == "BULLET_POINTS":
        prompt += "\n- Format: Present key points in Markdown bullet points."

    if correction_feedback:
        prompt += (
            f"\n\nYour previous answer had a quality issue: {correction_feedback}"
            "\nPlease correct it thoroughly."
        )
    elif is_biography:
        prompt += (
            "\n\nWrite a comprehensive, multi-paragraph answer (3-5 paragraphs). "
            "Cover: (1) who the person is and their background, "
            "(2) their major works, inventions, or achievements, "
            "(3) their historical context and impact, "
            "(4) their legacy. "
            "Use all relevant details from the context. Do not summarise into just 2-3 sentences."
        )
    else:
        prompt += (
            "\n\nAnswer clearly and completely using all relevant details from the context."
        )

    raw_answer = call_llm(system=GENERATION_SYSTEM_PROMPT, prompt=prompt)
    return clean_generated_answer(question_clean, raw_answer)
