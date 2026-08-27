import re
from llm_client import call_llm

GENERATION_SYSTEM_PROMPT = """You are an expert research assistant answering a user's question \
using ONLY the provided context.
Guidelines:
1. Directly and completely answer the question based strictly on the provided context.
2. For "Who is" questions, provide a detailed explanation including key identifying information \
(profession, achievements, background, notable facts) found in the context.
3. For historical or scientific origin questions (e.g. "Who invented X?"), if the context shows that \
multiple pioneers contributed over time, briefly state that it was developed by multiple key figures.
4. Do NOT hallucinate claims outside the provided context.
5. Do NOT include meta-commentary, knowledge cutoff disclaimers, or unrelated examples.
"""


def clean_generated_answer(question: str, answer: str) -> str:
    """
    Cleans up common LLM artifacts:
    1. Removes LLM pre-training knowledge cutoff boilerplate (e.g., "As of my knowledge cutoff in 2023...").
    2. Strips out-of-context trailing paragraphs (e.g., leaked physics/quantum examples) when
       the question is not about science/physics.
    Note: "who is/are" questions retain more content since biographical info is essential.
    """
    if not answer or not answer.strip():
        return answer

    q_lower = question.lower()
    is_who_question = q_lower.strip().startswith(("who is ", "who are ", "who was ", "who were "))

    # 1. Remove knowledge cutoff boilerplate prefixes
    cleaned = re.sub(
        r'^(As of my knowledge cutoff (in \d{4}|up to \w+ \d{4}),?\s*|'
        r'As of early \d{4},?\s*|'
        r'Please note that the information provided here is based on the context available up to [^,.]*[,.]?\s*)',
        '',
        answer.strip(),
        flags=re.IGNORECASE,
    )
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Memory hits are stored as "Q: ...\nA: ...". If a fallback or small local
    # model repeats that wrapper, unwrap it so the final answer is a paragraph.
    qa_match = re.search(r'(?:^|\s)Q:\s*.*?\s+A:\s*(.+)$', cleaned, flags=re.IGNORECASE | re.DOTALL)
    if qa_match:
        cleaned = qa_match.group(1).strip()

    cleaned = re.sub(r'^\s*A:\s*', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned = _strip_embedded_memory_questions(cleaned)

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
    # Detect "who is/are" questions
    is_who_question = question.lower().strip().startswith(("who is ", "who are ", "who was ", "who were "))
    
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    
    if correction_feedback:
        prompt += f"\n\nYour previous answer had a quality issue: {correction_feedback}\nPlease correct it thoroughly."
    elif is_who_question:
        prompt += "\n\nProvide a comprehensive explanation with key details about who this person is, their profession, achievements, and notable facts."
    
    raw_answer = call_llm(system=GENERATION_SYSTEM_PROMPT, prompt=prompt)
    return clean_generated_answer(question, raw_answer)
