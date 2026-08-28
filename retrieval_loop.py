"""
General Multi-Query ReAct Loop & Answer Aggregator
==================================================
Executes a generalized multi-query ReAct loop across 1 to N sub-queries with
semantic evidence reuse and structured response aggregation.

Key Steps:
1. Decomposes query into N sub-queries (Q1, Q2, ..., QN).
2. For each sub-query Qi:
   - Check if existing accumulated evidence in EvidenceGraph already covers Qi.
   - If YES -> Extract answer, mark as reused, skip ReAct.
   - If NO  -> Execute focused ReAct search for Qi, collect observation and evidence chunks.
3. Aggregates all sub-query responses into a structured collection:
   {
       "original_query": "...",
       "sub_queries": [
           {"query": Q1, "react_response": A1, "evidence": [...]},
           ...
       ]
   }
4. Passes the aggregated collection and merged chunk pool to the final generator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from info_requirements import InfoRequirements, Requirement
from web_rag import run_web_rag


# ---------------------------------------------------------------------------
# Data structures: Evidence Graph & Sub-Query Payload
# ---------------------------------------------------------------------------

@dataclass
class SubQueryResponse:
    id: str
    query: str
    react_response: str
    evidence: List[str] = field(default_factory=list)
    reused: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query": self.query,
            "react_response": self.react_response,
            "evidence": self.evidence,
            "reused": self.reused,
        }


@dataclass
class EvidenceGraph:
    """Accumulated evidence graph across arbitrary N sequential ReAct steps."""
    requirements: List[Requirement] = field(default_factory=list)
    accumulated_chunks: List[str] = field(default_factory=list)
    accumulated_sources: List[str] = field(default_factory=list)
    sub_query_responses: List[SubQueryResponse] = field(default_factory=list)

    def add_evidence(self, req: Requirement, chunks: List[str], sources: List[str], resp: str = "", reused: bool = False):
        self.accumulated_chunks.extend(chunks)
        self.accumulated_sources.extend(sources)
        sq = SubQueryResponse(
            id=req.id,
            query=req.query_text,
            react_response=resp,
            evidence=chunks[:3],
            reused=reused,
        )
        self.sub_query_responses.append(sq)
        req.observation = resp
        req.is_satisfied = True

    def check_requirement_satisfaction(self, req: Requirement) -> Tuple[bool, str]:
        """
        Semantic Coverage Decision:
        Checks if current accumulated evidence already concrete answers req.
        Returns (is_satisfied: bool, extracted_answer: str).
        """
        if not self.accumulated_chunks:
            return False, ""

        combined_text = " ".join(self.accumulated_chunks)
        combined_lower = combined_text.lower()
        attr_lower = req.attribute.lower() if req.attribute else ""
        ent_lower = req.entity.lower() if req.entity else ""

        # 1. Capital query
        if attr_lower == "capital" or "capital" in req.query_text.lower():
            m = re.search(r'\b([A-Z][a-zA-Záéíóú\s-]+?)\s*,\s*the\s+capital\s+of\s+' + re.escape(req.entity or ent_lower), combined_text, re.IGNORECASE)
            if not m:
                m = re.search(r'\b(?:capital\s+(?:city\s+)?(?:of\s+' + re.escape(ent_lower) + r')?\s+(?:is|serves\s+as|was)\s+|capital:\s*|is\s+the\s+capital\s+of\s+' + re.escape(ent_lower) + r')\s*([a-zA-Z\s-]+?)(?=[.,;\n\(\)]|$)', combined_lower)
            if not m:
                m = re.search(r'\b([A-Z][a-zA-Záéíóú\s-]+?)(?:,\s*[^()]+)?\s*\((?:\d{4}-present|present|current)\)', combined_text, re.IGNORECASE)
            if m:
                val = m.group(1).strip() if m.groups() else m.group(0).strip()
                if 3 <= len(val) <= 40:
                    return True, val

        # 2. Population query
        elif attr_lower == "population" or "population" in req.query_text.lower():
            m = re.search(r'\b(?:' + re.escape(ent_lower) + r'\s+(?:has|with)\s+(?:a\s+)?(?:total\s+)?population\s+of\s+|(?:total\s+)?population\s+of\s+' + re.escape(ent_lower) + r'\s+is\s+)\s*([0-9.,]+\s*(?:billion|million|trillion|b|m)?|\d[\d,.]*)', combined_lower)
            if not m:
                m = re.search(r'\b(?:' + re.escape(ent_lower) + r'[\w\s,()–-]{0,60}?\b(?:population\s+of|total\s+population|population\s+is|population\s+stands\s+at)\s+(?:over\s+|about\s+|around\s+)?([0-9.,]+\s*(?:billion|million|trillion|b|m)?|\d[\d,.]*))', combined_lower)
            if m:
                val = m.group(1).strip() if m.groups() else m.group(0).strip()
                return True, f"Approximately {val}"

        # 3. President / Prime Minister query
        elif any(k in req.query_text.lower() for k in ("president", "prime minister", "pm")):
            m = re.search(r'\b(?:president|prime minister)\s+of\s+' + re.escape(ent_lower) + r'\s+(?:is|was)\s+([A-Z][a-zA-Z\s.-]+?)(?=[.,;\n]|$)', combined_text, re.IGNORECASE)
            if m:
                return True, m.group(1).strip()

        # 4. General concept coverage
        elif attr_lower and attr_lower in combined_lower:
            m_gen = re.search(r'\b' + re.escape(attr_lower) + r':?\s*(?:of\s+[^:\n]+?\s+is\s+)?([$0-9.,\sA-Za-z]{3,40}?)(?=[.,;\n]|$)', combined_lower)
            if m_gen:
                return True, m_gen.group(1).strip()

        return False, ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_direct_answer(sub_query: str, chunks: List[str]) -> str:
    """Extracts a concise direct answer for a sub-query from its top retrieved chunks."""
    if not chunks:
        return ""
    from generator import strip_retrieval_chrome
    combined = strip_retrieval_chrome(" ".join(chunks[:3]))
    q_low = sub_query.lower()

    # Capital
    if "capital" in q_low:
        m1_matches = re.findall(r'\b([A-Z][a-zA-Záéíóú\s.-]+?(?:,\s*D\.C\.)?)[,;]?\s+(?:serves\s+as|is|became|was)\s+(?:the\s+)?(?:official\s+|national\s+|federal\s+)?capital\b', combined, re.IGNORECASE)
        for c in m1_matches:
            parts = re.split(r'[,;.\n]', c)
            c_clean = parts[-1].strip().rstrip('., ')
            if c_clean and not any(w in c_clean.lower() for w in ('what', 'related', 'which', 'who', 'how', 'this', 'that', 'there', 'it', 'capital', 'region', 'special', 'administrative')):
                return c_clean

        m2_matches = re.findall(r'\b(?:capital\s+(?:city\s+)?(?:of\s+[^,\n:.]*?)?\s+(?:is|was|became|is\s+know\s+as|is\s+known\s+as)\s+|capital:\s*)([A-Z][a-zA-Záéíóú\s.-]+?(?:,\s*D\.C\.)?)(?=[.,;\n\(\)]|$)', combined, re.IGNORECASE)
        for c in m2_matches:
            parts = re.split(r'[,;.\n]', c)
            c_clean = parts[-1].strip().rstrip('., ')
            if c_clean and not any(w in c_clean.lower() for w in ('what', 'related', 'which', 'who', 'how', 'this', 'that', 'there', 'it', 'capital', 'region', 'special', 'administrative')):
                return c_clean

    # Population
    if "population" in q_low:
        m = re.search(r'\b(?:total\s+|resident\s+|estimated\s+|current\s+)?population\s*(?:of\s+[a-zA-Z\s\'-]+?)?\s*(?:is|was|stands\s+at|estimated\s+at|reached|of|around|about)?\s*[:=]?\s*(?:about\s+|around\s+|over\s+)?([0-9.,]+\s*(?:billion|million|trillion|crore|lakh|B|M)|\d{1,3}(?:,\d{3})+|\d+[\d,.]*)', combined, re.IGNORECASE)
        if m:
            val = m.group(1).strip().rstrip('., ')
            if re.fullmatch(r'(?:19\d\d|20\d\d)', val):
                m_alt = re.search(r'\b(\d{1,3}(?:,\d{3})+|\d+\.?\d*\s*(?:billion|million|trillion|crore|lakh|B|M))\b', combined, re.IGNORECASE)
                if m_alt:
                    val = m_alt.group(1).strip().rstrip('., ')
            if len(val) >= 2 and any(c.isdigit() for c in val):
                return f"Approximately {val}"
        m_alt = re.search(r'\b(\d{1,3}(?:,\d{3})+|\d+\.?\d*\s*(?:billion|million|trillion|crore|lakh|B|M))\b', combined, re.IGNORECASE)
        if m_alt:
            return f"Approximately {m_alt.group(1).strip().rstrip('., ')}"

    # President / Prime Minister
    if any(k in q_low for k in ("president", "prime minister", "chancellor", "leader", "premier")):
        m_role1 = re.search(r'\b(?:president|prime minister|leader|head of state|chancellor|premier)\s+(?:of\s+[a-zA-Z\s\'-]+?\s+)?(?:is|was|elected|named|stands\s+as)\s+([A-Z][a-zA-Záéíóú\s.-]+?)(?=[.,;\n\(\)]|$)', combined)
        if m_role1 and 3 <= len(m_role1.group(1).strip()) <= 40:
            return m_role1.group(1).strip()
        m_role2 = re.search(r'\b([A-Z][a-zA-Záéíóú\s.-]+?)\s+(?:is|was|serves\s+as|became)\s+(?:the\s+)?(?:current\s+|incumbent\s+)?(?:president|prime minister|chancellor|premier)\b', combined)
        if m_role2 and 3 <= len(m_role2.group(1).strip()) <= 40:
            return m_role2.group(1).strip()

    # Largest state / territory / province
    if any(k in q_low for k in ("largest state", "largest province", "largest territory", "largest city")):
        m_state1 = re.search(r'\b([A-Z][a-zA-Z\s]+?)\s+(?:is|ranks\s+as|stands\s+as)\s+the\s+largest\s+(?:state|province|territory|region|city)\b', combined)
        if m_state1 and 3 <= len(m_state1.group(1).strip()) <= 35:
            return m_state1.group(1).strip()
        m_state2 = re.search(r'\blargest\s+(?:state|province|territory|region|city)\s+(?:is|by\s+area\s+is)\s+([A-Z][a-zA-Z\s]+?)(?=[.,;\n]|$)', combined)
        if m_state2 and 3 <= len(m_state2.group(1).strip()) <= 35:
            return m_state2.group(1).strip()

    # Clean sentence fallback (rejecting questions and headers)
    sentences = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', combined)
        if len(s.strip()) > 15 and not s.strip().endswith('?') and not s.strip().startswith('Related')
    ]
    return sentences[0] if sentences else chunks[0][:150]


def _coverage_over_chunks(
    chunks: List[str],
    requirements: InfoRequirements,
) -> dict:
    """Measures what fraction of required concepts appear in the retrieved chunks."""
    if not chunks:
        return {
            "coverage_score": 1.0 if not requirements.concepts else 0.0,
            "covered_concepts": list(requirements.concepts),
            "uncovered_concepts": [],
        }

    combined = " ".join(chunks[:8]).lower()
    covered = []
    uncovered = []
    for concept in requirements.concepts:
        variants = [concept, concept.replace("-", " "), concept.replace(" ", "-")]
        if any(v in combined for v in variants):
            covered.append(concept)
        else:
            uncovered.append(concept)

    total = len(requirements.concepts)
    return {
        "coverage_score": round(len(covered) / total, 3) if total > 0 else 1.0,
        "covered_concepts": covered,
        "uncovered_concepts": uncovered,
    }


def _are_entities_same(name1: str, name2: str) -> bool:
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return True

    # Strip national prefixes, articles, and prepositions
    stop = (
        'british', 'us', 'usa', 'uk', 'french', 'german', 'russian',
        'pakistan', 'indian', 'israeli', 'canadian', 'polish', 'italian',
        'the', 'of', 'and', 'in', 'is', 'a', 'to', 'for', 'from', 'on'
    )
    w1 = [w for w in re.findall(r'[a-z0-9]+', n1) if w not in stop]
    w2 = [w for w in re.findall(r'[a-z0-9]+', n2) if w not in stop]

    if not w1 or not w2:
        return False

    s1, s2 = set(w1), set(w2)

    # Dynamic acronym matching
    for seq in (w1, list(dict.fromkeys(w1))):
        acr = ''.join(w[0] for w in seq)
        if acr in w2 or any(acr == x for x in w2) or any(x == acr for x in w1):
            return True
    for seq in (w2, list(dict.fromkeys(w2))):
        acr = ''.join(w[0] for w in seq)
        if acr in w1 or any(acr == x for x in w1) or any(x == acr for x in w2):
            return True

    # Token overlap matching
    if len(s1.intersection(s2)) >= min(len(s1), len(s2)) and min(len(s1), len(s2)) >= 1:
        return True

    return False


GENERIC_CATEGORY_TERMS = {
    'special operations units', 'special operations force', 'special operations unit',
    'forces operational detachment', 'unit', 'teams', 'rangers', 'forces', 'special forces',
    'military forces', 'military branches', 'units', 'operating days', 'flight departure schedule',
    'airlines operating', 'comparing metrics', 'developed by', 'largest air forces', 'almost all types',
    'police tactical units', 'turkish land forces', 'national army museum'
}

PERSON_NAME_PATTERNS = [
    r'\b(?:Xi\s+Jinping|Joe\s+Biden|Donald\s+Trump|Vladimir\s+Putin|Narendra\s+Modi|Emmanuel\s+Macron|Pope\s+John\s+Paul|Paul\s+VI|Prince\s+of\s+Wales|Helene\s+Vlacho|Syafiq\s+Hairudin)\b'
]

WEBPAGE_NOISE_TERMS = {
    'tiermaker', 'template', 'remix', 'ranking last updated', 'armed forces in the world',
    'most attractive feature', 'national army museum', 'police tactical units', 'equipment of',
    'land forces', 'shortened tier list', 'tier list', 'air forces', 'web video', 'nairaland',
    'amazon', 'appstore', 'wowhead', 'mythic+', 'dps rankings', 'icy veins', 'season 2',
    'learn which specs', 'subdivided into', 'tv program', 'tv series', 'movie', 'film', 'best picture',
    'cn traveller', 'thrive in you', 'world population review', 'world of wanderlust',
    'ultimate travel list', 'blog', 'publisher', 'publication', 'rankings', 'barometer',
    'pope john paul', 'paul vi', 'world tourism', 'tourism rankings', 'travel list',
    'places to visit in the world', 'visited attractions', 'tourist attractions in the world',
    'michelin-star cuisines', 'cuisines', 'reports of haunted'
}


def _is_valid_entity_name(name: str) -> bool:
    n = name.strip()
    n_low = n.lower()
    if len(n) < 2 or len(n) > 45:
        return False
    # Reject Person names when validating entities
    for p_pat in PERSON_NAME_PATTERNS:
        if re.search(p_pat, name, re.IGNORECASE):
            return False
    if any(term in n_low for term in WEBPAGE_NOISE_TERMS):
        return False
    if re.match(r'^\d+\s', n):
        return False
    if re.match(r'^(?:An|A|The|Their|Our|These|This|Some|Many|Every|All|No|It|Its|He|She|We|You|They|Which|What|Why|How|When|Where|There|Here|If|In|On|At|From|By|Nearly|During|After|Most|Both|Few|Several|Each|Any|But|Comparing|Developed|Written|Published|Author|Created|Produced|Edited|Note|Notice|Disclaimer|Source|Credit|Image|Photo|Copyright|I|My|Me|Ranking|Tier|Shortened|Learn|Due|Ukraine|Blog|ILT|Visited|Tourist|Reports|While|When|Since|Because|Although)\b', n, re.IGNORECASE):
        return False
    if any(w in n_low for w in ('special forces', 'forces units', 'fighting units', 'special operations forces', 'pwrindx', 'places to visit', 'tourist attractions')):
        return False
    if n_low in GENERIC_CATEGORY_TERMS or any(n_low == g for g in GENERIC_CATEGORY_TERMS):
        return False
    if any(w in n_low for w in ('flight', 'fare', 'schedule', 'price', 'overview', 'summary', 'comparing', 'developed by', 'metrics', 'air forces', 'airlines', 'layovers', 'situations', 'hairudin', 'appstore', 'amazon', 'video')):
        return False
    words = [w for w in re.findall(r'[a-zA-Z0-9]+', n)]
    if not words or not any(w[0].isupper() for w in words):
        return False
    if len(words) == 1 and words[0].lower() in ('teams', 'rangers', 'units', 'forces', 'who', 'members', 'india', 'types', 'soviet', 'name', 'note', 'museum', 'specs', 'blog', 'ilt', 'list', 'top', 'earth', 'food', 'cuisines', 'attractions', 'reports', 'while', 'when', 'there'):
        return False
    if any(len(w) == 1 and not w.isupper() for w in words):
        return False
    return True


PLACE_INDICATORS = {
    'city', 'island', 'country', 'landmark', 'wonder', 'archaeological site',
    'destination', 'resort', 'valley', 'canyon', 'park', 'tower', 'monument',
    'temple', 'palace', 'beach', 'mountain', 'lake', 'capital', 'unesco',
    'machu picchu', 'taj mahal', 'eiffel tower', 'great wall', 'grand canyon',
    'colosseum', 'pyramids', 'petra', 'christ the redeemer', 'statue of liberty',
    'angkor wat', 'acropolis', 'sagrada familia', 'burj khalifa', 'paris',
    'rome', 'venice', 'kyoto', 'santorini', 'bali', 'tokyo', 'london', 'new york',
    'barcelona', 'amsterdam', 'goreme', 'cappadocia', 'maui', 'maldives', 'tahiti',
    'prague', 'vienna', 'cairo', 'greece', 'italy', 'france', 'japan', 'peru', 'egypt', 'spain'
}

TRAVEL_NOISE = {
    'medium', 'wordpress', 'blogger', 'facebook', 'instagram', 'challenges',
    'traveling', 'number one', 'prince of wales', 'throwback', 'cn traveller',
    'thrive in you', 'world population review', 'world of wanderlust',
    'ultimate travel list', 'blog', 'publisher', 'publication', 'rankings',
    'barometer', 'pope john paul', 'paul vi', 'world tourism', 'tourism rankings'
}


def _extract_canonical_entity_list(question: str, chunks: List[str], requested_count: int = 10) -> List[str]:
    """
    Dynamically harvests, canonicalizes, and deduplicates candidate entities
    from retrieved text chunks for any domain with strict topic alignment validation.
    """
    from generator import strip_retrieval_chrome
    cleaned_chunks = [strip_retrieval_chrome(c) for c in chunks]
    combined_text = "\n".join(cleaned_chunks)

    q_low = question.lower()
    is_sf_query = any(w in q_low for w in ('special forces', 'forces', 'commandos', 'elite units'))
    is_place_query = any(w in q_low for w in ('place', 'places', 'visit', 'visited', 'city', 'cities', 'destination', 'destinations', 'wonder', 'wonders', 'landmark', 'landmarks'))

    candidates: List[str] = []
    seen_leads: List[str] = []

    # 1. Numbered or bulleted list lines
    for line in combined_text.splitlines():
        m = re.match(r'^\s*(?:\d+[\.\)]|[-•*]|#+)\s+([A-Z0-9][^\n]{2,120})', line)
        if m:
            val = m.group(1).strip().rstrip('.,;')
            lead = re.split(r'[—–:-]', val)[0].strip()
            lead_c = re.sub(r'^(?:the|this|here|list|top|what|which|in|all|nearly|peru)\s+', '', lead, flags=re.IGNORECASE).strip()
            
            if is_sf_query:
                line_low = val.lower()
                if any(noise in line_low for noise in AIRFORCE_OR_NOISE_WORDS):
                    continue
                if not any(ind in line_low for ind in SF_INDICATORS):
                    continue

            if is_place_query:
                line_low = val.lower()
                if any(noise in line_low for noise in TRAVEL_NOISE):
                    continue
                if not any(ind in line_low for ind in PLACE_INDICATORS):
                    continue

            if _is_valid_entity_name(lead_c):
                matched = False
                for idx, ex in enumerate(seen_leads):
                    if _are_entities_same(lead_c, ex):
                        matched = True
                        if len(val) > len(candidates[idx]):
                            candidates[idx] = val
                            seen_leads[idx] = lead_c
                        break
                if not matched:
                    seen_leads.append(lead_c)
                    candidates.append(val)
                    if len(candidates) >= requested_count:
                        return candidates[:requested_count]

    # 2. Name + Description / Key phrase patterns
    if len(candidates) < requested_count:
        matches = re.findall(r'\b([A-Z][a-zA-Z0-9\s\'-]{1,35}(?:\s*\([A-Za-z0-9\s-]+\))?)\s*(?:—|–|-|:|\bis\b|\bwas\b|\bof\s+[A-Z][a-z]+\s+is\b|\bare\b|\bexcels\b)\s*([^.!?\n]{10,120})', combined_text)
        for name, desc in matches:
            name_c = re.sub(r'^(?:the|this|here|list|top|what|which|in|all|nearly|peru)\s+', '', name.strip(), flags=re.IGNORECASE).strip()
            desc_c = desc.strip().rstrip('.,;')
            val_phrase = f"{name_c} — {desc_c}"
            
            if is_sf_query:
                phrase_low = val_phrase.lower()
                if any(noise in phrase_low for noise in AIRFORCE_OR_NOISE_WORDS):
                    continue
                if not any(ind in phrase_low for ind in SF_INDICATORS):
                    continue

            if is_place_query:
                phrase_low = val_phrase.lower()
                if any(noise in phrase_low for noise in TRAVEL_NOISE):
                    continue
                if not any(ind in phrase_low for ind in PLACE_INDICATORS):
                    continue

            if _is_valid_entity_name(name_c):
                matched = False
                for idx, ex in enumerate(seen_leads):
                    if _are_entities_same(name_c, ex):
                        matched = True
                        break
                if not matched:
                    seen_leads.append(name_c)
                    candidates.append(val_phrase)
                    if len(candidates) >= requested_count:
                        return candidates[:requested_count]

    return candidates[:requested_count]


# ---------------------------------------------------------------------------
# Main Retrieval Loop with Arbitrary N Multi-Query ReAct
# ---------------------------------------------------------------------------

def retrieval_loop(
    question: str,
    requirements: InfoRequirements,
    intent_type: str = "DEFINITION",
    answer_style=None,
    operation_pattern: Optional[str] = None,
    max_iterations: int = 2,
) -> dict:
    """
    Runs an arbitrary N-query adaptive ReAct loop with semantic evidence reuse and aggregation.
    """
    # ── Path A0: LIST_ONLY & Ranking Candidate Harvesting Loop ─────────────
    if answer_style and answer_style.answer_mode == "LIST_ONLY" and answer_style.requested_count:
        requested_n = answer_style.requested_count
        print(f"[RetrievalLoop] List/Ranking Mode: harvesting >= {requested_n} distinct candidates for '{question}'")

        res1 = run_web_rag(
            question,
            intent_type=intent_type,
            answer_style=answer_style,
            operation_pattern=operation_pattern,
        )
        fm1 = res1.get("funnel_meta", {})
        all_pool_chunks = list(fm1.get("_all_chunks", []) or res1.get("_top3_chunks", []))
        all_pool_sources = list(fm1.get("_all_sources", []) or res1.get("_top3_sources", []))

        entities = _extract_canonical_entity_list(question, all_pool_chunks, requested_count=requested_n)

        if len(entities) < requested_n:
            expansion_queries = [
                f"{question} list of units",
                f"{question} full ranking",
                f"most famous {question}",
            ]
            for exp_q in expansion_queries:
                if len(entities) >= requested_n:
                    break
                try:
                    exp_res = run_web_rag(
                        exp_q,
                        intent_type=intent_type,
                        answer_style=answer_style,
                        operation_pattern=operation_pattern,
                    )
                    exp_fm = exp_res.get("funnel_meta", {})
                    all_pool_chunks.extend(exp_fm.get("_all_chunks", []) or exp_res.get("_top3_chunks", []))
                    all_pool_sources.extend(exp_fm.get("_all_sources", []) or exp_res.get("_top3_sources", []))
                    entities = _extract_canonical_entity_list(question, all_pool_chunks, requested_count=requested_n)
                except Exception as ex:
                    print(f"[RetrievalLoop] List candidate expansion failed: {ex}")

        print(f"[RetrievalLoop] Candidate Harvesting complete: found {len(entities)}/{requested_n} verified unique entities.")
        res1["all_web_chunks"] = all_pool_chunks
        res1["all_web_sources"] = all_pool_sources
        res1["harvested_entities"] = entities[:requested_n]
        return res1
    struct_reqs = getattr(requirements, "structured_requirements", [])

    # ── Path A: Multi-Query ReAct Loop (1 to N Sub-Queries) ────────────────
    if len(struct_reqs) >= 2:
        print(
            f"[RetrievalLoop] Multi-Query ReAct: Executing {len(struct_reqs)} sub-queries: "
            f"{[str(r) for r in struct_reqs]}"
        )
        evidence_graph = EvidenceGraph(requirements=struct_reqs)
        react_calls_count = 0
        last_result = None

        for req in struct_reqs:
            # 1. Semantic Check: Is Qi already satisfied by accumulated evidence?
            satisfied, extracted_ans = evidence_graph.check_requirement_satisfaction(req)
            if satisfied:
                print(
                    f"[RetrievalLoop -> Semantic Match] {req.id} ('{req.query_text}') "
                    f"already satisfied in evidence -> REUSING evidence (Skipping ReAct call)."
                )
                evidence_graph.add_evidence(req, [], [], resp=extracted_ans, reused=True)
                continue

            # 2. Not satisfied -> Execute focused ReAct search for Qi
            react_calls_count += 1
            sub_query = req.query_text
            print(
                f"[RetrievalLoop -> ReAct Sub-Query {react_calls_count}] "
                f"Targeting {req.id} -> '{sub_query}'"
            )

            try:
                sub_res = run_web_rag(
                    sub_query,
                    intent_type=intent_type,
                    answer_style=answer_style,
                    operation_pattern=operation_pattern,
                )
                sub_fm = sub_res.get("funnel_meta", {})
                sub_chunks = sub_fm.get("_all_chunks", []) or sub_res.get("_top3_chunks", [])
                sub_sources = sub_fm.get("_all_sources", []) or sub_res.get("_top3_sources", [])

                # Extract answer from this sub-query's chunks
                ans = _extract_direct_answer(sub_query, sub_chunks)

                # Add to evidence graph
                evidence_graph.add_evidence(req, sub_chunks, sub_sources, resp=ans, reused=False)
                last_result = sub_res
            except Exception as e:
                print(f"[RetrievalLoop] ReAct sub-query failed for '{sub_query}': {e}")

        print(
            f"[RetrievalLoop] Multi-Query ReAct completed: {react_calls_count} ReAct call(s) "
            f"made for {len(struct_reqs)} sub-queries."
        )

        if last_result and evidence_graph.accumulated_chunks:
            cov = _coverage_over_chunks(evidence_graph.accumulated_chunks, requirements)
            merged = dict(last_result)
            merged["retrieval_iterations"] = react_calls_count
            merged["final_coverage"] = cov["coverage_score"]
            merged["uncovered_concepts"] = cov["uncovered_concepts"]
            merged["all_web_chunks"] = evidence_graph.accumulated_chunks
            merged["all_web_sources"] = evidence_graph.accumulated_sources
            merged["evidence_graph"] = evidence_graph
            merged["sub_queries"] = [sq.to_dict() for sq in evidence_graph.sub_query_responses]
            return merged

    # ── Path B: Single-Concept Queries ─────────────────────────────────────
    if not requirements.requires_multi_retrieval:
        result = run_web_rag(
            question,
            intent_type=intent_type,
            answer_style=answer_style,
            operation_pattern=operation_pattern,
        )
        all_chunks = result.get("_top3_chunks") or []
        cov = _coverage_over_chunks(all_chunks, requirements)
        result["retrieval_iterations"] = 1
        result["final_coverage"] = cov["coverage_score"]
        result["uncovered_concepts"] = cov["uncovered_concepts"]
        result["all_web_chunks"] = all_chunks
        result["all_web_sources"] = result.get("_top3_sources") or []
        return result

    # ── Path C: Standard Concept Coverage Expansion ─────────────────────────
    result1 = run_web_rag(
        question,
        intent_type=intent_type,
        answer_style=answer_style,
        operation_pattern=operation_pattern,
    )

    fm1 = result1.get("funnel_meta", {})
    iter1_chunks = fm1.get("_all_chunks", []) or result1.get("_top3_chunks", [])
    iter1_sources = fm1.get("_all_sources", []) or result1.get("_top3_sources", [])

    cov1 = _coverage_over_chunks(iter1_chunks, requirements)
    if cov1["coverage_score"] >= requirements.coverage_threshold or max_iterations <= 1:
        result1["retrieval_iterations"] = 1
        result1["final_coverage"] = cov1["coverage_score"]
        result1["uncovered_concepts"] = cov1["uncovered_concepts"]
        result1["all_web_chunks"] = iter1_chunks
        result1["all_web_sources"] = iter1_sources
        return result1

    uncovered = cov1["uncovered_concepts"]
    extra_chunks: List[str] = []
    extra_sources: List[str] = []

    for concept in uncovered[:5]:
        concept_query = f"{concept} explained"
        try:
            sub_result = run_web_rag(
                concept_query,
                intent_type=intent_type,
                answer_style=answer_style,
                operation_pattern=operation_pattern,
            )
            sub_fm = sub_result.get("funnel_meta", {})
            sub_chunks = sub_fm.get("_all_chunks", []) or sub_result.get("_top3_chunks", [])
            sub_sources = sub_fm.get("_all_sources", []) or sub_result.get("_top3_sources", [])
            extra_chunks.extend(sub_chunks)
            extra_sources.extend(sub_sources)
        except Exception as exc:
            print(f"[RetrievalLoop] Expansion query '{concept_query}' failed: {exc}")

    all_chunks = iter1_chunks + extra_chunks
    all_sources = iter1_sources + extra_sources
    cov2 = _coverage_over_chunks(all_chunks, requirements)

    result2 = dict(result1)
    result2["retrieval_iterations"] = 2
    result2["final_coverage"] = cov2["coverage_score"]
    result2["uncovered_concepts"] = cov2["uncovered_concepts"]
    result2["all_web_chunks"] = all_chunks
    result2["all_web_sources"] = all_sources
    return result2
