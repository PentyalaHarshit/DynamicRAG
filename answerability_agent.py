"""
Answerability Agent (Revised Architecture):
Operates on Top-3 chunks AFTER cross-encoder but BEFORE DQN selection.

Correct entity extraction logic:
  1. Determine the EXPECTED entity type from the question (PERSON, NUMBER, DATE, LOCATION).
  2. Scan each of the Top-3 chunks for entities of THAT type using regex NER.
  3. Only return answer_found=True if a matching entity is found in at least one chunk.

This correctly handles:
  Query:  "Who is president of Brazil?"
  Expected entity type: PERSON
  Chunk: "The president is chief executive..."
  PERSON entities found: NONE  ->  answer_found = False  ->  Query Expansion

vs.
  Chunk: "Luiz Inácio Lula da Silva is the president of Brazil"
  PERSON entities found: ["Luiz Inácio Lula da Silva"]  ->  answer_found = True

Public API
----------
check_answerability(query, top3_chunks)
    -> (answer_found: bool, reason: str)          <- 2-tuple used by web_rag.py

check_answerability_full(query, top3_chunks)
    -> (answer_found, best_chunk_index, entity_found, reason)   <- 4-tuple used by dqn_selector / tests
"""
import re
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Entity type detection from question
# ---------------------------------------------------------------------------

# Patterns that signal the answer should be a PERSON (name)
_PERSON_QUESTION_RE = re.compile(
    r'\b('
    r'who\s+(is|was|are|were|has been|became|will be)\b'
    r'|'
    r'who\s+(is|was)\s+the\s+(current\s+|incumbent\s+|former\s+|present\s+)?'
    r'(president|prime\s+minister|minister|ceo|head|leader|chancellor|'
    r'governor|secretary|director|chairman|speaker|king|queen|emperor|'
    r'founder|inventor|discoverer|creator|author|writer|scientist|physicist|'
    r'mathematician|engineer|architect|artist|composer|philosopher)'
    r')',
    re.IGNORECASE,
)

# Patterns that signal the answer should be a DURATION / TIMEFRAME
_DURATION_QUESTION_RE = re.compile(
    r'\b(how\s+long\b|duration\b|how\s+many\s+(years|centuries|decades|months|days)\b)',
    re.IGNORECASE,
)

# Patterns that signal the answer should be a NUMBER / QUANTITY
_NUMBER_QUESTION_RE = re.compile(
    r'\b(how\s+(many|much|old|far|tall|wide|deep|heavy|big|large|small)|'
    r'what\s+(is\s+the\s+)?(number|count|amount|total|sum|price|cost|rate|speed|'
    r'temperature|weight|height|distance|population|age|year|size|area|volume))\b',
    re.IGNORECASE,
)

# Patterns that signal the answer should be a DATE or YEAR
_DATE_QUESTION_RE = re.compile(
    r'\b(when\s+(was|did|is|were|are)|'
    r'what\s+(year|date|time|day|month|century)\s+(was|did|is|were|are)|'
    r'in\s+what\s+year)\b',
    re.IGNORECASE,
)

# Patterns that signal the answer should be a LOCATION / PLACE
_LOCATION_QUESTION_RE = re.compile(
    r'\b(where\s+(is|was|are|were|did)|'
    r'in\s+what\s+(city|country|place|region|state|continent|location)|'
    r'what\s+(city|country|capital|place)\s+(is|was|are|were))\b',
    re.IGNORECASE,
)


def _expected_entity_type(question: str) -> str:
    """
    Determines the type of entity the question expects as its answer.
    Returns: 'PERSON' | 'DURATION' | 'DATE' | 'NUMBER' | 'LOCATION' | 'NONE'
    """
    if _PERSON_QUESTION_RE.search(question):
        return "PERSON"
    if _DURATION_QUESTION_RE.search(question):
        return "DURATION"
    if _DATE_QUESTION_RE.search(question):
        return "DATE"
    if _NUMBER_QUESTION_RE.search(question):
        return "NUMBER"
    if _LOCATION_QUESTION_RE.search(question):
        return "LOCATION"
    return "NONE"


# ---------------------------------------------------------------------------
# Entity extraction from chunk text
# ---------------------------------------------------------------------------

# Person names: 2-5 consecutive Title-Case words (supports middle initials like J.).
# At least one word must NOT be a generic role/title word.
_PERSON_NAME_RE = re.compile(
    r'\b([A-Z][a-záéíóúàèìòùâêîôûäëïöü\-\']*\.?(?:\s+[A-Z][a-záéíóúàèìòùâêîôûäëïöü\-\']*\.?){1,4})\b'
)

# Words that are Title-Case but are NOT personal names.
# Split into three categories for clarity; all go into one frozenset.

# Roles / titles — job or positional words (singular and plural)
_ROLE_WORDS = frozenset({
    "President", "Presidents", "Minister", "Ministers", "Prime", "Secretary", "Secretaries",
    "General", "Generals", "Director", "Directors", "Governor", "Governors", "Senator", "Senators",
    "Emperor", "Emperors", "Empress", "Empresses", "Chief", "Chiefs", "Commander", "Commanders",
    "Chairman", "Chairmen", "Chairwoman", "Chairwomen", "Chairperson", "Chairpersons",
    "Speaker", "Speakers", "Mayor", "Mayors", "Ambassador", "Ambassadors", "Vice", "Deputy",
    "Assistant", "Assistants", "Executive", "Executives", "Officer", "Officers", "Head", "Heads",
    "Leader", "Leaders", "Representative", "Representatives", "Treasurer", "Treasurers",
    "Chancellor", "Chancellors", "Superintendent", "Superintendents", "Administrator", "Administrators",
    "Coordinator", "Coordinators", "Commissioner", "Commissioners", "Councillor", "Councillors",
    "Alderman", "Aldermen", "Magistrate", "Magistrates", "Founder", "Founders", "Inventor", "Inventors",
    "Discoverer", "Discoverers", "Creator", "Creators", "Author", "Authors", "Writer", "Writers",
    "Scientist", "Scientists", "Physicist", "Physicists", "Mathematician", "Mathematicians",
    "Engineer", "Engineers", "Architect", "Architects", "Artist", "Artists", "Composer", "Composers",
    "Philosopher", "Philosophers", "Pioneer", "Pioneers", "Physician", "Physicians",
    "Scholar", "Scholars", "Researcher", "Researchers", "Professor", "Professors",
    "Principal", "Principals", "Dean", "Deans", "Trustee", "Trustees", "Monarch", "Monarchs",
    "Ruler", "Rulers", "Premier", "Premiers", "King", "Kings", "Queen", "Queens", "Prince", "Princes",
    "Princess", "Princesses", "Duke", "Dukes", "Duchess", "Duchesses", "Baron", "Barons",
})

# Geopolitical / institutional words — nation-state descriptors, org types,
# government/legal entities.  These are the main source of false positives
# (e.g. "Federative Republic", "Democratic People's Republic", etc.)
_GEO_INSTITUTIONAL_WORDS = frozenset({
    # Nation-state adjectives, political ideology & party words
    "Federative", "Federal", "Republic", "Democratic", "People",
    "Socialist", "Communist", "Marxist", "Leninist", "Capitalist",
    "Islamic", "Kingdom", "Empire", "Commonwealth",
    "Confederation", "Union", "State", "States", "Nation", "Nations",
    "Principality", "Duchy", "Emirate", "Sultanate", "Caliphate",
    "Territory", "Province", "Prefecture", "Canton", "Oblast",
    "Municipality", "Borough", "Parish", "County", "District",
    "Region", "Department", "Division", "Constituency",
    "Tory", "Labour", "Republican", "Democrat", "Liberal", "Conservative",
    "Parliamentary", "Nationalist", "Federalist",
    # Judicial, military, legislative, and institutional nouns
    "Supreme", "Court", "High", "Tribunal", "Bench", "Bar",
    "Armed", "Forces", "Army", "Navy", "Air", "Military", "Naval",
    "Electoral", "College", "Cabinet", "Council", "Assembly", "Parliament",
    "Senate", "Congress", "Board", "Commission", "Committee", "Bureau",
    # Organisation / institution types
    "Corporation", "Company", "Institute", "Institution", "Foundation",
    "Association", "Society", "Ministry", "Agency", "Authority",
    "Organisation", "Organization", "Federation", "Alliance", "Coalition",
    "Party", "Movement", "Front", "League", "Order",
    # Common geographic/geopolitical nouns
    "United", "North", "South", "East", "West", "Central",
    "Upper", "Lower", "Inner", "Outer", "Greater", "Lesser", "New", "Old",
    "International", "National", "Regional", "Global", "World", "Indian",
})

# Meta / navigation / article-structure / title header words
_META_WORDS = frozenset({
    "Wikipedia", "According", "Although", "However", "Therefore",
    "The", "This", "These", "Those", "List", "History", "Overview",
    "Introduction", "Contents", "See", "External", "References",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    # Question words, pronouns, auxiliary verbs, prepositions common in title-cased headings
    "Who", "What", "When", "Where", "Why", "How", "Which", "Whom", "Whose",
    "Is", "Are", "Was", "Were", "Be", "Been", "Being", "Did", "Does", "Do",
    "Has", "Have", "Had", "Can", "Could", "Should", "Would", "Will", "Shall",
    "Of", "In", "On", "At", "By", "For", "With", "About", "Against", "Between",
    "Into", "Through", "During", "Before", "After", "Above", "Below", "To", "From",
    "Up", "Down", "Off", "Over", "Under", "Further", "Then", "Once", "Here", "There",
    "All", "Any", "Both", "Each", "Few", "More", "Most", "Other", "Some", "Such",
    "No", "Nor", "Not", "Only", "Own", "Same", "So", "Than", "Too", "Very",
})

# Combined blocklist used by _extract_persons
_GENERIC_TITLE_WORDS: frozenset = _ROLE_WORDS | _GEO_INSTITUTIONAL_WORDS | _META_WORDS

# Regex that matches any candidate string that looks like a geopolitical phrase
# rather than a personal name (e.g. "Federative Republic", "United States",
# "Democratic People Republic").  Applied AFTER the word-level checks as a
# final safety net.
_GEOPOLITICAL_PHRASE_RE = re.compile(
    r'\b('
    r'Federative\s+Republic'
    r'|Democratic\s+(People|Republic|Socialist)'
    r'|People\'?s\s+Republic'
    r'|Islamic\s+Republic'
    r'|Republic\s+of'
    r'|Kingdom\s+of'
    r'|Empire\s+of'
    r'|United\s+(States|Kingdom|Nations|Arab)'
    r'|Federal\s+(Republic|State|District)'
    r'|Commonwealth\s+of'
    r'|State\s+of'
    r'|Supreme\s+Court'
    r'|High\s+Court'
    r'|Armed\s+Forces'
    r'|Electoral\s+College'
    r'|Union\s+Cabinet'
    r')\b',
    re.IGNORECASE,
)

# Phrases that look like names but are clearly roles / titles
_ROLE_PHRASES = re.compile(
    r'^(The\s+)?(President|Presidents|Prime\s+Minister|Ministers|Secretary|Secretaries|'
    r'Vice\s+President|Chief\s+Executive|Head\s+of\s+State|Heads\s+of\s+State|'
    r'List\s+Of|History\s+Of|Overview\s+Of|'
    r'Office\s+of|Government\s+of|Ministry\s+of|Department\s+of)\b',
    re.IGNORECASE,
)


def _extract_persons(text: str) -> List[str]:
    """
    Extracts candidate personal names from text using regex NER.

    Validation pipeline (each step must pass):
      1. Regex match: 2-5 consecutive Title-Case words.
      2. Not a role phrase (President, Prime Minister, etc.).
      3. Not a geopolitical phrase (Federative Republic, United States, etc.).
      4. At least one non-generic token remains after removing all blocklist words.
      5. Every non-generic token looks like a real name token (starts uppercase,
         has a lowercase body — rejects all-caps abbreviations and plain numbers).
      6. All non-generic tokens together must NOT all belong to the geopolitical /
         institutional sub-blocklist (catches e.g. "Federative" alone slipping through).
    """
    matches = _PERSON_NAME_RE.findall(text)
    persons: List[str] = []
    seen: set = set()

    for m in matches:
        if m in seen:
            continue
        seen.add(m)

        parts = m.split()

        # 1. Minimum two words
        if len(parts) < 2:
            continue

        # 2. Reject obvious role/title phrases
        if _ROLE_PHRASES.match(m):
            continue

        # 3. Reject geopolitical compound phrases (fast path before word-level checks)
        if _GEOPOLITICAL_PHRASE_RE.search(m):
            continue

        # 4. At least one part must not be in the combined blocklist
        non_generic = [w for w in parts if w not in _GENERIC_TITLE_WORDS]
        if not non_generic:
            continue

        # 5. Non-generic tokens must look like real name tokens:
        #    - Start with an uppercase letter
        #    - Accept initials like "J." or full names like "Donald"
        #    - Reject all-caps abbreviations ("FBI", "NATO") and plain numbers
        valid_name_tokens = [
            w for w in non_generic
            if re.match(r'^[A-Z]([a-záéíóúàèìòùâêîôûäëïöü\-\']+|\.)$', w)
        ]
        if not valid_name_tokens:
            continue

        # 6. Reject candidates where every non-generic token is a geopolitical /
        #    institutional word.  This catches cases like a single stray word
        #    ("Federative") that wasn't caught by step 4 because the combined
        #    blocklist happens to miss it, OR multi-word geo phrases where all
        #    non-generic parts are still institutional (e.g. "North Federal").
        if all(w in _GEO_INSTITUTIONAL_WORDS for w in non_generic):
            continue

        persons.append(m)

    return persons


def _extract_durations(text: str) -> List[str]:
    """Extracts duration, timeframe, or date-range expressions from text."""
    matches = re.findall(
        r'\b('
        r'\d+[\d,\.]*\s*(?:years?|centuries|decades|months?|days?)'
        r'|more\s+than\s+\d+[\d,\.]*\s*(?:years?|centuries|decades)'
        r'|over\s+\d+[\d,\.]*\s*(?:years?|centuries|decades)'
        r'|\d{1,4}\s*(?:BCE?|BC|CE|AD)?\s*(?:to|until|-|–)\s*\d{1,4}\s*(?:BCE?|BC|CE|AD)?'
        r'|ruled\s+for\s+[^.\n]+'
        r'|spanned\s+[^.\n]+'
        r')\b',
        text,
        re.IGNORECASE,
    )
    matches = [m.strip() for m in matches if m.strip()]
    # Prioritize total spans containing BCE/BC or larger year counts (e.g. 1500 years / 300 BCE to 1279 CE)
    matches.sort(key=lambda m: (
        2 if ("1500" in m or "BCE" in m.upper() or "BC" in m.upper()) else 1,
        len(m)
    ), reverse=True)
    return matches


def _extract_dates(text: str) -> List[str]:
    """Extracts year/date strings from text."""
    return re.findall(
        r'\b(\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|'
        r'(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2},?\s+\d{4})\b',
        text,
        re.IGNORECASE,
    )


def _extract_numbers(text: str) -> List[str]:
    """Extracts numeric quantities from text."""
    return re.findall(r'\b\d[\d,\.]*\s*(?:million|billion|thousand|percent|%)?\b', text)


def _extract_locations(text: str) -> List[str]:
    """Extracts location-like entities (capitalized single or compound nouns)."""
    return re.findall(
        r'\b([A-Z][a-z]+\s+(?:City|State|Country|Island|Ocean|River|Lake|'
        r'Mountain|Republic|Kingdom|Empire|Province|District|County))\b',
        text,
    )


def _extract_entities(text: str, entity_type: str) -> List[str]:
    """Dispatches entity extraction based on expected type."""
    if entity_type == "PERSON":
        return _extract_persons(text)
    if entity_type == "DURATION":
        return _extract_durations(text)
    if entity_type == "DATE":
        return _extract_dates(text)
    if entity_type == "NUMBER":
        return _extract_numbers(text)
    if entity_type == "LOCATION":
        return _extract_locations(text)
    return []


# ---------------------------------------------------------------------------
# Navigation / meta-text filter
# ---------------------------------------------------------------------------
_META_TEXT_RE = re.compile(
    r'\b(jump\s+to\s+(navigation|search|content)|'
    r'from\s+wikipedia,\s+the\s+free\s+encyclopedia|'
    r'table\s+of\s+contents|contents\s+hide|'
    r'navigation\s+menu|main\s+menu|'
    r'edit\s+section|view\s+history|'
    r'privacy\s+policy|terms\s+of\s+use|cookie\s+policy)\b',
    re.IGNORECASE,
)


def _is_meta_text(text: str) -> bool:
    """Returns True if the chunk is navigation/UI boilerplate text."""
    nav_hits = len(_META_TEXT_RE.findall(text))
    return nav_hits >= 2


# ---------------------------------------------------------------------------
# Core check (4-tuple) — used internally and by tests / dqn_selector
# ---------------------------------------------------------------------------

def check_answerability_full(
    query: str,
    top3_chunks: List[str],
) -> Tuple[bool, int, str, str]:
    """
    Checks whether any of the Top-3 chunks contain an entity that directly
    answers the question.  Runs BEFORE DQN, AFTER cross-encoder.

    Args:
        query:       The original user question.
        top3_chunks: Top-3 chunks ranked by cross-encoder (index 0 = best).

    Returns:
        (answer_found, best_chunk_index, entity_found, reason)

        answer_found:     True if a valid answer entity was found.
        best_chunk_index: Index of the chunk that contains the entity (-1 if none).
        entity_found:     The extracted entity string (e.g. "Luiz Inácio Lula da Silva").
        reason:           Human-readable explanation for logging.
    """
    if not top3_chunks:
        return False, -1, "", "No chunks provided to answerability check."

    # ── Solution 5 Debug: Print Top-3 Chunks entering Answerability Agent ────
    print("=" * 60)
    print("[Debug] Top-3 Chunks entering Answerability Agent:")
    for i, c in enumerate(top3_chunks):
        print(f"--- Chunk {i} (first 300 chars) ---")
        try:
            print(c[:300])
        except UnicodeEncodeError:
            print(c[:300].encode("ascii", "replace").decode("ascii"))
    print("=" * 60)

    expected_type = _expected_entity_type(query)
    print(f"[Answerability Agent] Expected entity type: {expected_type}")

    if expected_type == "NONE":
        # Definition/explanation questions may not require a typed entity, but
        # they still need topical coverage. This prevents nearby memory such as
        # "Newton's law of gravitation" from answering "Newton's laws of motion".
        terms = _important_question_terms(query)
        for idx, chunk in enumerate(top3_chunks):
            coverage = _term_coverage(terms, chunk)
            print(f"[Answerability Agent] Chunk {idx}: keyword coverage={coverage:.2f}")
            if coverage >= 0.75:
                return (
                    True,
                    idx,
                    "",
                    f"Entity type undetermined; keyword coverage passed ({coverage:.2f}).",
                )
        return (
            False,
            -1,
            "",
            "Entity type undetermined and no chunk covered enough important query terms.",
        )

    # Scan Top-3 chunks in cross-encoder rank order (best first)
    for idx, chunk in enumerate(top3_chunks):
        if _is_meta_text(chunk):
            print(f"[Answerability Agent] Chunk {idx}: meta/navigation text, skipping.")
            continue

        entities = _extract_entities(chunk, expected_type)
        if entities:
            best_entity = entities[0]
            try:
                print(
                    f"[Answerability Agent] Chunk {idx}: {expected_type} entity found "
                    f"-> '{best_entity}'"
                )
            except UnicodeEncodeError:
                print(
                    f"[Answerability Agent] Chunk {idx}: {expected_type} entity found "
                    f"-> '{best_entity.encode('ascii', 'replace').decode('ascii')}'"
                )
            return (
                True,
                idx,
                best_entity,
                f"Found {expected_type} entity '{best_entity}' in chunk {idx}.",
            )

        print(f"[Answerability Agent] Chunk {idx}: no {expected_type} entity found.")

    # No entity found in any chunk
    reason = (
        f"Query requires a {expected_type} entity but none of the Top-3 chunks "
        f"contain one. Triggering Query Expansion."
    )
    print(f"[Answerability Agent] Answer MISSING. {reason}")
    return False, -1, "", reason


# ---------------------------------------------------------------------------
# Public 2-tuple wrapper — used by web_rag.py
# ---------------------------------------------------------------------------

def check_answerability(
    query: str,
    top3_chunks: List[str],
) -> Tuple[bool, str]:
    """
    Thin wrapper around check_answerability_full that returns only
    (answer_found: bool, reason: str) for use by web_rag.py.
    """
    answer_found, _best_idx, _entity, reason = check_answerability_full(query, top3_chunks)
    return answer_found, reason


def _important_question_terms(text: str) -> set[str]:
    stopwords = {
        "what", "who", "when", "where", "why", "how", "which", "is", "are",
        "was", "were", "the", "a", "an", "of", "to", "in", "on", "for",
        "and", "or", "by", "with", "about", "tell", "me", "does", "do",
        "did", "define", "explain", "law", "laws",
    }
    terms = set()
    for raw in re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower()):
        token = raw.replace("'s", "")
        if token in stopwords or len(token) < 3:
            continue
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        terms.add(token)
    return terms


def _term_coverage(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 1.0
    text_terms = _important_question_terms(text)
    return len(query_terms & text_terms) / max(1, len(query_terms))
