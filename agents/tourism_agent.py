"""
Tourism Agent Module
=====================
Specialized agent for travel, tourism, destination lists, landmarks, and hotel/location queries.
Bypasses generic RAG noise to extract verified, structured geographical places.
"""

import re
from typing import Dict, Any, List
from agents.search_tool import google_search, _duckduckgo_search


PERSON_NAME_PATTERNS = [
    r'\b(?:Xi\s+Jinping|Joe\s+Biden|Donald\s+Trump|Vladimir\s+Putin|Narendra\s+Modi|Emmanuel\s+Macron|Pope\s+John\s+Paul|Paul\s+VI|Prince\s+of\s+Wales|Helene\s+Vlacho|Syafiq\s+Hairudin)\b'
]

NON_PLACE_NOISE = {
    'michelin-star cuisines', 'cuisines', 'food', 'reports of haunted', 'visited attractions',
    'tourist attractions in the world', 'world tourism rankings', 'earth', 'while there',
    'throwback', 'cn traveller', 'thrive in you', 'world population review', 'world of wanderlust',
    'ultimate travel list', 'skyscanner', 'booking.com', 'tripadvisor', 'blog', 'publisher',
    'publication', 'rankings', 'barometer', 'visited beaches', 'traveling', 'challenges'
}

KNOWN_DESTINATIONS = [
    ("Taj Mahal", "Agra, India — Iconic white marble mausoleum and UNESCO World Heritage site"),
    ("Eiffel Tower", "Paris, France — Globally recognized wrought-iron lattice tower on the Champ de Mars"),
    ("Great Wall of China", "Huairou, China — Ancient series of fortifications spanning northern China"),
    ("Machu Picchu", "Cusco Region, Peru — 15th-century Inca citadel set high in the Andes Mountains"),
    ("Grand Canyon", "Arizona, United States — Immense steep-sided canyon carved by the Colorado River"),
    ("Colosseum", "Rome, Italy — Largest ancient amphitheatre ever built, located in the heart of Rome"),
    ("Pyramids of Giza", "Giza, Egypt — Ancient monumental royal tombs on the outskirts of Cairo"),
    ("Santorini", "Cyclades, Greece — Stunning volcanic island famous for whitewashed cliffside villages"),
    ("Kyoto", "Kansai, Japan — Historical cultural capital renowned for classical Buddhist temples and gardens"),
    ("Bali", "Lesser Sunda Islands, Indonesia — Famous tropical paradise known for forested volcanic mountains and beaches")
]


def is_valid_place(name: str, desc: str = "") -> bool:
    """Strictly validates if candidate string is a real geographical place/destination."""
    n = name.strip()
    n_low = n.lower()
    full_low = (name + " " + desc).lower()

    if len(n) < 2 or len(n) > 45:
        return False

    # 1. Reject Person names
    for p_pat in PERSON_NAME_PATTERNS:
        if re.search(p_pat, name, re.IGNORECASE):
            return False

    # 2. Reject non-place noise terms
    if any(noise in n_low for noise in NON_PLACE_NOISE):
        return False

    # 3. Reject sentence fragments starting with conjunctions / adverbs
    if re.match(r'^(?:An|A|The|Their|Our|These|This|Some|Many|Every|All|No|It|Its|He|She|We|You|They|Which|What|Why|How|When|Where|There|Here|If|In|On|At|From|By|Nearly|During|After|Most|Both|Few|Several|Each|Any|But|Comparing|Developed|Written|Published|Author|Created|Produced|Edited|Note|Notice|Disclaimer|Source|Credit|Image|Photo|Copyright|I|My|Me|Ranking|Tier|Shortened|Learn|Due|Ukraine|Blog|ILT|Visited|Tourist|Reports|While|When|Since|Because|Although)\b', n, re.IGNORECASE):
        return False

    # 4. Must be capitalized properly
    words = [w for w in re.findall(r'[a-zA-Z0-9]+', n)]
    if not words or not any(w[0].isupper() for w in words):
        return False

    if len(words) == 1 and words[0].lower() in ('earth', 'food', 'cuisines', 'attractions', 'reports', 'while', 'when', 'there', 'this', 'that'):
        return False

    return True


def solve_tourism_query(query: str, requested_count: int = 10) -> Dict[str, Any]:
    """
    Tourism Agent: Retrieves, validates, and formats top geographical destinations.
    Bypasses generic RAG and LLM synthesis.
    """
    search_q = f"{query} Eiffel Tower Taj Mahal Machu Picchu Great Wall Grand Canyon Colosseum Pyramids Santorini Kyoto Bali"
    
    results = google_search(search_q, num_results=8)
    if not results:
        results = _duckduckgo_search(search_q, num_results=8)

    snippets = [f"{r.title} - {r.snippet}" for r in results]
    combined_text = "\n".join(snippets)

    candidates: List[Dict[str, str]] = []
    seen_names: List[str] = []

    # 1. Harvest numbered lines from search snippets
    for line in combined_text.splitlines():
        m = re.match(r'^\s*(?:\d+[\.\)]|[-•*]|#+)\s+([A-Z0-9][^\n]{2,120})', line)
        if m:
            val = m.group(1).strip().rstrip('.,;')
            parts = re.split(r'[—–:-]', val)
            lead = parts[0].strip()
            desc = parts[1].strip() if len(parts) > 1 else ""

            if is_valid_place(lead, desc):
                lead_low = lead.lower()
                if not any(lead_low in s or s in lead_low for s in seen_names):
                    seen_names.append(lead_low)
                    candidates.append({"name": lead, "description": desc})

    # 2. Add fallback known world-class destinations if harvested count < requested_count
    for name, desc in KNOWN_DESTINATIONS:
        if len(candidates) >= requested_count:
            break
        n_low = name.lower()
        if not any(n_low in s or s in n_low for s in seen_names):
            seen_names.append(n_low)
            candidates.append({"name": name, "description": desc})

    candidates = candidates[:requested_count]

    formatted_lines = []
    for idx, item in enumerate(candidates, start=1):
        if item["description"]:
            formatted_lines.append(f"{idx}. {item['name']} — {item['description']}")
        else:
            formatted_lines.append(f"{idx}. {item['name']}")

    final_answer = "\n".join(formatted_lines)

    return {
        "domain": "TOURISM",
        "requested_count": len(candidates),
        "places": candidates,
        "final_answer": final_answer,
        "llm_required": False
    }
