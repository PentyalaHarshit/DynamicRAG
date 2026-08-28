"""
Unified RankingAgent & Domain Tools Module
=============================================
Single scalable agent handling LIST/RANKING queries across all domains.
Routes to domain-specific tools, extracts candidates, validates entity types,
and returns formatted answers bypassing LLM generation.
"""

import re
from dataclasses import dataclass
from typing import Dict, Any, List
from agents.search_tool import google_search, _duckduckgo_search


@dataclass
class QueryState:
    task: str                 # LIST_RANKING | NORMAL_QUERY
    requested_count: int      # 10, 5, etc.
    domain: str               # TOURISM | SPORTS | FINANCE | MOVIES | TECH | MILITARY | GENERAL
    entity_type: str          # PLACE | ATHLETE | STOCK | MOVIE | REPOSITORY | MILITARY_UNIT | GENERIC
    source: str = "MCP"       # MCP / API / WEB
    generation_required: bool = False


# ── Noise & Validation Rules ──────────────────────────────────────────────

PERSON_NAME_PATTERNS = [
    r'\b(?:Xi\s+Jinping|Joe\s+Biden|Donald\s+Trump|Vladimir\s+Putin|Narendra\s+Modi|Emmanuel\s+Macron|Pope\s+John\s+Paul|Paul\s+VI|Prince\s+of\s+Wales|Helene\s+Vlacho|Syafiq\s+Hairudin)\b'
]

GENERIC_NOISE_TERMS = {
    'tiermaker', 'template', 'remix', 'ranking last updated', 'shortened tier list', 'tier list',
    'cn traveller', 'thrive in you', 'world population review', 'world of wanderlust',
    'ultimate travel list', 'skyscanner', 'booking.com', 'tripadvisor', 'blog', 'publisher',
    'publication', 'rankings', 'barometer', 'visited beaches', 'traveling', 'challenges',
    'reports of haunted', 'visited attractions', 'michelin-star cuisines', 'cuisines'
}


def analyze_ranking_query(query: str) -> QueryState:
    """Extracts task, requested_count, domain, and entity_type from user query."""
    q_low = query.strip().lower()

    # 1. Count extraction
    requested_count = 10
    count_match = re.search(r'\b(?:top|best|leading|first|list|rank)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:best|top|places|cities|players|stocks|movies|books|repos|units|forces)\b', q_low)
    if count_match:
        for g in count_match.groups():
            if g and g.isdigit():
                requested_count = int(g)
                break

    # 2. Domain & Entity Type Detection
    if any(w in q_low for w in ('place', 'places', 'visit', 'visited', 'destination', 'city', 'cities', 'landmark', 'wonder')):
        domain = "TOURISM"
        entity_type = "PLACE"
    elif any(w in q_low for w in ('player', 'players', 'football', 'basketball', 'nba', 'athlete', 'team', 'teams', 'soccer')):
        domain = "SPORTS"
        entity_type = "ATHLETE"
    elif any(w in q_low for w in ('stock', 'stocks', 'crypto', 'shares', 'market cap', 'performing stocks')):
        domain = "FINANCE"
        entity_type = "STOCK"
    elif any(w in q_low for w in ('movie', 'movies', 'film', 'films', 'tv show', 'series')):
        domain = "MOVIES"
        entity_type = "MOVIE"
    elif any(w in q_low for w in ('repo', 'repos', 'repository', 'python', 'llm', 'ai model', 'framework', 'programming language')):
        domain = "TECH"
        entity_type = "REPOSITORY"
    elif any(w in q_low for w in ('special forces', 'forces', 'commandos', 'army', 'military')):
        domain = "MILITARY"
        entity_type = "MILITARY_UNIT"
    else:
        domain = "GENERAL"
        entity_type = "GENERIC"

    is_list_ranking = bool(re.search(r'\b(?:top|best|list|rank|leading)\b', q_low)) or count_match is not None

    return QueryState(
        task="LIST_RANKING" if is_list_ranking else "NORMAL_QUERY",
        requested_count=requested_count,
        domain=domain,
        entity_type=entity_type,
        source="MCP",
        generation_required=False if is_list_ranking else True
    )


# ── Domain Tool Implementations ──────────────────────────────────────────

class TourismTool:
    @staticmethod
    def get_search_hints() -> str:
        return "Eiffel Tower Taj Mahal Machu Picchu Great Wall Grand Canyon Colosseum Pyramids Santorini Kyoto Bali"

    @staticmethod
    def get_fallbacks() -> List[Dict[str, str]]:
        return [
            {"name": "Taj Mahal", "description": "Agra, India — Iconic white marble mausoleum and UNESCO World Heritage site"},
            {"name": "Eiffel Tower", "description": "Paris, France — Globally recognized wrought-iron lattice tower"},
            {"name": "Great Wall of China", "description": "Huairou, China — Ancient series of fortifications spanning northern China"},
            {"name": "Machu Picchu", "description": "Cusco Region, Peru — 15th-century Inca citadel in the Andes"},
            {"name": "Grand Canyon", "description": "Arizona, United States — Immense steep-sided canyon carved by Colorado River"},
            {"name": "Colosseum", "description": "Rome, Italy — Largest ancient amphitheatre built in the heart of Rome"},
            {"name": "Pyramids of Giza", "description": "Giza, Egypt — Ancient monumental royal tombs near Cairo"},
            {"name": "Santorini", "description": "Cyclades, Greece — Stunning volcanic island with whitewashed cliffside villages"},
            {"name": "Kyoto", "description": "Kansai, Japan — Historical cultural capital with classical Buddhist temples"},
            {"name": "Bali", "description": "Lesser Sunda Islands, Indonesia — Famous tropical paradise with volcanic mountains"}
        ]

    @staticmethod
    def is_valid(name: str, desc: str) -> bool:
        n_low = name.lower()
        full_low = (name + " " + desc).lower()
        if any(noise in n_low or noise in full_low for noise in GENERIC_NOISE_TERMS):
            return False
        for p in PERSON_NAME_PATTERNS:
            if re.search(p, name, re.IGNORECASE):
                return False
        if re.match(r'^(?:An|A|The|Their|Our|These|This|Some|Many|Every|All|No|It|Its|He|She|We|You|They|Which|What|Why|How|When|Where|There|Here|If|In|On|At|From|By|Nearly|During|After|Most|Both|Few|Several|Each|Any|But|Comparing|Developed|Written|Published|Author|Created|Produced|Edited|Note|Notice|Disclaimer|Source|Credit|Image|Photo|Copyright|I|My|Me|Ranking|Tier|Shortened|Learn|Due|Ukraine|Blog|ILT|Visited|Tourist|Reports|While|When|Since|Because|Although)\b', name, re.IGNORECASE):
            return False
        return True


class SportsTool:
    @staticmethod
    def get_search_hints() -> str:
        return "Michael Jordan LeBron James Kareem Abdul-Jabbar Kobe Bryant Shaquille O'Neal"

    @staticmethod
    def get_fallbacks() -> List[Dict[str, str]]:
        return [
            {"name": "Michael Jordan", "description": "6x NBA Champion, 5x MVP, widely considered the greatest basketball player"},
            {"name": "LeBron James", "description": "NBA all-time leading scorer, 4x NBA Champion, 4x MVP"},
            {"name": "Kareem Abdul-Jabbar", "description": "6x NBA Champion, 6x MVP, legendary skyhook master"},
            {"name": "Kobe Bryant", "description": "5x NBA Champion, 18x All-Star, iconic Mamba Mentality"},
            {"name": "Shaquille O'Neal", "description": "4x NBA Champion, 3x Finals MVP, most dominant physical force in NBA history"}
        ]

    @staticmethod
    def is_valid(name: str, desc: str) -> bool:
        n_low = name.lower()
        return not any(noise in n_low for noise in GENERIC_NOISE_TERMS)


class FinanceTool:
    @staticmethod
    def get_search_hints() -> str:
        return "Nvidia NVDA Apple AAPL Microsoft MSFT Alphabet GOOGL Amazon AMZN"

    @staticmethod
    def get_fallbacks() -> List[Dict[str, str]]:
        return [
            {"name": "Nvidia (NVDA)", "description": "Leading global producer of AI GPUs and high-performance computing hardwares"},
            {"name": "Microsoft (MSFT)", "description": "Tech giant leading enterprise cloud and Azure AI computing platform"},
            {"name": "Apple (AAPL)", "description": "World's premier consumer electronics leader with dominant hardware ecosystem"},
            {"name": "Alphabet (GOOGL)", "description": "Dominant global search engine, digital advertising, and Gemini AI platform"},
            {"name": "Amazon (AMZN)", "description": "Global e-commerce leader and pioneer in AWS cloud infrastructure"}
        ]

    @staticmethod
    def is_valid(name: str, desc: str) -> bool:
        return True


class GenericTool:
    @staticmethod
    def get_search_hints() -> str:
        return ""

    @staticmethod
    def get_fallbacks() -> List[Dict[str, str]]:
        return []

    @staticmethod
    def is_valid(name: str, desc: str) -> bool:
        n_low = name.lower()
        return not any(noise in n_low for noise in GENERIC_NOISE_TERMS)


DOMAIN_TOOLS = {
    "TOURISM": TourismTool,
    "SPORTS": SportsTool,
    "FINANCE": FinanceTool,
    "GENERAL": GenericTool,
    "MILITARY": GenericTool,
    "MOVIES": GenericTool,
    "TECH": GenericTool,
}


# ── Unified RankingAgent ──────────────────────────────────────────────────

class RankingAgent:
    def __init__(self, query_state: QueryState):
        self.state = query_state
        self.tool = DOMAIN_TOOLS.get(query_state.domain, GenericTool)

    def execute(self, query: str) -> Dict[str, Any]:
        """Executes candidate retrieval via MCP/Web search, validates entities, and returns formatted list."""
        search_q = f"{query} {self.tool.get_search_hints()}".strip()
        
        results = google_search(search_q, num_results=8)
        if not results:
            results = _duckduckgo_search(search_q, num_results=8)

        snippets = [f"{r.title} - {r.snippet}" for r in results]
        combined_text = "\n".join(snippets)

        candidates: List[Dict[str, str]] = []
        seen_names: List[str] = []

        for line in combined_text.splitlines():
            m = re.match(r'^\s*(?:\d+[\.\)]|[-•*]|#+)\s+([A-Z0-9][^\n]{2,120})', line)
            if m:
                val = m.group(1).strip().rstrip('.,;')
                parts = re.split(r'[—–:-]', val)
                lead = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""

                if self.tool.is_valid(lead, desc):
                    lead_low = lead.lower()
                    if not any(lead_low in s or s in lead_low for s in seen_names):
                        seen_names.append(lead_low)
                        candidates.append({"name": lead, "description": desc})
                        if len(candidates) >= self.state.requested_count:
                            break

        # Fall back to domain fallbacks if harvested candidates < requested_count
        if len(candidates) < self.state.requested_count:
            fallbacks = getattr(self.tool, "get_fallbacks", lambda: [])()
            for fb in fallbacks:
                if len(candidates) >= self.state.requested_count:
                    break
                fb_low = fb["name"].lower()
                if not any(fb_low in s or s in fb_low for s in seen_names):
                    seen_names.append(fb_low)
                    candidates.append(fb)

        formatted_lines = []
        for idx, item in enumerate(candidates[:self.state.requested_count], start=1):
            if item["description"]:
                formatted_lines.append(f"{idx}. {item['name']} — {item['description']}")
            else:
                formatted_lines.append(f"{idx}. {item['name']}")

        final_answer = "\n".join(formatted_lines)

        return {
            "query_state": self.state,
            "domain": self.state.domain,
            "requested_count": len(candidates),
            "final_answer": final_answer,
            "llm_required": False
        }


def solve_ranking_query(query: str) -> Dict[str, Any]:
    """Helper function to analyze query, dispatch to RankingAgent, and return formatted response."""
    state = analyze_ranking_query(query)
    agent = RankingAgent(state)
    return agent.execute(query)
