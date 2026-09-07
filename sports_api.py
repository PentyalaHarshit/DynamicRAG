"""
Sports data lookup using TheSportsDB free REST API.
No API key required for v1 endpoints.
API docs: https://www.thesportsdb.com/api.php

Public endpoints used:
  - Search events by name:  /api/v1/json/3/searchevents.php?e={name}
  - Search for teams:       /api/v1/json/3/searchteams.php?t={name}
  - League seasons results: /api/v1/json/3/eventsseason.php?id={league_id}&s={season}
  - Last 5 events by team:  /api/v1/json/3/eventslast.php?id={team_id}
  - Search events by round: /api/v1/json/3/eventsround.php?id={league_id}&r={round}&s={season}
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

_BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"
_TIMEOUT = 10

# ── Well-known league IDs on TheSportsDB ────────────────────────────────────
_LEAGUE_IDS: Dict[str, str] = {
    # Cricket
    "ipl":                     "4501",   # Indian Premier League
    "indian premier league":   "4501",
    "cricket world cup":       "4506",
    "icc cricket world cup":   "4506",
    "big bash":                "4506",   # fallback
    # Football / Soccer
    "fifa world cup":          "4328",
    "world cup":               "4328",
    "premier league":          "4328",
    "epl":                     "4328",
    "la liga":                 "4335",
    "bundesliga":              "4331",
    "serie a":                 "4332",
    "champions league":        "4480",
    "uefa champions league":   "4480",
    # Basketball
    "nba":                     "4387",
    "nba finals":              "4387",
    # American Football
    "nfl":                     "4391",
    "super bowl":              "4391",
    # Baseball
    "mlb":                     "4424",
    "world series":            "4424",
    # Formula 1
    "formula 1":               "4370",
    "f1":                      "4370",
    # Tennis
    "wimbledon":               "4607",
    "us open tennis":          "4605",
    # Rugby
    "rugby world cup":         "4515",
    # Hockey
    "nhl":                     "4380",
    "stanley cup":             "4380",
}

# ── Season format helpers ────────────────────────────────────────────────────
def _year_to_season(year: str, league: str) -> str:
    """Convert a year string to TheSportsDB season format."""
    y = int(year)
    league_lower = league.lower()
    # Cricket (IPL) and NBA use single-year or YYYY-YYYY
    if any(k in league_lower for k in ("ipl", "cricket", "nba", "nfl", "mlb", "nhl")):
        return str(y)
    # Football/Soccer European leagues: YYYY-YYYY
    return f"{y}-{y+1}"


# ── Query parser ─────────────────────────────────────────────────────────────
def parse_sports_query(question: str) -> Dict[str, Optional[str]]:
    """
    Extract sport/league, team, season/year from a natural language question.

    Returns dict with keys: league, year, team, query_text
    """
    q = question.strip()
    q_lower = q.lower()

    # Extract year (4-digit)
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", q)
    year = year_match.group(1) if year_match else None

    # Detect league (longest match wins)
    detected_league: Optional[str] = None
    for league_name in sorted(_LEAGUE_IDS.keys(), key=len, reverse=True):
        if league_name in q_lower:
            detected_league = league_name
            break

    # Detect team names (any word sequence of 1-3 words after certain keywords)
    team: Optional[str] = None
    team_match = re.search(
        r"(?:did\s+)?([A-Z][a-zA-Z\s]{2,30}?)\s+(?:win|won|beat|defeat|champion)",
        q,
    )
    if team_match:
        candidate = team_match.group(1).strip()
        if len(candidate.split()) <= 5 and candidate.lower() not in ("who", "which team"):
            team = candidate

    return {
        "league": detected_league,
        "year": year,
        "team": team,
        "query_text": q,
    }


# ── API helpers ───────────────────────────────────────────────────────────────
def _get(endpoint: str, params: Dict[str, str]) -> Dict[str, Any]:
    """Make a GET request to TheSportsDB and return parsed JSON."""
    url = f"{_BASE_URL}/{endpoint}"
    resp = requests.get(url, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def search_events_by_name(event_name: str) -> List[Dict[str, Any]]:
    """Search for events by name (e.g. 'IPL 2020 final')."""
    data = _get("searchevents.php", {"e": event_name})
    return data.get("event") or []


def search_teams_by_name(team_name: str) -> List[Dict[str, Any]]:
    """Search for teams by name."""
    data = _get("searchteams.php", {"t": team_name})
    return data.get("teams") or []


def get_last_events_by_team_name(team_name: str) -> List[Dict[str, Any]]:
    """Get last 5 events for a team (by name lookup then ID)."""
    teams = search_teams_by_name(team_name)
    if not teams:
        return []
    team_id = teams[0].get("idTeam", "")
    if not team_id:
        return []
    data = _get("eventslast.php", {"id": team_id})
    return data.get("results") or []


def get_season_events(league_name: str, year: str) -> List[Dict[str, Any]]:
    """Get all events for a league/season."""
    league_id = _LEAGUE_IDS.get(league_name.lower())
    if not league_id:
        return []
    season = _year_to_season(year, league_name)
    data = _get("eventsseason.php", {"id": league_id, "s": season})
    return data.get("events") or []


def _summarize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the essential fields from a TheSportsDB event object."""
    home = event.get("strHomeTeam", "")
    away = event.get("strAwayTeam", "")
    score_home = event.get("intHomeScore")
    score_away = event.get("intAwayScore")
    winner = None

    if score_home is not None and score_away is not None:
        try:
            sh, sa = int(score_home), int(score_away)
            if sh > sa:
                winner = home
            elif sa > sh:
                winner = away
            else:
                winner = "Draw / Tie"
        except (ValueError, TypeError):
            winner = event.get("strResult", None)
    else:
        # Some events store result as text
        result_text = event.get("strResult") or event.get("strDescription") or ""
        if result_text:
            winner = result_text

    return {
        "event_name": event.get("strEvent", ""),
        "date": event.get("dateEvent", ""),
        "home_team": home,
        "away_team": away,
        "score": f"{score_home}-{score_away}" if score_home is not None else "N/A",
        "winner": winner,
        "venue": event.get("strVenue", ""),
        "season": event.get("strSeason", ""),
        "league": event.get("strLeague", ""),
        "status": event.get("strStatus", ""),
    }


# ── Main entrypoint ───────────────────────────────────────────────────────────
def get_sports_data(question: str) -> Dict[str, Any]:
    """
    Main entrypoint: parse the query, call TheSportsDB API, return structured result.

    Returns dict with:
        answer   (str)  — formatted natural language answer
        events   (list) — raw event summaries
        winner   (str)  — the winning team/player (if determinable)
        source   (str)  — "thesportsdb"
        error    (str)  — populated only on failure
    """
    parsed = parse_sports_query(question)
    league = parsed["league"]
    year = parsed["year"]
    team = parsed["team"]
    q = parsed["query_text"]

    events: List[Dict[str, Any]] = []
    winner: Optional[str] = None
    answer = ""

    # Strategy 1: Search events by name directly
    # Build a good search query from the question parts
    search_terms_parts = []
    if league:
        search_terms_parts.append(league.upper() if league in ("ipl", "nba", "nfl", "mlb", "nhl", "f1") else league.title())
    if year:
        search_terms_parts.append(year)
    if not search_terms_parts:
        # Fallback: use significant words from question
        stop = {"who", "won", "which", "team", "the", "did", "win", "is", "are", "was", "were", "a", "an"}
        search_terms_parts = [w for w in q.split() if w.lower() not in stop and len(w) > 2][:4]

    search_query = " ".join(search_terms_parts)
    raw_events = search_events_by_name(search_query) if search_query else []

    # Filter for final/championship events first, then any event
    finals = [e for e in raw_events if any(kw in (e.get("strEvent") or "").lower() for kw in ("final", "finals", "championship", "champion"))]
    relevant = finals if finals else raw_events

    if relevant:
        events = [_summarize_event(e) for e in relevant[:5]]
        # Prefer events that have a score/result
        scored = [ev for ev in events if ev["score"] != "N/A" and ev["winner"]]
        if scored:
            top = scored[0]
            winner = top["winner"]
            answer = _build_answer(top, q)

    # Strategy 2: Season events (if we have league + year)
    if not winner and league and year:
        season_events = get_season_events(league, year)
        # Find the final/championship
        final_events = [
            e for e in season_events
            if any(kw in (e.get("strEvent") or "").lower() for kw in ("final", "champion", "winner"))
        ]
        target_events = final_events or season_events[-3:]  # last 3 events of season
        if target_events:
            ev_summaries = [_summarize_event(e) for e in target_events[:5]]
            events = ev_summaries
            scored = [ev for ev in ev_summaries if ev["score"] != "N/A" and ev["winner"]]
            if scored:
                top = scored[0]
                winner = top["winner"]
                answer = _build_answer(top, q)

    # Strategy 3: Team last events (if team name found)
    if not winner and team:
        team_events = get_last_events_by_team_name(team)
        if team_events:
            ev_summaries = [_summarize_event(e) for e in team_events[:3]]
            events = ev_summaries
            if ev_summaries:
                top = ev_summaries[0]
                answer = _build_answer(top, q)
                winner = top.get("winner")

    if not answer:
        answer = (
            f"I could not find specific sports data for your query. "
            f"Please try asking with more specific terms like the league name and year."
        )

    return {
        "answer": answer,
        "winner": winner,
        "events": events,
        "parsed": parsed,
        "source": "thesportsdb",
    }


def _build_answer(event: Dict[str, Any], question: str) -> str:
    """Build a natural language answer from a summarized event."""
    winner = event.get("winner", "")
    home = event.get("home_team", "")
    away = event.get("away_team", "")
    score = event.get("score", "N/A")
    event_name = event.get("event_name", "")
    date = event.get("date", "")
    venue = event.get("venue", "")
    league = event.get("league", "")
    season = event.get("season", "")

    if not winner:
        return f"The event '{event_name}' was found but the result is not yet available."

    # Lead with winner (matches the "who won" format users expect)
    parts = [f"**{winner}** won"]

    # Add tournament/event name context
    if event_name:
        parts.append(f"the **{event_name}**")
    elif league:
        display = f"**{league}**"
        if season:
            display += f" ({season})"
        parts.append(display)

    answer = " ".join(parts) + "."

    # Add score context if available
    if score != "N/A" and score != "None-None":
        answer += f" Score: {home} {score} {away}."

    # Add date/venue if useful
    if date:
        answer += f" Played on {date}"
        if venue:
            answer += f" at {venue}"
        answer += "."

    return answer
