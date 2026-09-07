"""
Sports Agent — resolves sports queries (who won, live scores, results, champions).

Uses TheSportsDB free API (no key required) as the primary data source,
then synthesizes a clean natural language answer via the LLM if needed.

Examples handled:
  - "Who won IPL 2020?"
  - "Who won the 2022 FIFA World Cup?"
  - "NBA Finals 2023 winner"
  - "India vs Pakistan cricket score"
  - "Who is the current NFL champion?"
"""
from typing import Any, Dict

from sports_api import get_sports_data
from llm_client import call_llm


_SPORTS_SYNTHESIS_SYSTEM = """You are a sports facts expert. You have been given structured sports data from TheSportsDB API.
Your task is to write a clear, accurate, and concise answer to the user's question.

Rules:
1. If the data clearly shows a winner, START your answer with "[Team/Player Name] won the [Tournament] [Year]."
2. Include the score if available.
3. Keep the answer to 2-3 sentences.
4. Do NOT add speculative information beyond what is provided.
5. Do NOT mention "TheSportsDB" or API details in your answer.
"""


def solve_sports_query(question: str) -> Dict[str, Any]:
    """
    Main entrypoint for the Sports Agent.

    1. Calls TheSportsDB API via sports_api.get_sports_data()
    2. If a clean winner is found, builds the answer directly.
    3. If the API result is ambiguous, uses LLM to synthesize from the event data.

    Returns a structured dict with final_answer, winner, events, route metadata.
    """
    api_result = get_sports_data(question)
    winner = api_result.get("winner")
    events = api_result.get("events", [])
    api_answer = api_result.get("answer", "")
    parsed = api_result.get("parsed", {})

    print(
        f"[SportsAgent] Parsed: league={parsed.get('league')}, "
        f"year={parsed.get('year')}, team={parsed.get('team')}"
    )
    print(f"[SportsAgent] API winner: {winner!r} | events found: {len(events)}")

    # If the API returned a clear winner-led answer, use it directly
    if winner and api_answer and api_answer.startswith(f"**{winner}**"):
        final_answer = api_answer
        print(f"[SportsAgent] Direct API answer used.")
    elif events or api_answer:
        # Synthesize using LLM with the structured data as context
        context_parts = [f"Question: {question}"]

        if winner:
            context_parts.append(f"Winner identified by API: {winner}")

        if events:
            context_parts.append("\nMatch/Event data:")
            for ev in events[:3]:
                ev_line = (
                    f"  Event: {ev.get('event_name', 'Unknown')}"
                    f" | Date: {ev.get('date', 'N/A')}"
                    f" | {ev.get('home_team', '')} vs {ev.get('away_team', '')}"
                    f" | Score: {ev.get('score', 'N/A')}"
                    f" | Winner: {ev.get('winner', 'N/A')}"
                    f" | Venue: {ev.get('venue', '')}"
                )
                context_parts.append(ev_line)

        context = "\n".join(context_parts)
        prompt = (
            f"{context}\n\n"
            f"Based on the above sports data, answer this question clearly: {question}\n"
            f"If a winner is identified, start your answer with the winner's name."
        )
        final_answer = call_llm(system=_SPORTS_SYNTHESIS_SYSTEM, prompt=prompt)
        print(f"[SportsAgent] LLM-synthesized answer.")
    else:
        # No data found — fallback to web search suggestion
        final_answer = (
            f"I could not retrieve real-time sports data for your query. "
            f"Please check ESPN, BBC Sport, or Cricinfo for the latest results."
        )
        print(f"[SportsAgent] No data found — fallback response.")

    return {
        "final_answer": final_answer,
        "winner": winner,
        "events": events,
        "parsed_query": parsed,
        "api_source": "thesportsdb",
        "status": "success" if (winner or events) else "no_data",
    }
