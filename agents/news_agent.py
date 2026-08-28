"""
News Agent — Live News & Current Event Retrieval
Handles targeted queries regarding current events ("What happened today?").
"""

from typing import Any, Dict, Optional
from agents.search_tool import google_search, _duckduckgo_search


def fetch_live_news(query: str) -> Dict[str, Any]:
    """Fetches live news snippets targeting breaking news / current events."""
    search_q = f"news {query}"
    results = google_search(search_q)
    if not results:
        results = _duckduckgo_search(f"latest news {query}", num_results=5)

    snippets = []
    sources = []
    for r in (results or [])[:5]:
        title = r.title if hasattr(r, 'title') else r.get('title', '')
        link = r.link if hasattr(r, 'link') else r.get('link', '')
        snippet = r.snippet if hasattr(r, 'snippet') else r.get('snippet', '')
        if snippet:
            snippets.append(f"[{title}] ({link}): {snippet}")
            sources.append(link)

    summary = "\n".join(snippets) if snippets else "No recent breaking news articles found."
    return {
        "status": "success",
        "agent": "NewsAgent",
        "sources": sources,
        "summary": summary
    }
