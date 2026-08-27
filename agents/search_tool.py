"""
Search Tool — Web Retrieval with Clean Content Extraction
==========================================================

Extraction pipeline (per URL):

  URL
   │
   ├── wikipedia.org? ──► Wikipedia REST API (clean article text, no HTML)
   │                         /api/rest_v1/page/summary   -> lead paragraph
   │                         /w/api.php?prop=extracts     -> full article text
   │
   └── other ──────────► BeautifulSoup targeted extraction
                             Priority: <article> -> <main> -> <div role=main>
                             Fallback:  all <p> tags concatenated
                             (sidebars, nav, footer already removed)

Content quality filter (_is_content_chunk):
  Rejects a chunk if it looks like navigation/menu text:
    - nav-keyword density too high  (jump to, see also, contents, …)
    - too many short lines           (link lists, bullet menus)
    - too few words

This means the embedding model, cross-encoder, DQN, and Answerability Agent
only ever see article-body paragraphs — never "Jump to content / Navigation /
See also / List of …".
"""
from dataclasses import dataclass
from typing import List, Optional
import re
import requests

import config


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str


# ---------------------------------------------------------------------------
# Navigation / junk pattern detector
# Used both as a chunk-level filter AND inside _extract_main_content
# ---------------------------------------------------------------------------

_NAV_PHRASES = re.compile(
    r'\b(jump\s+to|jump\s+to\s+content|jump\s+to\s+navigation|'
    r'from\s+wikipedia|the\s+free\s+encyclopedia|'
    r'table\s+of\s+contents|contents\s+hide|'
    r'navigation\s+menu|main\s+menu|main\s+page|'
    r'see\s+also|external\s+links|further\s+reading|'
    r'retrieved\s+from|this\s+page\s+was\s+last\s+edited|'
    r'privacy\s+policy|terms\s+of\s+use|cookie\s+policy|'
    r'creative\s+commons|wikimedia\s+foundation|'
    r'edit\s+source|view\s+history|talk\s+page|'
    r'search\s+wikipedia|log\s+in|create\s+account)\b',
    re.IGNORECASE,
)


def _is_content_chunk(text: str, min_words: int = 30) -> bool:
    """
    Returns True if the chunk looks like real article content.

    Rejection criteria (ANY one fails the chunk):
      1. Fewer than min_words words.
      2. More than 3 nav-phrase matches (it is a navigation section).
      3. More than 50% of lines are very short (<= 4 words) — link/menu lists.
    """
    words = text.split()
    if len(words) < min_words:
        return False

    nav_hits = len(_NAV_PHRASES.findall(text))
    if nav_hits > 3:
        return False

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        short_lines = sum(1 for l in lines if len(l.split()) <= 4)
        if short_lines / len(lines) > 0.50:
            return False

    return True


# ---------------------------------------------------------------------------
# Wikipedia REST API — clean article text, zero HTML scraping
# ---------------------------------------------------------------------------

_WIKI_TITLE_RE = re.compile(
    r'https?://(?:[a-z]+\.)?wikipedia\.org/wiki/([^#?]+)', re.IGNORECASE
)


def _wikipedia_title_from_url(url: str) -> Optional[str]:
    """Extracts the Wikipedia article title from a URL, or returns None."""
    m = _WIKI_TITLE_RE.match(url)
    if m:
        return requests.utils.unquote(m.group(1).replace('_', ' '))
    return None


def _fetch_wikipedia_clean(url: str, max_chars: int = 6000) -> str:
    """
    Uses the Wikipedia MediaWiki API to fetch the plain-text extract of an
    article.  Returns clean article prose — no navigation, no HTML, no
    infobox markup.

    Strategy:
      1. Parse the article title from the URL.
      2. Call /w/api.php with prop=extracts&exintro=false&explaintext=true
         which returns the full article as plain text, section by section.
      3. Return up to max_chars of that text.

    Falls back to empty string on any error so the caller can try HTML
    scraping instead.
    """
    title = _wikipedia_title_from_url(url)
    if not title:
        return ""

    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":       "query",
                "prop":         "extracts",
                "titles":       title,
                "exintro":      False,      # include body, not just intro
                "explaintext":  True,       # plain text, not HTML
                "exsectionformat": "plain", # no == Section == markers
                "format":       "json",
                "redirects":    1,
            },
            timeout=4,
            headers={"User-Agent": "HybridRAG/1.0 (research project)"},
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                # Strip section headers like '== See also ==', '== References ==', etc.
                extract = re.sub(r'==\s*(See also|References|External links|Notes|Further reading)\s*==.*$', '', extract, flags=re.DOTALL | re.IGNORECASE)
                # Collapse excessive blank lines
                extract = re.sub(r'\n{3,}', '\n\n', extract).strip()
                print(
                    f"[Search Tool] Wikipedia API: '{title}' -> "
                    f"{len(extract)} chars clean text"
                )
                return extract[:max_chars]
    except Exception as e:
        print(f"[Search Tool] Wikipedia API failed for '{title}': {e}")

    return ""


# ---------------------------------------------------------------------------
# General HTML -> article text extractor (non-Wikipedia pages)
# ---------------------------------------------------------------------------

def _extract_main_content(html: str) -> str:
    """
    Extracts only the main article body from arbitrary HTML.

    Priority:
      1. <article> tag (most semantic sites)
      2. <main> tag or <div role="main">
      3. All <p> tags concatenated (universal fallback)

    Nav, aside, header, footer, script, style are stripped before any
    extraction attempt.  Returns clean whitespace-normalised text.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # No BeautifulSoup — fall back to regex strip
        text = re.sub(r'<[^>]+>', ' ', html)
        return re.sub(r'\s+', ' ', text).strip()

    soup = BeautifulSoup(html, "html.parser")

    # ── Remove all non-content elements first ─────────────────────────────
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe", "button",
                     "input", "select", "textarea", "figure", "figcaption",
                     "table"]):   # tables often hold infobox/navigation grids
        tag.decompose()

    # Remove elements with nav/ad-flavoured class or id names
    _JUNK_ATTR = re.compile(
        r'nav|menu|sidebar|footer|header|banner|cookie|popup|promo|'
        r'ad[\-_]|advertisement|breadcrumb|toc|mw-jump|mw-navigation|'
        r'mw-head|catlinks|contentSub|siteSub|jump-to-nav|mw-editsection|'
        r'printfooter|mw-data-after-content|coordinates',
        re.IGNORECASE,
    )
    for tag in soup.find_all(True):
        if getattr(tag, 'decomposed', False):
            continue
        try:
            raw_cls = tag.get('class') if hasattr(tag, 'get') else None
            cls = ' '.join(raw_cls) if isinstance(raw_cls, list) else str(raw_cls or '')
            tid = str(tag.get('id') if hasattr(tag, 'get') else '')
            role = str(tag.get('role') if hasattr(tag, 'get') else '')
            if (
                _JUNK_ATTR.search(cls)
                or _JUNK_ATTR.search(tid)
                or role in ('navigation', 'banner', 'complementary', 'contentinfo')
            ):
                tag.decompose()
        except Exception:
            continue

    # ── Extract main content in priority order ────────────────────────────
    content_tag = (
        soup.find('article')
        or soup.find('main')
        or soup.find(attrs={'role': 'main'})
        or soup.find('div', id=re.compile(r'^(content|main|bodyContent|mw-content-text)', re.I))
    )

    if content_tag:
        # Within the main content area, gather all paragraph text
        paragraphs = content_tag.find_all('p')
        if paragraphs:
            text = ' '.join(p.get_text(separator=' ') for p in paragraphs)
        else:
            text = content_tag.get_text(separator=' ')
    else:
        # Universal fallback: all <p> tags in the document
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(separator=' ') for p in paragraphs)

    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Page fetching — orchestrates Wikipedia API vs HTML extraction
# ---------------------------------------------------------------------------

def fetch_page_text(url: str, max_chars: int = 6000) -> str:
    """
    Fetches a URL and returns clean article-body text up to max_chars.

    For Wikipedia URLs: uses the Wikipedia MediaWiki API (plain text, no nav).
    For all other URLs: downloads HTML and extracts the main content area.
    """
    # ── Wikipedia: use the API, never scrape ──────────────────────────────
    if 'wikipedia.org' in url:
        text = _fetch_wikipedia_clean(url, max_chars=max_chars)
        if text:
            return text
        # API failed — fall through to HTML scraping as last resort

    # ── General pages: targeted HTML extraction ───────────────────────────
    try:
        resp = requests.get(
            url,
            timeout=4,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return ""
        text = _extract_main_content(resp.text)
        return text[:max_chars]
    except Exception as e:
        print(f"[Search Tool] fetch_page_text({url}) failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Chunk extraction with content-quality filter
# ---------------------------------------------------------------------------

def _ensure_complete_sentence(text: str) -> str:
    """
    Ensures a chunk ends with a complete sentence (full stop, question mark, or exclamation).
    If the chunk doesn't end with punctuation, truncates to the last complete sentence.
    """
    text = text.strip()
    if not text:
        return text
    
    # If already ends with sentence-ending punctuation, return as-is
    if text[-1] in '.!?':
        return text
    
    # Find the last sentence-ending punctuation
    for i in range(len(text) - 1, -1, -1):
        if text[i] in '.!?':
            return text[:i + 1]
    
    # No punctuation found — add a period
    return text + '.'


def extract_chunks_from_page(
    url: str,
    chunk_words: int = 500,
    max_chunks: int = 6,
) -> List[str]:
    """
    Fetches a page, splits the article text into clean paragraph-aligned chunks,
    and returns only chunks that pass the _is_content_chunk quality filter.
    
    Each chunk is guaranteed to:
    - End with a complete sentence (full stop, question mark, or exclamation)
    - Be at least chunk_words in length (when possible)
    - Respect paragraph boundaries
    """
    text = fetch_page_text(url, max_chars=chunk_words * max_chunks * 6)
    if not text:
        return []

    # Split by double newlines first to respect paragraph boundaries (Solution 1 & 2)
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip().split()) >= 15]

    raw_chunks: List[str] = []
    current_words: List[str] = []

    for p in paragraphs:
        p_words = p.split()
        if len(current_words) + len(p_words) <= chunk_words:
            current_words.extend(p_words)
        else:
            if current_words:
                chunk_text = " ".join(current_words)
                chunk_text = _ensure_complete_sentence(chunk_text)
                raw_chunks.append(chunk_text)
            current_words = p_words[:]

    if current_words:
        chunk_text = " ".join(current_words)
        chunk_text = _ensure_complete_sentence(chunk_text)
        raw_chunks.append(chunk_text)

    # Apply quality filter — reject nav/menu/link-list chunks
    good_chunks: List[str] = []
    for chunk in raw_chunks:
        if _is_content_chunk(chunk):
            good_chunks.append(chunk)
            if len(good_chunks) >= max_chunks:
                break
        else:
            print(
                f"[Search Tool] Chunk rejected by quality filter "
                f"({len(chunk.split())} words, "
                f"nav_hits={len(_NAV_PHRASES.findall(chunk))}): "
                f"'{chunk[:80]}...'"
            )

    return good_chunks


# ---------------------------------------------------------------------------
# Fallback search (DuckDuckGo HTML + Wikipedia API)
# ---------------------------------------------------------------------------

def _duckduckgo_search(query: str, num_results: int = 10) -> List[SearchResult]:
    """
    Fallback web search using DuckDuckGo HTML search when Google CSE keys are not set.
    Returns live web search results (news, current facts) with titles, links, and snippets.
    """
    results: List[SearchResult] = []
    try:
        resp = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
            timeout=10,
        )
        if resp.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for result in soup.find_all("div", class_=re.compile(r"result\b")):
                a_title = result.find("a", class_="result__a")
                a_snippet = (
                    result.find("a", class_="result__snippet")
                    or result.find("div", class_="result__snippet")
                )
                if a_title:
                    title = a_title.get_text().strip()
                    link = a_title.get("href", "")
                    # Clean duckduckgo redirect link if present
                    if "/l/?" in link:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                        if "uddg" in parsed:
                            link = parsed["uddg"][0]
                    snippet = a_snippet.get_text().strip() if a_snippet else ""
                    if title and snippet:
                        results.append(SearchResult(title=title, link=link, snippet=snippet))
        if results:
            print(f"[Search Tool] DuckDuckGo fallback: returned {len(results)} live web results")
    except Exception as e:
        print(f"[Search Tool] DuckDuckGo fallback failed: {e}")

    return results[:num_results]


def _wikipedia_search(query: str, num_results: int = 10) -> List[SearchResult]:
    """
    Fallback search using public Wikipedia API.
    Strips artificial year suffixes (e.g. '2026') to prevent zero-result returns.
    """
    clean_query = re.sub(r'\b(202[0-9]|2030)\b', '', query).strip()
    clean_query = re.sub(r'\s+', ' ', clean_query)

    results: List[SearchResult] = []
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":   "query",
                "list":     "search",
                "srsearch": clean_query,
                "format":   "json",
                "srlimit":  num_results,
            },
            timeout=10,
            headers={"User-Agent": "HybridRAG/1.0"},
        )
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("query", {}).get("search", []):
                title   = item.get("title", "")
                snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                results.append(SearchResult(title=title, link=page_url, snippet=snippet))
    except Exception as e:
        print(f"[Search Tool] Wikipedia fallback failed: {e}")

    return results[:num_results]


def _fallback_search(query: str, num_results: int = 10) -> List[SearchResult]:
    """
    Combined fallback search: tries DuckDuckGo live web search first,
    then tops up with Wikipedia search if needed.
    """
    ddg_results = _duckduckgo_search(query, num_results)
    if len(ddg_results) >= num_results:
        return ddg_results

    # Top up with Wikipedia
    needed = num_results - len(ddg_results)
    wiki_results = _wikipedia_search(query, needed)
    seen_links = {r.link for r in ddg_results}
    combined = ddg_results + [r for r in wiki_results if r.link not in seen_links]
    return combined[:num_results]


# ---------------------------------------------------------------------------
# Primary search entry point
# ---------------------------------------------------------------------------

def google_search(query: str, num_results: int = None) -> List[SearchResult]:
    """
    Google Custom Search JSON API with web fallback (DuckDuckGo + Wikipedia).
    Default 10 results for a proper Top-10 pool.
    """
    num_results = num_results if num_results is not None else config.GOOGLE_SEARCH_NUM_RESULTS

    if not config.GOOGLE_API_KEY or not config.GOOGLE_CSE_ID:
        print("[Search Tool] Google CSE keys not set – using web search fallback...")
        return _fallback_search(query, num_results)

    try:
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": config.GOOGLE_API_KEY,
                "cx":  config.GOOGLE_CSE_ID,
                "q":   query,
                "num": min(num_results, 10),
            },
            timeout=10,
        )
        resp.raise_for_status()
        results = [
            SearchResult(
                title=i.get("title", ""),
                link=i.get("link", ""),
                snippet=i.get("snippet", ""),
            )
            for i in resp.json().get("items", [])
        ]
        if len(results) < num_results:
            extra = _fallback_search(query, num_results - len(results))
            existing = {r.link for r in results}
            results += [r for r in extra if r.link not in existing]
        return results[:num_results]
    except Exception as e:
        print(f"[Search Tool] Google API failed ({e}). Using web search fallback.")
        return _fallback_search(query, num_results)


# ---------------------------------------------------------------------------
# MCP tool schema
# ---------------------------------------------------------------------------

GOOGLE_SEARCH_TOOL_SCHEMA = {
    "name": "google_search",
    "description": (
        "Search the public web via Google Custom Search and return the top "
        "results as (title, link, snippet) objects."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query":       {"type": "string",  "description": "The search query"},
            "num_results": {"type": "integer", "description": "How many results (max 10)"},
        },
        "required": ["query"],
    },
}
