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
    r'tiktok\s+video|generated\s+by\s+ai|#capcut|#midjourney|#aiart|'
    r'please\s+mark\s+me\s+as\s+brainliest|explore\s+all\s+similar\s+answers|'
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
        # Within the main content area, gather paragraph and code block elements
        elements = content_tag.find_all(['p', 'pre', 'code'])
        if elements:
            blocks = []
            for el in elements:
                t = el.get_text().strip()
                if el.name in ('pre', 'code') and len(t.split()) >= 2:
                    blocks.append(f"```python\n{t}\n```")
                elif t:
                    blocks.append(t)
            text = '\n\n'.join(blocks)
        else:
            text = content_tag.get_text(separator=' ')
    else:
        # Universal fallback: all <p>, <pre>, <code> tags in the document
        elements = soup.find_all(['p', 'pre', 'code'])
        blocks = []
        for el in elements:
            t = el.get_text().strip()
            if el.name in ('pre', 'code') and len(t.split()) >= 2:
                blocks.append(f"```python\n{t}\n```")
            elif t:
                blocks.append(t)
        text = '\n\n'.join(blocks) if blocks else soup.get_text(separator=' ')

    return re.sub(r'[ \t]+', ' ', text).strip()


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
    Fallback web search using DDGS package when Google CSE keys are not set.
    Returns live web search results (fitness plans, news, current facts) with titles, links, and snippets.
    """
    results: List[SearchResult] = []
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            _JUNK_DOMAINS = re.compile(
                r'(?i)\b(?:tiktok\.com|pinterest\.com|instagram\.com|facebook\.com|'
                r'twitter\.com|x\.com|youtube\.com|dailymotion\.com|wonderslist\.com|'
                r'solatatech\.com|therichest\.com|toptenz\.net)\b'
            )
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=num_results + 5))
                for item in raw_results:
                    title = item.get("title", "")
                    link = item.get("href", "")
                    snippet = item.get("body", "")
                    if _JUNK_DOMAINS.search(link) or _JUNK_DOMAINS.search(title):
                        continue
                    if title and snippet:
                        results.append(SearchResult(title=title, link=link, snippet=snippet))
                    if len(results) >= num_results:
                        break

        if results:
            print(f"[Search Tool] DDGS live web search returned {len(results)} web results for '{query}'")
            return results[:num_results]
    except Exception as e:
        print(f"[Search Tool] DDGS live web search failed: {e}")

    # Fallback to direct HTML parsing if DDGS is unavailable
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
                    if "/l/?" in link:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(link).query)
                        if "uddg" in parsed:
                            link = parsed["uddg"][0]
                    snippet = a_snippet.get_text().strip() if a_snippet else ""
                    if _JUNK_DOMAINS.search(link) or _JUNK_DOMAINS.search(title):
                        continue
                    if title and snippet:
                        results.append(SearchResult(title=title, link=link, snippet=snippet))
                    if len(results) >= num_results:
                        break
    except Exception:
        pass

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
    Combined fallback search: queries DDGS and includes authoritative Wikipedia articles
    for topic depth and reliability.
    """
    ddg_results = _duckduckgo_search(query, num_results)
    
    # For ranking, comparison, or military queries, enrich pool with authoritative Wikipedia sources
    if any(w in query.lower() for w in ("special forces", "military", "best", "strongest", "top", "compare")):
        wiki_results = _wikipedia_search(query, num_results=5)
        combined = list(ddg_results)
        seen_links = {r.link for r in combined}
        for wr in wiki_results:
            if wr.link not in seen_links:
                seen_links.add(wr.link)
                combined.append(wr)
        return combined[:num_results + 3]

    if ddg_results:
        return ddg_results

    return _wikipedia_search(query, num_results)


# ---------------------------------------------------------------------------
# Live currency exchange rate  (no API key needed)
# ---------------------------------------------------------------------------

# Maps common currency names / symbols to ISO 4217 codes
_CURRENCY_ALIASES: dict = {
    # Symbols
    "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY",
    "₹": "INR", "₩": "KRW", "₺": "TRY", "₽": "RUB",
    "฿": "THB",
    # Full names (lower)
    "dollar": "USD", "dollars": "USD",
    "euro": "EUR", "euros": "EUR",
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP",
    "yen": "JPY",
    "rupee": "INR", "rupees": "INR",
    "yuan": "CNY", "renminbi": "CNY",
    "won": "KRW",
    "franc": "CHF", "francs": "CHF",
    "peso": "MXN", "pesos": "MXN",
    "ruble": "RUB", "rubles": "RUB",
    "dirham": "AED",
    "riyal": "SAR",
    "baht": "THB",
    "ringgit": "MYR",
    "lira": "TRY",
}

# Regex to pull amount + from-currency + to-currency out of a query string
# Works on the already-shell-processed string ($ stripped by shell → digit only)
_CURRENCY_PARSE_RE = re.compile(
    r"""
    (?:(?P<symbol>[$€£¥₹₩₺₽฿])\s*)?          # optional leading symbol
    (?P<amount>\d[\d,\.]*)?                    # optional amount
    \s*
    (?P<from_code>[A-Z]{3}|                    # ISO code OR
        dollar s?|euro s?|pound s?|yen|        # English names
        rupee s?|yuan|won|franc s?|peso s?|
        ruble s?|dirham|riyal|baht|ringgit|
        lira|sterling)
    \s+
    (?:in|to|into|=|->)\s+
    (?P<to_code>[A-Z]{3}|
        dollar s?|euro s?|pound s?|yen|
        rupee s?|yuan|won|franc s?|peso s?|
        ruble s?|dirham|riyal|baht|ringgit|
        lira|sterling)
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Alternate form: "how many rupees in a dollar" / "how many dollars is 500 rupees"
_CURRENCY_HOW_MANY_RE = re.compile(
    r"how\s+many\s+(?P<to_name>\w+)\s+(?:is|are|in|to|per|for)\s+"
    r"(?:a\s+|an\s+|one\s+)?(?:(?P<amount>\d[\d,\.]*)\s+)?(?P<from_name>\w+)",
    re.IGNORECASE,
)


def _resolve_currency(token: str, symbol: str = "") -> str:
    """Convert a currency name, symbol, or ISO code to an uppercase ISO 4217 code."""
    if not token:
        token = ""
    t = token.strip().lower().rstrip("s")   # normalise plural → singular
    # Direct alias lookup
    code = _CURRENCY_ALIASES.get(t) or _CURRENCY_ALIASES.get(symbol)
    if code:
        return code
    # If it already looks like an ISO code (3 uppercase letters), use it directly
    upper = token.strip().upper()
    if re.fullmatch(r"[A-Z]{3}", upper):
        return upper
    return ""


def parse_currency_query(query: str) -> tuple[float, str, str]:
    """
    Parse a currency query string into (amount, from_code, to_code).

    Returns (0.0, "", "") when parsing fails.

    Examples
    --------
    "100 USD in INR"       → (100.0, "USD", "INR")
    "$50 to euros"         → (50.0,  "USD", "EUR")
    "convert 200 GBP to JPY" → (200.0, "GBP", "JPY")
    "how many rupees in a dollar" → (1.0,  "USD", "INR")
    " USD in INR"          → (1.0,   "USD", "INR")   ← shell stripped the $
    """
    # Strip shell-expanded noise: leading spaces, standalone "in INR?"
    q = query.strip()

    # Try the main pattern first
    m = _CURRENCY_PARSE_RE.search(q)
    if m:
        raw_amount = (m.group("amount") or "1").replace(",", "")
        amount = float(raw_amount) if raw_amount else 1.0
        symbol = m.group("symbol") or ""
        from_code = _resolve_currency(m.group("from_code"), symbol)
        to_code   = _resolve_currency(m.group("to_code"))
        if from_code and to_code:
            return amount, from_code, to_code

    # Try "how many X in a Y"
    m2 = _CURRENCY_HOW_MANY_RE.search(q)
    if m2:
        raw_amount = (m2.group("amount") or "1").replace(",", "")
        amount = float(raw_amount) if raw_amount else 1.0
        from_code = _resolve_currency(m2.group("from_name"))
        to_code   = _resolve_currency(m2.group("to_name"))
        if from_code and to_code:
            return amount, from_code, to_code

    return 0.0, "", ""


def fetch_live_exchange_rate(
    from_currency: str,
    to_currency: str,
    amount: float = 1.0,
) -> dict:
    """
    Fetches the live exchange rate from the free open.er-api.com endpoint.
    No API key required for the v6 free tier (1 500 req/month).

    Returns a dict with:
        rate          — float, the exchange rate (1 unit of from → X units of to)
        converted     — float, amount * rate
        from_currency — str, ISO code
        to_currency   — str, ISO code
        amount        — float
        source        — str, attribution
        answer        — str, human-readable answer sentence

    Raises RuntimeError on HTTP / parse failure (caller should catch).
    """
    from_currency = from_currency.upper()
    to_currency   = to_currency.upper()

    url = f"https://open.er-api.com/v6/latest/{from_currency}"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise RuntimeError(f"Exchange rate API failed: {exc}") from exc

    if data.get("result") != "success":
        raise RuntimeError(
            f"Exchange rate API returned non-success: {data.get('result')}"
        )

    rates = data.get("rates", {})
    if to_currency not in rates:
        raise RuntimeError(
            f"Currency '{to_currency}' not found in rates. "
            f"Available: {', '.join(list(rates.keys())[:10])} …"
        )

    rate = float(rates[to_currency])
    converted = round(amount * rate, 4)

    # Format the converted amount nicely
    if converted == int(converted):
        converted_str = f"{int(converted):,}"
    else:
        converted_str = f"{converted:,.4f}".rstrip("0").rstrip(".")

    if amount == 1.0:
        answer = (
            f"1 {from_currency} = {rate:,.4f} {to_currency}. "
            f"Exchange rate as of {data.get('time_last_update_utc', 'today')} "
            f"(source: open.er-api.com)."
        )
    else:
        # Format the input amount
        if amount == int(amount):
            amt_str = f"{int(amount):,}"
        else:
            amt_str = f"{amount:,g}"
        answer = (
            f"{amt_str} {from_currency} = {converted_str} {to_currency}. "
            f"(Rate: 1 {from_currency} = {rate:,.4f} {to_currency}, "
            f"as of {data.get('time_last_update_utc', 'today')} — open.er-api.com)"
        )

    return {
        "rate":          rate,
        "converted":     converted,
        "from_currency": from_currency,
        "to_currency":   to_currency,
        "amount":        amount,
        "source":        "open.er-api.com",
        "answer":        answer,
    }


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
