"""Structured stock quote lookup for the finance agent."""
from datetime import datetime, timezone
from typing import Any, Dict

import requests


_YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

_COMPANY_TICKERS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "berkshire hathaway": "BRK-B",
}


def resolve_ticker(company_or_ticker: str) -> str:
    """Resolve a common company name or ticker to a Yahoo Finance symbol."""
    cleaned = company_or_ticker.strip(" .,?!'\"").lower()
    cleaned = cleaned.removesuffix("'s").strip()
    if cleaned in _COMPANY_TICKERS:
        return _COMPANY_TICKERS[cleaned]
    if cleaned.upper().replace("-", "").isalnum() and len(cleaned) <= 6:
        return cleaned.upper()
    raise ValueError(f"I could not resolve '{company_or_ticker}' to a stock ticker.")


def get_stock_quote(company_or_ticker: str) -> Dict[str, Any]:
    """Return a current structured quote from Yahoo Finance."""
    ticker = resolve_ticker(company_or_ticker)
    response = requests.get(
        _YAHOO_CHART_URL.format(symbol=ticker),
        params={"range": "1d", "interval": "1m"},
        headers={"User-Agent": "HybridRAG/1.0"},
        timeout=8,
    )
    response.raise_for_status()
    chart = response.json().get("chart", {})
    if chart.get("error"):
        raise ValueError(chart["error"].get("description", "The market API returned an error."))
    result = (chart.get("result") or [None])[0]
    if not result:
        raise ValueError(f"No quote was returned for {ticker}.")

    meta = result.get("meta", {})
    price = meta.get("regularMarketPrice")
    previous_close = meta.get("previousClose", meta.get("chartPreviousClose"))
    if price is None:
        raise ValueError(f"No current price was returned for {ticker}.")

    change = price - previous_close if previous_close is not None else None
    change_percent = (change / previous_close * 100) if previous_close else None
    timestamp = meta.get("regularMarketTime")
    observed_at = (
        datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        if timestamp else None
    )
    return {
        "ticker": ticker,
        "company": meta.get("longName") or meta.get("shortName") or ticker,
        "price": price,
        "currency": meta.get("currency", "USD"),
        "previous_close": previous_close,
        "change": change,
        "change_percent": change_percent,
        "market_status": meta.get("marketState", "UNKNOWN"),
        "timestamp": observed_at,
    }
