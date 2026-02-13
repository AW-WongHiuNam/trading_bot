#!/usr/bin/env python3
"""Simple Alpha Vantage NEWS_SENTIMENT helper.

Usage example:
  python get_news.py --apikey BSE23K9THV7KR1X7 --tickers AAPL --limit 5
"""
import time
from typing import List, Dict, Optional

import requests

from config import get_settings

API_URL = "https://www.alphavantage.co/query"


def get_news_sentiment(api_key: str, tickers: Optional[List[str]] = None, topics: Optional[List[str]] = None, retries: int = 3, backoff: float = 1.0) -> Dict:
    """Fetch NEWS_SENTIMENT from Alpha Vantage."""
    params = {"function": "NEWS_SENTIMENT", "apikey": api_key}
    if tickers:
        params["tickers"] = ",".join(tickers)
    if topics:
        params["topics"] = ",".join(topics)

    attempt = 0
    while True:
        try:
            resp = requests.get(API_URL, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))


def extract_articles(response: Dict) -> List[Dict]:
    """Extract the article feed list from the Alpha Vantage response."""
    return response.get("feed", [])


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fetch news sentiment from Alpha Vantage")
    parser.add_argument("--apikey", required=False, help="Alpha Vantage API key (overrides .env)", default=None)
    parser.add_argument("--tickers", required=False, help="Comma-separated tickers (default AAPL)", default="AAPL")
    parser.add_argument("--topics", required=False, help="Comma-separated topics", default=None)
    parser.add_argument("--limit", required=False, type=int, help="Max articles to print", default=10)
    args = parser.parse_args()

    cfg = get_settings()
    api_key = args.apikey or cfg.alphavantage_api_key
    if not api_key:
        raise ValueError("Alpha Vantage API key is required. Set ALPHAVANTAGE_API_KEY in .env or pass --apikey.")

    tickers = args.tickers.split(",") if args.tickers else None
    topics = args.topics.split(",") if args.topics else None

    data = get_news_sentiment(api_key=api_key, tickers=tickers, topics=topics)
    articles = extract_articles(data)[: args.limit]
    print(json.dumps(articles, indent=2))
