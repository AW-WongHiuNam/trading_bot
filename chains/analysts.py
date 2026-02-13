from prompts import MARKET_PROMPT, ANALYST_PROMPT
from tools.ollama_client import generate as ollama_generate
import json
from datetime import datetime

def market_analyst(raw_market_text: str, ticker: str = "TSLA") -> dict:
    prompt = MARKET_PROMPT.replace('{input}', raw_market_text)
    resp = ollama_generate(prompt)
    # For demo, return structured minimal JSON
    return {
        "role": "MARKET_ANALYST",
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": resp[:1000],
        "key_points": [],
        "confidence": 0.8,
        "recommendation_hint": "HOLD",
    }

def social_analyst(raw_social_text: str, ticker: str = "TSLA") -> dict:
    prompt = ANALYST_PROMPT.replace('{input}', raw_social_text)
    resp = ollama_generate(prompt)
    return {
        "role": "SOCIAL_ANALYST",
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": resp[:1000],
        "key_points": [],
        "confidence": 0.75,
        "recommendation_hint": "HOLD",
    }

def news_analyst(raw_news_text: str, ticker: str = "TSLA") -> dict:
    prompt = ANALYST_PROMPT.replace('{input}', raw_news_text)
    resp = ollama_generate(prompt)
    return {
        "role": "NEWS_ANALYST",
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": resp[:1000],
        "key_points": [],
        "confidence": 0.8,
        "recommendation_hint": "HOLD",
    }

def fundamentals_analyst(raw_fund_text: str, ticker: str = "TSLA") -> dict:
    prompt = ANALYST_PROMPT.replace('{input}', raw_fund_text)
    resp = ollama_generate(prompt)
    return {
        "role": "FUNDAMENTALS_ANALYST",
        "ticker": ticker,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": resp[:1000],
        "key_points": [],
        "confidence": 0.85,
        "recommendation_hint": "HOLD",
    }
