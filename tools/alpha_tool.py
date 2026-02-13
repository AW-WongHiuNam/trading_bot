from typing import Any, Dict, Optional
from alpha_fetch import fetch_av
from config import get_settings


def call_alpha(function: str, symbol: Optional[str] = None, tickers: Optional[str] = None, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    cfg = get_settings()
    api_key = cfg.alphavantage_api_key
    if not api_key:
        raise ValueError("Alpha Vantage API key not configured (ALPHAVANTAGE_API_KEY)")

    call_params = dict(params or {})
    if symbol:
        call_params["symbol"] = symbol
    if tickers:
        call_params["tickers"] = tickers

    # Use fetch_av helper to call Alpha Vantage
    return fetch_av(function, api_key=api_key, params=call_params)
