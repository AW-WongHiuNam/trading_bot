from datetime import datetime
from typing import Any, Dict, Optional
from scripts.alpha_fetch import fetch_av


def _normalize_iso_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date().isoformat()
    except Exception:
        return None


def _resolve_daily_bar(series: dict[str, dict], as_of_date: str) -> tuple[str, dict] | None:
    eligible = [key for key in series.keys() if key <= as_of_date]
    if not eligible:
        return None
    pick = max(eligible)
    return pick, series[pick]


def _global_quote_from_daily(symbol: str, day: str, bar: dict, previous_close: float | None) -> Dict[str, Any]:
    open_px = float(bar.get("1. open", 0.0))
    high_px = float(bar.get("2. high", 0.0))
    low_px = float(bar.get("3. low", 0.0))
    close_px = float(bar.get("4. close", 0.0))
    volume = str(bar.get("5. volume", "0"))
    prev = float(previous_close) if previous_close is not None else close_px
    change = close_px - prev
    pct = (change / prev * 100.0) if prev else 0.0

    return {
        "Global Quote": {
            "01. symbol": symbol,
            "02. open": f"{open_px:.4f}",
            "03. high": f"{high_px:.4f}",
            "04. low": f"{low_px:.4f}",
            "05. price": f"{close_px:.4f}",
            "06. volume": volume,
            "07. latest trading day": day,
            "08. previous close": f"{prev:.4f}",
            "09. change": f"{change:.4f}",
            "10. change percent": f"{pct:.4f}%",
        }
    }


def _fetch_price_as_of(function: str, symbol: str, as_of_date: str) -> Dict[str, Any]:
    base = fetch_av("TIME_SERIES_DAILY", api_key=None, params={"symbol": symbol, "outputsize": "compact"})
    series = base.get("Time Series (Daily)") if isinstance(base, dict) else None
    if not isinstance(series, dict):
        return base

    resolved = _resolve_daily_bar(series, as_of_date)
    if not resolved:
        return {
            "Error Message": f"No daily bar available for symbol={symbol} as_of_date={as_of_date}",
            "requested_as_of_date": as_of_date,
        }

    day, bar = resolved
    prior_days = sorted([d for d in series.keys() if d < day])
    prev_close = None
    if prior_days:
        prev_close = float((series[prior_days[-1]] or {}).get("4. close", 0.0))

    if function == "GLOBAL_QUOTE":
        return _global_quote_from_daily(symbol, day, bar, prev_close)

    return {
        "Meta Data": {
            "1. Information": f"{function} reconstructed from TIME_SERIES_DAILY as-of date",
            "2. Symbol": symbol,
            "3. Requested As-Of Date": as_of_date,
            "4. Resolved Date": day,
        },
        "Time Series (Daily)": {day: bar},
    }


def call_alpha(function: str, symbol: Optional[str] = None, tickers: Optional[str] = None, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    call_params = dict(params or {})
    if symbol:
        call_params["symbol"] = symbol
    if tickers:
        call_params["tickers"] = tickers

    as_of_date = _normalize_iso_date(call_params.get("as_of_date") if isinstance(call_params, dict) else None)
    if as_of_date and symbol and function in {"GLOBAL_QUOTE", "TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED"}:
        return _fetch_price_as_of(function=function, symbol=symbol, as_of_date=as_of_date)

    # Use fetch_av helper to call Alpha Vantage
    return fetch_av(function, api_key=None, params=call_params)
