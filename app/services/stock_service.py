import csv
import os
from datetime import datetime, timedelta

from scripts.alpha_fetch import fetch_av
from app.config import settings


def _csv_path(ticker: str, start: str, end: str) -> str:
    os.makedirs("./app/data/stocks", exist_ok=True)
    return f"./app/data/stocks/{ticker}_{start}_{end}.csv"


def _generate_fake(ticker: str, start: str, end: str) -> list[dict]:
    try:
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
    except ValueError as exc:
        raise ValueError("start/end must be ISO dates like YYYY-MM-DD") from exc
    points = []
    cur = s
    price = 100.0
    while cur <= e:
        price += 0.5
        points.append(
            {
                "date": cur.strftime("%Y-%m-%d"),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price + 0.2,
                "volume": 1000000,
            }
        )
        cur += timedelta(days=1)
    return points


def _save_csv(path: str, points: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        writer.writeheader()
        writer.writerows(points)


def _load_csv(path: str) -> list[dict]:
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return [
            {
                "date": r["date"],
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": int(r["volume"]),
            }
            for r in reader
        ]


def _fetch_alpha_vantage(ticker: str) -> dict:
    payload = fetch_av("TIME_SERIES_DAILY", api_key=None, params={"symbol": ticker})
    if isinstance(payload, dict):
        for key in ("Error Message", "Note", "Information"):
            msg = payload.get(key)
            if isinstance(msg, str) and msg.strip():
                raise RuntimeError(msg.strip())
    return payload


def get_stock_data(ticker: str, start: str, end: str) -> list[dict]:
    path = _csv_path(ticker, start, end)
    if os.path.exists(path):
        return _load_csv(path)

    if settings.stock_fake_data:
        points = _generate_fake(ticker, start, end)
        _save_csv(path, points)
        return points

    data = _fetch_alpha_vantage(ticker)
    ts = data.get("Time Series (Daily)", {})
    points = []
    for date, row in ts.items():
        if start <= date <= end:
            points.append(
                {
                    "date": date,
                    "open": float(row["1. open"]),
                    "high": float(row["2. high"]),
                    "low": float(row["3. low"]),
                    "close": float(row["4. close"]),
                    "volume": int(float(row["5. volume"])),
                }
            )
    points.sort(key=lambda x: x["date"])
    _save_csv(path, points)
    return points
