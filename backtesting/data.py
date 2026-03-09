from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


@dataclass(frozen=True)
class PriceBar:
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int


def parse_us_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%m/%d/%Y").date()


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _num(raw: str) -> float:
    cleaned = raw.strip().replace("$", "").replace(",", "")
    return float(cleaned)


def load_price_csv(csv_path: str | Path) -> list[PriceBar]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Price CSV not found: {path}")

    bars: list[PriceBar] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Date", "Close/Last", "Volume", "Open", "High", "Low"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV schema mismatch. Need columns: {sorted(required)}")

        for row in reader:
            bars.append(
                PriceBar(
                    date=parse_us_date(row["Date"]),
                    open=_num(row["Open"]),
                    high=_num(row["High"]),
                    low=_num(row["Low"]),
                    close=_num(row["Close/Last"]),
                    volume=int(_num(row["Volume"])),
                )
            )

    bars.sort(key=lambda item: item.date)
    return bars


def snapshot_bars(bars: list[PriceBar], as_of_date: date, lookback_days: int) -> list[PriceBar]:
    eligible = [bar for bar in bars if bar.date <= as_of_date]
    if not eligible:
        return []
    if lookback_days <= 0:
        return eligible
    return eligible[-lookback_days:]


def write_snapshot_csv(snapshot: list[PriceBar], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for bar in snapshot:
            writer.writerow([bar.date.isoformat(), bar.open, bar.high, bar.low, bar.close, bar.volume])
    return path


def load_agent_payload(json_path: str | Path) -> dict:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent output JSON not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict) and isinstance(raw.get("result"), dict):
        return {
            "ticker": raw.get("ticker"),
            "target_date": raw.get("target_date"),
            "result": raw["result"],
            "raw": raw,
        }

    if isinstance(raw, dict):
        return {
            "ticker": raw.get("trader_proposal", {}).get("ticker") or raw.get("ticker"),
            "target_date": raw.get("target_date"),
            "result": raw,
            "raw": raw,
        }

    raise ValueError("Unsupported JSON payload format")
