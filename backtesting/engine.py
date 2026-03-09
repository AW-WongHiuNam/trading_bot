from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from backtesting.data import PriceBar
from backtesting.signal import TradeSignal


@dataclass
class TradeResult:
    executed: bool
    reason: str
    entry_date: str | None = None
    exit_date: str | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    gross_return: float = 0.0
    net_return: float = 0.0


def _find_next_index(bars: list[PriceBar], trade_date: date) -> int | None:
    for idx, bar in enumerate(bars):
        if bar.date > trade_date:
            return idx
    return None


def simulate_single_trade(
    bars: list[PriceBar],
    *,
    target_date: date,
    signal: TradeSignal,
    hold_days: int = 5,
    fee_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> TradeResult:
    if signal.side == "NO_TRADE":
        return TradeResult(executed=False, reason=signal.blocked_reason or "no_trade_signal")

    entry_idx = _find_next_index(bars, target_date)
    if entry_idx is None:
        return TradeResult(executed=False, reason="no_next_bar_for_entry")

    exit_idx = min(entry_idx + max(1, hold_days), len(bars) - 1)
    entry = bars[entry_idx]
    exit_bar = bars[exit_idx]

    gross = (exit_bar.close - entry.open) / entry.open
    if signal.side == "SELL":
        gross = -gross

    total_cost = (fee_bps + slippage_bps) / 10000.0
    net = gross - total_cost

    return TradeResult(
        executed=True,
        reason="executed",
        entry_date=entry.date.isoformat(),
        exit_date=exit_bar.date.isoformat(),
        entry_price=entry.open,
        exit_price=exit_bar.close,
        gross_return=gross,
        net_return=net,
    )


def summarize_results(results: list[TradeResult]) -> dict:
    executed = [item for item in results if item.executed]
    if not executed:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_net_return": 0.0,
            "total_net_return": 0.0,
            "equity_end": 1.0,
        }

    wins = sum(1 for item in executed if item.net_return > 0)
    equity = 1.0
    for item in executed:
        equity *= (1.0 + item.net_return)

    total_net = sum(item.net_return for item in executed)
    avg_net = total_net / len(executed)

    return {
        "trades": len(executed),
        "win_rate": wins / len(executed),
        "avg_net_return": avg_net,
        "total_net_return": total_net,
        "equity_end": equity,
    }
