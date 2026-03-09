from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.data import load_price_csv, parse_iso_date
from backtesting.engine import simulate_single_trade, summarize_results
from backtesting.sanity import run_sanity_checks
from backtesting.signal import derive_signal


def _discussion_text(result: dict) -> str:
    researchers = result.get("researchers") or {}
    text = researchers.get("discussion")
    if isinstance(text, str):
        return text
    return ""


def _estimate_notional(cash: float, size: float, entry_price: float | None) -> float:
    if size <= 0:
        return 0.0
    if size <= 1.0:
        return max(0.0, cash * size)
    if entry_price and entry_price > 0:
        return max(0.0, min(cash, size * entry_price))
    return 0.0


def run_period(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    price_csv: Path,
    template_path: str,
    model: str | None,
    initial_cash: float,
    initial_shares: float,
    hold_days: int,
    fee_bps: float,
    slippage_bps: float,
    allow_price_fetch: bool,
    allow_news_fetch: bool,
    enforce_runtime_cutoff: bool,
    decision_policy: str,
    out_dir: Path,
) -> dict:
    from chains.langchain_chains import run_langchain_flow

    bars = load_price_csv(price_csv)
    start = parse_iso_date(start_date)
    end = parse_iso_date(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")

    trading_days = [bar.date for bar in bars if start <= bar.date <= end]
    if not trading_days:
        raise ValueError("No trading days found in selected period")

    out_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = out_dir / "daily_outputs"
    daily_dir.mkdir(parents=True, exist_ok=True)

    cash = float(initial_cash)
    shares = float(initial_shares)
    trade_results = []
    rows: list[dict] = []
    total_days = len(trading_days)

    for idx, day in enumerate(trading_days, start=1):
        day_iso = day.isoformat()
        print(f"[day_by_day] {idx}/{total_days} ({(idx / total_days) * 100:.1f}%) running {ticker} @ {day_iso}", flush=True)

        flow = run_langchain_flow(
            template_path=template_path,
            model=model,
            ticker=ticker,
            target_date=day_iso,
            account_cash=cash,
            account_shares=shares,
            allow_price_fetch=allow_price_fetch,
        )

        daily_file = daily_dir / f"{ticker}_{day_iso}.json"
        daily_file.write_text(json.dumps(flow, ensure_ascii=False, indent=2), encoding="utf-8")

        sanity = run_sanity_checks(
            result=flow,
            cutoff_date=day,
            forbid_price_fetch=not allow_price_fetch,
            allow_news_fetch=allow_news_fetch,
            enforce_runtime_cutoff=enforce_runtime_cutoff,
        )
        signal = derive_signal(flow, ticker_fallback=ticker, decision_policy=decision_policy)

        if sanity.passed:
            trade = simulate_single_trade(
                bars,
                target_date=day,
                signal=signal,
                hold_days=hold_days,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )
        else:
            trade = simulate_single_trade(
                bars,
                target_date=day,
                signal=type(signal)(
                    ticker=signal.ticker,
                    side="NO_TRADE",
                    size=0.0,
                    confidence=signal.confidence,
                    rationale=signal.rationale,
                    blocked_reason="sanity_check_failed",
                ),
                hold_days=hold_days,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )

        trade_results.append(trade)

        notional = _estimate_notional(cash=cash, size=signal.size, entry_price=trade.entry_price)
        pnl = notional * trade.net_return if trade.executed else 0.0
        cash = cash + pnl

        row = {
            "date": day_iso,
            "cash_before": cash - pnl,
            "cash_after": cash,
            "shares_context": shares,
            "sanity_passed": sanity.passed,
            "sanity_error_count": len(sanity.errors),
            "side": signal.side,
            "size": signal.size,
            "signal_confidence": signal.confidence,
            "trade_executed": trade.executed,
            "trade_reason": trade.reason,
            "entry_date": trade.entry_date,
            "exit_date": trade.exit_date,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "net_return": trade.net_return,
            "pnl": pnl,
            "daily_output": str(daily_file).replace("\\", "/"),
            "discussion": _discussion_text(flow),
            "bull_rounds": len(((flow.get("researchers") or {}).get("bull_rounds") or [])),
            "bear_rounds": len(((flow.get("researchers") or {}).get("bear_rounds") or [])),
        }
        rows.append(row)
        print(
            f"[day_by_day] done {day_iso} | sanity={sanity.passed} | side={signal.side} | executed={trade.executed} | cash={cash:.2f}",
            flush=True,
        )

    summary = summarize_results(trade_results)
    summary["final_cash"] = cash
    summary["trading_days"] = len(trading_days)

    csv_path = out_dir / "daily_summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "meta": {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "price_csv": str(price_csv).replace("\\", "/"),
            "template_path": template_path,
            "model": model,
            "initial_cash": initial_cash,
            "initial_shares": initial_shares,
            "hold_days": hold_days,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "allow_price_fetch": allow_price_fetch,
            "allow_news_fetch": allow_news_fetch,
            "enforce_runtime_cutoff": enforce_runtime_cutoff,
            "decision_policy": decision_policy,
            "trading_days": len(trading_days),
        },
        "summary": summary,
        "artifacts": {
            "daily_outputs_dir": str(daily_dir).replace("\\", "/"),
            "daily_summary_csv": str(csv_path).replace("\\", "/"),
        },
        "days": rows,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run day-by-day backtesting over a date range and persist full daily agent discussions")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--price-csv", required=True)
    parser.add_argument("--template-path", default="OUTPUT_TEMPLATE.TXT")
    parser.add_argument("--model", default=None)
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--initial-shares", type=float, default=0.0)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--allow-price-fetch", action="store_true")
    parser.add_argument("--disallow-news-fetch", action="store_true")
    parser.add_argument("--enforce-runtime-cutoff", action="store_true")
    parser.add_argument("--decision-policy", choices=["strict", "backtest", "auto"], default="strict")
    parser.add_argument("--out-dir", help="Output folder. Default: outputs/backtesting/day_by_day_<ticker>_<start>_<end>_<run_id>")
    parser.add_argument("--report-out", help="Optional path to report JSON")
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs/backtesting") / f"day_by_day_{args.ticker}_{args.start_date}_to_{args.end_date}_{run_id}"

    report = run_period(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        price_csv=Path(args.price_csv),
        template_path=args.template_path,
        model=args.model,
        initial_cash=args.initial_cash,
        initial_shares=args.initial_shares,
        hold_days=args.hold_days,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        allow_price_fetch=args.allow_price_fetch,
        allow_news_fetch=not args.disallow_news_fetch,
        enforce_runtime_cutoff=args.enforce_runtime_cutoff,
        decision_policy=args.decision_policy,
        out_dir=out_dir,
    )

    report_out = Path(args.report_out) if args.report_out else out_dir / "report.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"\nReport written: {str(report_out).replace('\\', '/')}")
    print(f"Daily discussion logs: {str((out_dir / 'daily_outputs')).replace('\\', '/')}")


if __name__ == "__main__":
    main()
