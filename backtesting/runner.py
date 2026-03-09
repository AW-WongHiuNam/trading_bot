from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.data import (
    load_agent_payload,
    load_price_csv,
    parse_iso_date,
    snapshot_bars,
    write_snapshot_csv,
)
from backtesting.engine import simulate_single_trade, summarize_results
from backtesting.sanity import run_sanity_checks
from backtesting.signal import derive_signal


def run_one(
    *,
    result_path: Path,
    csv_path: Path,
    lookback_days: int,
    hold_days: int,
    fee_bps: float,
    slippage_bps: float,
    forbid_price_fetch: bool,
    allow_news_fetch: bool,
    enforce_runtime_cutoff: bool,
    snapshot_out: Path | None,
    target_date_override: str | None,
    decision_policy: str,
) -> dict:
    payload = load_agent_payload(result_path)
    result = payload["result"]
    bars = load_price_csv(csv_path)

    target_date_raw = target_date_override or payload.get("target_date")
    if not target_date_raw:
        raise ValueError("target_date is required in payload or via wrapper JSON")

    target_date = parse_iso_date(target_date_raw)

    snapshot = snapshot_bars(bars, as_of_date=target_date, lookback_days=lookback_days)
    if not snapshot:
        raise ValueError(f"No price bars available up to target_date={target_date}")

    if snapshot_out:
        write_snapshot_csv(snapshot, snapshot_out)

    sanity = run_sanity_checks(
        result=result,
        cutoff_date=target_date,
        forbid_price_fetch=forbid_price_fetch,
        allow_news_fetch=allow_news_fetch,
        enforce_runtime_cutoff=enforce_runtime_cutoff,
    )

    signal = derive_signal(result, ticker_fallback=payload.get("ticker") or "UNKNOWN", decision_policy=decision_policy)

    if sanity.passed:
        trade = simulate_single_trade(
            snapshot,
            target_date=target_date,
            signal=signal,
            hold_days=hold_days,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
    else:
        trade = simulate_single_trade(
            snapshot,
            target_date=target_date,
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

    summary = summarize_results([trade])

    return {
        "meta": {
            "result_json": str(result_path).replace("\\", "/"),
            "price_csv": str(csv_path).replace("\\", "/"),
            "target_date": target_date.isoformat(),
            "lookback_days": lookback_days,
            "hold_days": hold_days,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "forbid_price_fetch": forbid_price_fetch,
            "allow_news_fetch": allow_news_fetch,
            "enforce_runtime_cutoff": enforce_runtime_cutoff,
            "decision_policy": decision_policy,
        },
        "sanity": {
            "passed": sanity.passed,
            "errors": sanity.errors,
            "warnings": sanity.warnings,
        },
        "signal": {
            "ticker": signal.ticker,
            "side": signal.side,
            "size": signal.size,
            "confidence": signal.confidence,
            "rationale": signal.rationale,
            "blocked_reason": signal.blocked_reason,
        },
        "trade": trade.__dict__,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest agent JSON against historical price CSV")
    parser.add_argument("--result-json", required=True, help="Path to agent output JSON")
    parser.add_argument("--price-csv", required=True, help="Path to historical price CSV")
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--allow-price-fetch", action="store_true", help="Disable strict check blocking alpha price fetch calls")
    parser.add_argument("--disallow-news-fetch", action="store_true", help="Disallow NEWS_SENTIMENT calls in sanity checks")
    parser.add_argument("--enforce-runtime-cutoff", action="store_true", help="Fail when runtime timestamps are after target_date")
    parser.add_argument("--target-date", help="Override target date (YYYY-MM-DD) for raw archived JSON files")
    parser.add_argument("--decision-policy", choices=["strict", "backtest", "auto"], default="strict", help="How to map manager/manual outcomes into trade signals")
    parser.add_argument("--snapshot-out", help="Optional CSV path for snapshot data exposed to agent")
    parser.add_argument("--report-out", help="Optional path for report JSON")
    args = parser.parse_args()

    result_path = Path(args.result_json)
    price_csv = Path(args.price_csv)
    snapshot_out = Path(args.snapshot_out) if args.snapshot_out else None

    report = run_one(
        result_path=result_path,
        csv_path=price_csv,
        lookback_days=args.lookback_days,
        hold_days=args.hold_days,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        forbid_price_fetch=not args.allow_price_fetch,
        allow_news_fetch=not args.disallow_news_fetch,
        enforce_runtime_cutoff=args.enforce_runtime_cutoff,
        snapshot_out=snapshot_out,
        target_date_override=args.target_date,
        decision_policy=args.decision_policy,
    )

    if args.report_out:
        report_out = Path(args.report_out)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("outputs/backtesting") / f"single_run_{run_id}"
        report_out = run_dir / "report.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nReport written: {str(report_out).replace('\\\\', '/')}")


if __name__ == "__main__":
    main()
