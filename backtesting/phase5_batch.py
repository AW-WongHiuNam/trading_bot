from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.runner import run_one


def run_batch(
    *,
    results_dir: Path,
    price_csv: Path,
    target_date: str,
    lookback_days: int,
    hold_days: int,
    fee_bps: float,
    slippage_bps: float,
    forbid_price_fetch: bool,
    allow_news_fetch: bool,
    decision_policy: str,
) -> dict:
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in {results_dir}")

    reports = []
    total_files = len(files)
    for idx, file_path in enumerate(files, start=1):
        print(
            f"[phase5_batch] {idx}/{total_files} ({(idx / total_files) * 100:.1f}%) processing {file_path.name}",
            flush=True,
        )
        try:
            report = run_one(
                result_path=file_path,
                csv_path=price_csv,
                lookback_days=lookback_days,
                hold_days=hold_days,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                forbid_price_fetch=forbid_price_fetch,
                allow_news_fetch=allow_news_fetch,
                enforce_runtime_cutoff=False,
                snapshot_out=None,
                target_date_override=target_date,
                decision_policy=decision_policy,
            )
            report["_file"] = str(file_path).replace("\\", "/")
            reports.append(report)
            executed = bool(((report.get("trade") or {}).get("executed") is True))
            sanity_passed = bool(((report.get("sanity") or {}).get("passed") is True))
            print(
                f"[phase5_batch] done {file_path.name} | sanity={sanity_passed} | executed={executed}",
                flush=True,
            )
        except Exception as exc:
            reports.append(
                {
                    "_file": str(file_path).replace("\\", "/"),
                    "sanity": {"passed": False, "errors": [f"batch_exception: {exc}"], "warnings": []},
                    "summary": {"trades": 0, "win_rate": 0.0, "avg_net_return": 0.0, "total_net_return": 0.0, "equity_end": 1.0},
                    "trade": {"executed": False, "reason": "batch_exception"},
                }
            )
            print(f"[phase5_batch] error {file_path.name} | {exc}", flush=True)

    total = len(reports)
    sane = sum(1 for item in reports if (item.get("sanity") or {}).get("passed") is True)
    executed = [item for item in reports if ((item.get("trade") or {}).get("executed") is True)]
    wins = sum(1 for item in executed if float((item.get("trade") or {}).get("net_return") or 0.0) > 0.0)

    total_net_return = sum(float((item.get("trade") or {}).get("net_return") or 0.0) for item in reports)
    equity = 1.0
    for item in reports:
        net = float((item.get("trade") or {}).get("net_return") or 0.0)
        equity *= (1.0 + net)

    error_buckets: dict[str, int] = {}
    for item in reports:
        for err in (item.get("sanity") or {}).get("errors") or []:
            error_buckets[err] = error_buckets.get(err, 0) + 1

    return {
        "meta": {
            "results_dir": str(results_dir).replace("\\", "/"),
            "price_csv": str(price_csv).replace("\\", "/"),
            "target_date": target_date,
            "lookback_days": lookback_days,
            "hold_days": hold_days,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            "forbid_price_fetch": forbid_price_fetch,
            "allow_news_fetch": allow_news_fetch,
            "decision_policy": decision_policy,
            "file_count": total,
        },
        "aggregate": {
            "total_files": total,
            "sanity_passed": sane,
            "sanity_failed": total - sane,
            "executed_trades": len(executed),
            "win_rate_executed": (wins / len(executed)) if executed else 0.0,
            "total_net_return": total_net_return,
            "equity_end": equity,
            "top_sanity_errors": sorted(error_buckets.items(), key=lambda item: item[1], reverse=True)[:20],
        },
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-5 batch backtest for multiple agent JSON outputs")
    parser.add_argument("--results-dir", required=True, help="Folder containing flow_output*.json files")
    parser.add_argument("--price-csv", required=True)
    parser.add_argument("--target-date", required=True, help="As-of date applied to all files in batch")
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--allow-price-fetch", action="store_true")
    parser.add_argument("--disallow-news-fetch", action="store_true")
    parser.add_argument("--decision-policy", choices=["strict", "backtest", "auto"], default="strict")
    parser.add_argument("--report-out", help="Optional output path for aggregate JSON")
    args = parser.parse_args()

    report = run_batch(
        results_dir=Path(args.results_dir),
        price_csv=Path(args.price_csv),
        target_date=args.target_date,
        lookback_days=args.lookback_days,
        hold_days=args.hold_days,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        forbid_price_fetch=not args.allow_price_fetch,
        allow_news_fetch=not args.disallow_news_fetch,
        decision_policy=args.decision_policy,
    )

    if args.report_out:
        report_out = Path(args.report_out)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("outputs/backtesting") / f"phase5_run_{run_id}"
        report_out = run_dir / "report.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(f"\nReport written: {str(report_out).replace('\\', '/')}")


if __name__ == "__main__":
    main()
