from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import time
import csv

from sqlalchemy.orm import Session
import requests

from app.config import settings
from app.db.models import BacktestingJob, BacktestingState, TradeHistory
from app.db.session import SessionLocal
try:
    from tools.rag_tool import _get_store as _get_rag_store
except Exception:
    _get_rag_store = None


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _to_iso(value) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _add_state(db: Session, job_id: int, stage: str, payload: dict) -> None:
    row = BacktestingState(job_id=job_id, stage=stage, payload_json=json.dumps(payload, ensure_ascii=False))
    db.add(row)
    db.commit()


def _set_status(db: Session, job_id: int, status: str, progress: int | None = None, error: str | None = None) -> None:
    row = db.query(BacktestingJob).filter(BacktestingJob.id == job_id).first()
    if not row:
        return
    row.status = status
    if progress is not None:
        row.progress = progress
    row.error = error
    db.commit()


def create_backtesting_job(db: Session, ticker: str, start_date: str, end_date: str, decision_policy: str) -> int:
    row = BacktestingJob(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        decision_policy=decision_policy,
        status="queued",
        progress=0,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    _add_state(db, row.id, "queued", {"ticker": ticker, "start_date": start_date, "end_date": end_date, "at": _now_iso()})
    return row.id


def list_backtesting_jobs(db: Session, ticker: str | None = None) -> list[dict]:
    query = db.query(BacktestingJob)
    if ticker:
        query = query.filter(BacktestingJob.ticker == ticker)
    rows = query.order_by(BacktestingJob.id.desc()).all()
    return [
        {
            "jobId": row.id,
            "ticker": row.ticker,
            "start_date": row.start_date,
            "end_date": row.end_date,
            "decision_policy": row.decision_policy,
            "status": row.status,
            "progress": row.progress,
            "created_at": _to_iso(row.created_at),
            "updated_at": _to_iso(row.updated_at),
        }
        for row in rows
    ]


def get_backtesting_status(db: Session, job_id: int) -> dict | None:
    row = db.query(BacktestingJob).filter(BacktestingJob.id == job_id).first()
    if not row:
        return None
    latest = (
        db.query(BacktestingState)
        .filter(BacktestingState.job_id == job_id)
        .order_by(BacktestingState.id.desc())
        .first()
    )
    latest_payload = json.loads(latest.payload_json) if latest else None
    trades = (
        db.query(TradeHistory)
        .filter(TradeHistory.source_job_id == job_id, TradeHistory.source_type == "backtesting")
        .order_by(TradeHistory.trade_date)
        .all()
    )
    return {
        "jobId": row.id,
        "status": row.status,
        "progress": row.progress,
        "error": row.error,
        "latest_state": latest_payload,
        "trade_history": [_trade_row_to_dict(t) for t in trades],
    }


def get_backtesting_result(db: Session, job_id: int) -> dict | None:
    row = db.query(BacktestingJob).filter(BacktestingJob.id == job_id).first()
    if not row:
        return None
    summary = json.loads(row.summary_json) if row.summary_json else None
    artifacts = None
    initial_cash = None
    if row.report_path:
        report_file = Path(row.report_path)
        if report_file.exists():
            try:
                payload = json.loads(report_file.read_text(encoding="utf-8"))
                artifacts = payload.get("artifacts")
                if summary is None:
                    summary = payload.get("summary")
                meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
                if isinstance(meta.get("initial_cash"), (int, float)):
                    initial_cash = float(meta["initial_cash"])
                elif isinstance(meta.get("initial_cash"), str):
                    try:
                        initial_cash = float(meta["initial_cash"].strip())
                    except Exception:
                        initial_cash = None
            except Exception:
                artifacts = None

    # Ensure consumers can use explicit starting money instead of inferring
    # from potentially non-cash-normalized metrics like equity_end.
    if initial_cash is not None:
        if not isinstance(summary, dict):
            summary = {}
        summary.setdefault("initial_cash", initial_cash)
        summary.setdefault("starting_money", initial_cash)

    trades = (
        db.query(TradeHistory)
        .filter(TradeHistory.source_job_id == job_id, TradeHistory.source_type == "backtesting")
        .order_by(TradeHistory.trade_date)
        .all()
    )
    return {
        "jobId": row.id,
        "status": row.status,
        "summary": summary,
        "report_path": row.report_path,
        "artifacts": artifacts,
        "trade_history": [_trade_row_to_dict(t) for t in trades],
    }


def _trade_row_to_dict(row: TradeHistory) -> dict:
    return {
        "id": row.id,
        "ticker": row.ticker,
        "trade_date": row.trade_date,
        "side": row.side,
        "size": row.size,
        "entry_date": row.entry_date,
        "exit_date": row.exit_date,
        "entry_price": row.entry_price,
        "exit_price": row.exit_price,
        "net_return": row.net_return,
        "pnl": row.pnl,
        "source_type": row.source_type,
        "source_job_id": row.source_job_id,
    }


def _normalize_price_row(columns: list, row: list) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for index, col in enumerate(columns):
        key = str(col).strip().lower()
        normalized[key] = row[index] if index < len(row) else None
    return normalized


def _pick_first(row: dict[str, object], *keys: str):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _format_backend_date_for_csv(value: object) -> str:
    if value is None:
        raise ValueError("Missing date in backend price row")
    text = str(value).strip()
    if not text:
        raise ValueError("Empty date in backend price row")
    if "T" in text:
        text = text.split("T", 1)[0]
    parsed = datetime.strptime(text, "%Y-%m-%d")
    return parsed.strftime("%m/%d/%Y")


def _format_price_for_csv(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _format_volume_for_csv(value: object) -> str:
    if value is None:
        return "0"
    return str(int(float(value)))


def _fetch_backend_price_csv(ticker: str, start_date: str, end_date: str) -> Path:
    base_url = settings.easytradingbackend_base_url.rstrip("/")
    url = f"{base_url}/api/price"
    payload = {
        "ticker": ticker.upper(),
        "start": f"{start_date}T00:00:00+00:00",
        "end": f"{end_date}T23:59:59+00:00",
    }

    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    columns = data.get("columns")
    rows = data.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise ValueError("Unexpected /api/price response shape")
    if not rows:
        raise ValueError(f"No price rows returned by EasyTradingBackend for {ticker} {start_date}..{end_date}")

    cache_dir = Path("app/data/backtesting_price_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"HistoricalData_{ticker.upper()}_{start_date}_{end_date}.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date", "Close/Last", "Volume", "Open", "High", "Low"])
        for raw_row in rows:
            if not isinstance(raw_row, list):
                raise ValueError("Unexpected row type in /api/price response")
            row = _normalize_price_row(columns, raw_row)
            date_value = _pick_first(row, "date")
            open_value = _pick_first(row, "open")
            high_value = _pick_first(row, "high")
            low_value = _pick_first(row, "low")
            close_value = _pick_first(row, "close", "close/last", "adj close", "adj_close")
            volume_value = _pick_first(row, "volume")

            if None in (date_value, open_value, high_value, low_value, close_value, volume_value):
                raise ValueError("Missing OHLCV field in /api/price response row")

            writer.writerow([
                _format_backend_date_for_csv(date_value),
                _format_price_for_csv(close_value),
                _format_volume_for_csv(volume_value),
                _format_price_for_csv(open_value),
                _format_price_for_csv(high_value),
                _format_price_for_csv(low_value),
            ])

    return csv_path


def _resolve_price_csv_path(ticker: str, start_date: str, end_date: str, price_csv: str | None) -> tuple[Path, str, str | None]:
    if price_csv:
        return Path(price_csv), "explicit_csv", None

    if settings.backtesting_use_backend_price_data:
        try:
            csv_path = _fetch_backend_price_csv(ticker, start_date, end_date)
            return csv_path, "easytradingbackend_api", None
        except Exception as exc:
            return (
                Path("stock_price") / f"HistoricalData_{ticker.upper()}.csv",
                "local_csv_fallback",
                f"EasyTradingBackend price fetch failed: {exc}",
            )

    return Path("stock_price") / f"HistoricalData_{ticker.upper()}.csv", "local_csv_fallback", None


def _store_single_trade_row(db: Session, job_id: int, ticker: str, item: dict) -> None:
    """Insert one executed trade row if it doesn't exist yet for this job+date."""
    existing = (
        db.query(TradeHistory)
        .filter(
            TradeHistory.source_job_id == job_id,
            TradeHistory.source_type == "backtesting",
            TradeHistory.trade_date == str(item.get("date") or ""),
        )
        .first()
    )
    if existing:
        return
    row = TradeHistory(
        ticker=ticker.upper(),
        trade_date=str(item.get("date") or ""),
        side=str(item.get("side") or "NO_TRADE"),
        size=str(item.get("size") or "0"),
        entry_date=str(item["entry_date"]) if item.get("entry_date") else None,
        exit_date=str(item["exit_date"]) if item.get("exit_date") else None,
        entry_price=str(item["entry_price"]) if item.get("entry_price") is not None else None,
        exit_price=str(item["exit_price"]) if item.get("exit_price") is not None else None,
        net_return=str(item["net_return"]) if item.get("net_return") is not None else None,
        pnl=str(item["pnl"]) if item.get("pnl") is not None else None,
        source_type="backtesting",
        source_job_id=job_id,
    )
    db.add(row)
    db.commit()


def _store_trade_history_for_job(db: Session, job_id: int, report: dict) -> None:
    db.query(TradeHistory).filter(TradeHistory.source_job_id == job_id, TradeHistory.source_type == "backtesting").delete()
    db.commit()

    rows = report.get("days") or []
    for item in rows:
        if not item.get("trade_executed"):
            continue
        history = TradeHistory(
            ticker=str(item.get("ticker") or report.get("meta", {}).get("ticker") or "").upper(),
            trade_date=str(item.get("date") or ""),
            side=str(item.get("side") or "NO_TRADE"),
            size=str(item.get("size") or "0"),
            entry_date=str(item.get("entry_date")) if item.get("entry_date") else None,
            exit_date=str(item.get("exit_date")) if item.get("exit_date") else None,
            entry_price=str(item.get("entry_price")) if item.get("entry_price") is not None else None,
            exit_price=str(item.get("exit_price")) if item.get("exit_price") is not None else None,
            net_return=str(item.get("net_return")) if item.get("net_return") is not None else None,
            pnl=str(item.get("pnl")) if item.get("pnl") is not None else None,
            source_type="backtesting",
            source_job_id=job_id,
        )
        db.add(history)
    db.commit()


def _store_trades_to_vector_store(job_id: int, report: dict) -> None:
    """Best-effort: store executed trades and a backtest summary into the vector DB for RAG usage."""
    if _get_rag_store is None:
        return
    try:
        vs = _get_rag_store()
        if vs is None:
            return
    except Exception:
        return

    # store each executed trade as a JSON document
    for item in report.get("days") or []:
        try:
            if not item.get("trade_executed"):
                continue
            payload = {
                "job_id": job_id,
                "ticker": (item.get("ticker") or report.get("meta", {}).get("ticker") or "").upper(),
                "date": item.get("date"),
                "side": item.get("side"),
                "size": item.get("size"),
                "entry_price": item.get("entry_price"),
                "exit_price": item.get("exit_price"),
                "net_return": item.get("net_return"),
                "pnl": item.get("pnl"),
                "daily_output": item.get("daily_output"),
            }
            metadata = {
                "source": "backtesting",
                "type": "trade",
                "symbol": payload.get("ticker"),
                "stage": "trade_history",
                "run_id": str(job_id),
                "job_id": job_id,
                "timestamp": int(time.time()),
            }
            try:
                vs.store_json(payload, metadata=metadata)
            except Exception:
                # best-effort: don't raise
                pass
        except Exception:
            continue

    # also store a compact backtest summary
    try:
        summary = report.get("summary") or {}
        meta = {"source": "backtesting", "type": "backtest_summary", "job_id": job_id, "timestamp": int(time.time())}
        try:
            vs.store_json({"job_id": job_id, "summary": summary}, metadata=meta)
        except Exception:
            pass
    except Exception:
        pass


def run_backtesting_job(job_id: int, ticker: str, start_date: str, end_date: str, decision_policy: str, price_csv: str | None = None) -> None:
    db = SessionLocal()
    try:
        _set_status(db, job_id, "running", progress=5)
        _add_state(db, job_id, "started", {"at": _now_iso(), "ticker": ticker, "start_date": start_date, "end_date": end_date})

        if settings.jobs_fake_run:
            summary = {
                "trades": 1,
                "win_rate": 1.0,
                "avg_return": 0.012,
                "total_return": 0.012,
                "final_cash": 101200.0,
                "trading_days": 1,
            }
            report = {
                "meta": {
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "decision_policy": decision_policy,
                },
                "summary": summary,
                "artifacts": {
                    "daily_outputs_dir": "",
                    "daily_summary_csv": "",
                },
                "days": [
                    {
                        "ticker": ticker,
                        "date": start_date,
                        "side": "BUY",
                        "size": 0.2,
                        "trade_executed": True,
                        "entry_date": start_date,
                        "exit_date": end_date,
                        "entry_price": 100.0,
                        "exit_price": 101.2,
                        "net_return": 0.012,
                        "pnl": 1200.0,
                    }
                ],
            }
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path("outputs/backtesting") / f"api_backtest_{job_id}_{run_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            from backtesting.day_by_day import run_period

            def _on_progress(payload: dict) -> None:
                try:
                    p = payload if isinstance(payload, dict) else {}
                    progress_val = p.get("progress")
                    progress = int(progress_val) if progress_val is not None else None
                    if progress is not None:
                        progress = max(5, min(progress, 99))
                    _set_status(db, job_id, "running", progress=progress)
                    stage = f"running_{str(p.get('event') or 'update')}"
                    _add_state(
                        db,
                        job_id,
                        stage,
                        {
                            "at": _now_iso(),
                            "ticker": ticker,
                            **p,
                        },
                    )
                except Exception:
                    # best-effort: progress reporting must not break the run
                    pass
                if p.get("event") == "day_completed" and p.get("trade_executed"):
                    try:
                        _store_single_trade_row(db, job_id, ticker, p)
                    except Exception:
                        pass

            csv_path, price_source, price_source_detail = _resolve_price_csv_path(ticker, start_date, end_date, price_csv)
            _add_state(
                db,
                job_id,
                "price_data_resolved",
                {
                    "at": _now_iso(),
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "price_source": price_source,
                    "price_csv": str(csv_path).replace("\\", "/"),
                    "detail": price_source_detail,
                },
            )
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path("outputs/backtesting") / f"api_backtest_{job_id}_{ticker.upper()}_{run_id}"
            report = run_period(
                ticker=ticker.upper(),
                start_date=start_date,
                end_date=end_date,
                price_csv=csv_path,
                template_path="OUTPUT_TEMPLATE.TXT",
                model=None,
                initial_cash=100000.0,
                initial_shares=0.0,
                hold_days=5,
                fee_bps=5.0,
                slippage_bps=5.0,
                allow_price_fetch=False,
                allow_news_fetch=True,
                enforce_runtime_cutoff=False,
                decision_policy=decision_policy,
                out_dir=out_dir,
                progress_callback=_on_progress,
            )
            report_path = out_dir / "report.json"
            report.setdefault("meta", {})["price_source"] = price_source
            if price_source_detail:
                report.setdefault("meta", {})["price_source_detail"] = price_source_detail
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        row = db.query(BacktestingJob).filter(BacktestingJob.id == job_id).first()
        if row:
            row.report_path = str(report_path).replace("\\", "/")
            row.summary_json = json.dumps(report.get("summary") or {}, ensure_ascii=False)
            row.progress = 100
            row.status = "completed"
            row.error = None
            db.commit()

        _store_trade_history_for_job(db, job_id, report)
        # best-effort: also mirror trade history / summary into the vector DB so agents can RAG on past trades
        try:
            _store_trades_to_vector_store(job_id, report)
        except Exception:
            pass
        _add_state(db, job_id, "completed", {"at": _now_iso(), "report_path": str(report_path).replace("\\", "/")})
    except Exception as exc:
        _set_status(db, job_id, "failed", progress=100, error=str(exc))
        _add_state(db, job_id, "failed", {"at": _now_iso(), "error": str(exc)})
    finally:
        db.close()


def get_trade_history(db: Session, ticker: str | None = None, start: str | None = None, end: str | None = None) -> list[dict]:
    query = db.query(TradeHistory)
    if ticker:
        query = query.filter(TradeHistory.ticker == ticker.upper())
    if start:
        query = query.filter(TradeHistory.trade_date >= start)
    if end:
        query = query.filter(TradeHistory.trade_date <= end)

    rows = query.order_by(TradeHistory.trade_date.desc(), TradeHistory.id.desc()).all()
    return [
        {
            "id": row.id,
            "ticker": row.ticker,
            "trade_date": row.trade_date,
            "side": row.side,
            "size": row.size,
            "entry_date": row.entry_date,
            "exit_date": row.exit_date,
            "entry_price": row.entry_price,
            "exit_price": row.exit_price,
            "net_return": row.net_return,
            "pnl": row.pnl,
            "source_type": row.source_type,
            "source_job_id": row.source_job_id,
        }
        for row in rows
    ]
