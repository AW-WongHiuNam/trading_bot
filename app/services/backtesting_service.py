from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import time

from sqlalchemy.orm import Session

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
    return {
        "jobId": row.id,
        "status": row.status,
        "progress": row.progress,
        "error": row.error,
        "latest_state": latest_payload,
    }


def get_backtesting_result(db: Session, job_id: int) -> dict | None:
    row = db.query(BacktestingJob).filter(BacktestingJob.id == job_id).first()
    if not row:
        return None
    summary = json.loads(row.summary_json) if row.summary_json else None
    artifacts = None
    if row.report_path:
        report_file = Path(row.report_path)
        if report_file.exists():
            try:
                payload = json.loads(report_file.read_text(encoding="utf-8"))
                artifacts = payload.get("artifacts")
                if summary is None:
                    summary = payload.get("summary")
            except Exception:
                artifacts = None
    return {
        "jobId": row.id,
        "status": row.status,
        "summary": summary,
        "report_path": row.report_path,
        "artifacts": artifacts,
    }


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

            csv_path = Path(price_csv) if price_csv else Path("stock_price") / f"HistoricalData_{ticker.upper()}.csv"
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
            )
            report_path = out_dir / "report.json"
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
