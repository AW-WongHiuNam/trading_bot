from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.schemas.backtesting import (
    BacktestingAnalyzeIn,
    BacktestingAnalyzeOut,
    BacktestingListOut,
    BacktestingResultOut,
    BacktestingStatusOut,
)
from app.services.backtesting_service import (
    create_backtesting_job,
    get_backtesting_result,
    get_backtesting_status,
    list_backtesting_jobs,
    run_backtesting_job,
)

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/backtesting/list", response_model=BacktestingListOut)
def list_jobs(ticker: str | None = Query(default=None), db: Session = Depends(get_db)):
    return {"items": list_backtesting_jobs(db, ticker=ticker)}


@router.post("/backtesting/analyze", response_model=BacktestingAnalyzeOut)
def analyze(payload: BacktestingAnalyzeIn, bg: BackgroundTasks, db: Session = Depends(get_db)):
    ticker = payload.ticker.upper()
    job_id = create_backtesting_job(
        db,
        ticker=ticker,
        start_date=payload.start_date,
        end_date=payload.end_date,
        decision_policy=payload.decision_policy,
    )
    if settings.jobs_fake_run:
        run_backtesting_job(
            job_id=job_id,
            ticker=ticker,
            start_date=payload.start_date,
            end_date=payload.end_date,
            decision_policy=payload.decision_policy,
            price_csv=payload.price_csv,
        )
    else:
        bg.add_task(
            run_backtesting_job,
            job_id,
            ticker,
            payload.start_date,
            payload.end_date,
            payload.decision_policy,
            payload.price_csv,
        )
    return {"jobId": job_id}


@router.get("/backtesting/status", response_model=BacktestingStatusOut)
def status(jobId: int, db: Session = Depends(get_db)):
    result = get_backtesting_status(db, jobId)
    if not result:
        return {"jobId": jobId, "status": "not_found", "progress": 0, "error": "job not found", "latest_state": None}
    return result


@router.get("/backtesting/result", response_model=BacktestingResultOut)
def result(jobId: int, db: Session = Depends(get_db)):
    result_payload = get_backtesting_result(db, jobId)
    if not result_payload:
        return {"jobId": jobId, "status": "not_found", "summary": None, "report_path": None, "artifacts": None}
    return result_payload
