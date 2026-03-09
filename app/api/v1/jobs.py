from fastapi import APIRouter, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.schemas.jobs import AnalyzeJobIn, AnalyzeJobOut, JobStatusOut
from app.services.jobs_service import create_job, get_job_status, run_job
from app.config import settings

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/jobs/analyze", response_model=AnalyzeJobOut)
def analyze(payload: AnalyzeJobIn, bg: BackgroundTasks, db: Session = Depends(get_db)):
    job_id = create_job(db, payload.ticker, payload.target_date)
    if settings.jobs_fake_run:
        run_job(db, job_id, payload.ticker, payload.target_date, payload.account_cash, payload.account_shares)
    else:
        bg.add_task(run_job, db, job_id, payload.ticker, payload.target_date, payload.account_cash, payload.account_shares)
    return {"jobId": job_id}


@router.get("/jobs/status", response_model=JobStatusOut)
def status(jobId: int, db: Session = Depends(get_db)):
    result = get_job_status(db, jobId)
    if not result:
        return {"jobId": jobId, "status": "not_found", "error": "job not found", "latest_state": None}
    return result
