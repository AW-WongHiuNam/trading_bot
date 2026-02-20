import json
from sqlalchemy.orm import Session
from app.db.models import Job, JobState
from app.services.orchestrator import run_analysis_job


def create_job(db: Session, ticker: str, target_date: str) -> int:
    job = Job(ticker=ticker, target_date=target_date, status="queued")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job.id


def set_job_status(db: Session, job_id: int, status: str, error: str | None = None) -> None:
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        job.status = status
        job.error = error
        db.commit()


def add_job_state(db: Session, job_id: int, stage: str, payload: dict) -> None:
    st = JobState(job_id=job_id, stage=stage, payload_json=json.dumps(payload))
    db.add(st)
    db.commit()


def get_job_status(db: Session, job_id: int) -> dict | None:
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return None
    latest = (
        db.query(JobState)
        .filter(JobState.job_id == job_id)
        .order_by(JobState.id.desc())
        .first()
    )
    latest_payload = json.loads(latest.payload_json) if latest else None
    return {
        "jobId": job.id,
        "status": job.status,
        "error": job.error,
        "latest_state": latest_payload,
    }


def run_job(db: Session, job_id: int, ticker: str, target_date: str) -> None:
    set_job_status(db, job_id, "running")
    result = run_analysis_job(ticker, target_date)
    if "error" in result:
        set_job_status(db, job_id, "failed", error=result["error"])
    else:
        add_job_state(db, job_id, "FINAL", result)
        set_job_status(db, job_id, "completed")
