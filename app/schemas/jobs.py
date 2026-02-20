from pydantic import BaseModel


class AnalyzeJobIn(BaseModel):
    ticker: str
    target_date: str


class AnalyzeJobOut(BaseModel):
    jobId: int


class JobStatusOut(BaseModel):
    jobId: int
    status: str
    error: str | None
    latest_state: dict | None
