from pydantic import BaseModel


class AnalyzeJobIn(BaseModel):
    ticker: str
    target_date: str
    account_cash: float = 100000.0
    account_shares: float = 0.0


class AnalyzeJobOut(BaseModel):
    jobId: int


class JobStatusOut(BaseModel):
    jobId: int
    status: str
    error: str | None
    latest_state: dict | None
