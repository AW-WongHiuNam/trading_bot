from pydantic import BaseModel


class BacktestingAnalyzeIn(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    decision_policy: str = "auto"
    price_csv: str | None = None


class BacktestingAnalyzeOut(BaseModel):
    jobId: int


class BacktestingStatusOut(BaseModel):
    jobId: int
    status: str
    progress: int
    error: str | None
    latest_state: dict | None


class BacktestingJobItem(BaseModel):
    jobId: int
    ticker: str
    start_date: str
    end_date: str
    decision_policy: str
    status: str
    progress: int
    created_at: str | None = None
    updated_at: str | None = None


class BacktestingListOut(BaseModel):
    items: list[BacktestingJobItem]


class BacktestingResultOut(BaseModel):
    jobId: int
    status: str
    summary: dict | None
    report_path: str | None
    artifacts: dict | None = None
