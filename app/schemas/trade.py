from pydantic import BaseModel


class TradeHistoryItem(BaseModel):
    id: int
    ticker: str
    trade_date: str
    side: str
    size: str
    entry_date: str | None = None
    exit_date: str | None = None
    entry_price: str | None = None
    exit_price: str | None = None
    net_return: str | None = None
    pnl: str | None = None
    source_type: str
    source_job_id: int | None = None


class TradeHistoryOut(BaseModel):
    items: list[TradeHistoryItem]
