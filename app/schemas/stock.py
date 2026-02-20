from pydantic import BaseModel


class StockPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class StockOut(BaseModel):
    ticker: str
    start: str
    end: str
    points: list[StockPoint]
