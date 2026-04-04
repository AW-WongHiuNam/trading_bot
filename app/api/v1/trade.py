from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.schemas.trade import TradeHistoryOut
from app.services.backtesting_service import get_trade_history

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/trade-history", response_model=TradeHistoryOut)
def trade_history(
    ticket: str | None = Query(default=None),
    ticker: str | None = Query(default=None),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    symbol = (ticket or ticker)
    items = get_trade_history(db, ticker=symbol, start=start, end=end)
    return {"items": items}
