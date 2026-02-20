from fastapi import APIRouter, HTTPException
from app.schemas.stock import StockOut
from app.services.stock_service import get_stock_data

router = APIRouter()


@router.get("/stock/{ticker}", response_model=StockOut)
def stock(ticker: str, start: str, end: str):
    try:
        points = get_stock_data(ticker, start, end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="stock fetch failed") from exc
    return {"ticker": ticker, "start": start, "end": end, "points": points}
