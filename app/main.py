from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.backtesting import router as backtesting_router
from app.api.v1.config import router as config_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.stock import router as stock_router
from app.api.v1.trade import router as trade_router
from app.db.init_db import init_db

app = FastAPI(title="Trading Agents API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


app.include_router(config_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(stock_router, prefix="/api/v1")
app.include_router(backtesting_router, prefix="/api/v1")
app.include_router(trade_router, prefix="/api/v1")
