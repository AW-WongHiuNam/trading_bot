from fastapi import FastAPI
from app.api.v1.config import router as config_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.stock import router as stock_router
from app.db.init_db import init_db

app = FastAPI(title="Trading Agents API", version="1.0.0")


@app.on_event("startup")
def on_startup() -> None:
    init_db()


app.include_router(config_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(stock_router, prefix="/api/v1")
