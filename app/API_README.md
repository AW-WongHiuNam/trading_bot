# FastAPI (Trading Agents) - Implementation Notes

## Run
```bash
uvicorn app.main:app --reload
```

## Test (all APIs)
```bash
set JOBS_FAKE_RUN=1
set STOCK_FAKE_DATA=1
pytest -q
```

## Endpoints
- GET /api/v1/config
- PUT /api/v1/config
- POST /api/v1/config/prompts
- PUT /api/v1/config/prompts
- POST /api/v1/config/reset
- POST /api/v1/jobs/analyze
- GET /api/v1/jobs/status?jobId=
- GET /api/v1/stock/{ticker}?start=&end=
