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

## Preferred scripts
- `python scripts/test_api_calls.py`
- `python scripts/run_demo.py`

## PowerShell examples
```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/v1/jobs/analyze' -Method Post -Body '{"ticker":"TSLA","target_date":"2026-02-12"}' -ContentType 'application/json'
```

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/v1/jobs/status?jobId=3'
```

```powershell
(Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/v1/jobs/status?jobId=3' -UseBasicParsing).Content
```

> 注意：PowerShell 中 `curl` 是 `Invoke-WebRequest` alias。若要使用真正 curl，請用 `curl.exe`。