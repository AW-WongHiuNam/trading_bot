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
- GET /api/v1/backtesting/list
- POST /api/v1/backtesting/analyze
- GET /api/v1/backtesting/status?jobId=
- GET /api/v1/backtesting/result?jobId=
- GET /api/v1/trade-history?ticket=&start=&end=

## Frontend API Contract (Trading/Backtesting)

### 1) Create Backtesting Job
- **POST** `/api/v1/backtesting/analyze`

Request body:
```json
{
	"ticker": "NVDA",
	"start_date": "2025-10-01",
	"end_date": "2026-01-22",
	"decision_policy": "auto",
	"price_csv": "D:/school/fyp/bot_v1/stock_price/HistoricalData_NVDA.csv"
}
```

Notes:
- `price_csv` is optional.
- If omitted, the backtesting service now tries to fetch stock data from EasyTradingBackend `POST /api/price` and writes a compatible cached CSV for the existing backtesting engine.
- If that backend fetch fails, it falls back to `stock_price/HistoricalData_<TICKER>.csv`.

Response:
```json
{
	"jobId": 12
}
```

### 2) Backtesting Job Status
- **GET** `/api/v1/backtesting/status?jobId=12`

Response:
```json
{
	"jobId": 12,
	"status": "completed",
	"progress": 100,
	"error": null,
	"latest_state": {
		"at": "2026-03-24T08:00:00",
		"report_path": "outputs/backtesting/api_backtest_12_20260324_080000/report.json"
	}
}
```

### 3) Backtesting Result
- **GET** `/api/v1/backtesting/result?jobId=12`

Response:
```json
{
	"jobId": 12,
	"status": "completed",
	"summary": {
		"trades": 65,
		"win_rate": 0.44,
		"avg_net_return": -0.0033,
		"total_net_return": -0.21,
		"final_cash": 98529.74,
		"trading_days": 78
	},
	"report_path": "outputs/backtesting/api_backtest_12_20260324_080000/report.json",
	"artifacts": {
		"daily_outputs_dir": "outputs/backtesting/api_backtest_12_20260324_080000/daily_outputs",
		"daily_summary_csv": "outputs/backtesting/api_backtest_12_20260324_080000/daily_summary.csv"
	}
}
```

### 4) Backtesting Job List
- **GET** `/api/v1/backtesting/list`
- Optional filter: `/api/v1/backtesting/list?ticker=NVDA`

Response:
```json
{
	"items": [
		{
			"jobId": 12,
			"ticker": "NVDA",
			"start_date": "2025-10-01",
			"end_date": "2026-01-22",
			"decision_policy": "auto",
			"status": "completed",
			"progress": 100,
			"created_at": "2026-03-24T07:50:00",
			"updated_at": "2026-03-24T08:00:02"
		}
	]
}
```

### 5) Trade History
- **GET** `/api/v1/trade-history?ticket=NVDA&start=2025-10-01&end=2026-01-22`
- `ticker` query is also accepted; `ticket` and `ticker` are treated as same filter.

Response:
```json
{
	"items": [
		{
			"id": 101,
			"ticker": "NVDA",
			"trade_date": "2025-10-01",
			"side": "BUY",
			"size": "0.25",
			"entry_date": "2025-10-02",
			"exit_date": "2025-10-09",
			"entry_price": "189.6",
			"exit_price": "192.57",
			"net_return": "0.0146645569",
			"pnl": "366.6139",
			"source_type": "backtesting",
			"source_job_id": 12
		}
	]
}
```

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