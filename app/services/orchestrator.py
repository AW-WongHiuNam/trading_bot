from app.config import settings


def run_analysis_job(ticker: str, target_date: str, account_cash: float = 100000.0, account_shares: float = 0.0) -> dict:
    if settings.jobs_fake_run:
        return {
            "ticker": ticker,
            "target_date": target_date,
            "result": {
                "mode": "FAKE_RUN_OK",
                "context": {
                    "target_date": target_date,
                    "account_cash": account_cash,
                    "account_shares": account_shares,
                },
            },
        }
    try:
        from chains.langchain_chains import run_langchain_flow

        result = run_langchain_flow(
            ticker=ticker,
            target_date=target_date,
            account_cash=account_cash,
            account_shares=account_shares,
        )
        return {"ticker": ticker, "target_date": target_date, "result": result}
    except Exception as e:
        return {"ticker": ticker, "target_date": target_date, "error": str(e)}
