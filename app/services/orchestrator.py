from app.config import settings


def run_analysis_job(ticker: str, target_date: str) -> dict:
    if settings.jobs_fake_run:
        return {
            "ticker": ticker,
            "target_date": target_date,
            "result": "FAKE_RUN_OK",
        }
    try:
        from chains.langchain_chains import run_langchain_flow

        result = run_langchain_flow(ticker=ticker)
        return {"ticker": ticker, "target_date": target_date, "result": result}
    except Exception as e:
        return {"ticker": ticker, "target_date": target_date, "error": str(e)}
