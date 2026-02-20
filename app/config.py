import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    db_url: str = os.getenv("APP_DB_URL", "sqlite:///./app/data/app.db")
    jobs_fake_run: bool = os.getenv("JOBS_FAKE_RUN", "0") == "1"
    stock_fake_data: bool = os.getenv("STOCK_FAKE_DATA", "0") == "1"
    alpha_vantage_api_key: str | None = os.getenv("ALPHAVANTAGE_API_KEY")


settings = Settings()
