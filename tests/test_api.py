import os
from fastapi.testclient import TestClient
from app.main import app

os.environ.setdefault("JOBS_FAKE_RUN", "1")
os.environ.setdefault("STOCK_FAKE_DATA", "1")

client = TestClient(app)


def test_config_get_put_reset():
    r = client.get("/api/v1/config")
    assert r.status_code == 200

    r = client.put(
        "/api/v1/config",
        json={"enabled_agents": ["market", "news"], "model": "qwen2.5:14b", "params": {"temp": 0.2}},
    )
    assert r.status_code == 200

    r = client.post("/api/v1/config/reset")
    assert r.status_code == 200


def test_prompts_post_put():
    r = client.post("/api/v1/config/prompts", json={"agent_name": "MARKET_ANALYST", "prompt_text": "hello"})
    assert r.status_code == 200

    r = client.post("/api/v1/config/prompts", json={"agent_name": "MARKET_ANALYST"})
    assert r.status_code == 200

    r = client.put("/api/v1/config/prompts", json={"agent_name": "MARKET_ANALYST", "prompt_text": "updated"})
    assert r.status_code == 200


def test_jobs_analyze_and_status():
    r = client.post("/api/v1/jobs/analyze", json={"ticker": "TSLA", "target_date": "2026-02-12"})
    assert r.status_code == 200
    job_id = r.json()["jobId"]

    r = client.get(f"/api/v1/jobs/status?jobId={job_id}")
    assert r.status_code == 200
    assert r.json()["status"] in ["completed", "running", "queued"]


def test_stock():
    r = client.get("/api/v1/stock/TSLA?start=2026-02-01&end=2026-02-05")
    assert r.status_code == 200
    assert r.json()["ticker"] == "TSLA"
    assert len(r.json()["points"]) > 0
