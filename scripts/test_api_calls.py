import os
import time
import requests

BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")


def show(resp):
    print(f"\n{resp.request.method} {resp.url}")
    print(f"Status: {resp.status_code}")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


def main():
    show(requests.get(f"{BASE}/api/v1/config"))

    show(
        requests.put(
            f"{BASE}/api/v1/config",
            json={"enabled_agents": ["market", "news"], "model": "qwen2.5:14b", "params": {"temp": 0.2}},
        )
    )

    show(requests.post(f"{BASE}/api/v1/config/reset"))

    show(requests.post(f"{BASE}/api/v1/config/prompts", json={"agent_name": "MARKET_ANALYST", "prompt_text": "test prompt"}))

    show(requests.post(f"{BASE}/api/v1/config/prompts", json={"agent_name": "MARKET_ANALYST"}))

    show(requests.put(f"{BASE}/api/v1/config/prompts", json={"agent_name": "MARKET_ANALYST", "prompt_text": "updated prompt"}))

    r = requests.post(f"{BASE}/api/v1/jobs/analyze", json={"ticker": "TSLA", "target_date": "2026-02-12"})
    show(r)
    job_id = r.json().get("jobId")

    if job_id:
        time.sleep(0.5)
        show(requests.get(f"{BASE}/api/v1/jobs/status", params={"jobId": job_id}))

    show(requests.get(f"{BASE}/api/v1/stock/TSLA", params={"start": "2026-02-01", "end": "2026-02-05"}))


if __name__ == "__main__":
    main()
