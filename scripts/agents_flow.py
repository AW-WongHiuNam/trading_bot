import json

from chains.analysts import fundamentals_analyst, market_analyst, news_analyst, social_analyst
from chains.researchers import discussion, researcher_stance
from chains.risk import risk_analyst, risk_manager
from chains.trader import trader_proposal


def load_template(path: str = "OUTPUT_TEMPLATE.TXT"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_flow(template_path: str = "OUTPUT_TEMPLATE.TXT"):
    data = load_template(template_path)
    # For demo, extract existing reports
    stages = data.get("stages", {})
    market_report = stages.get("MARKET_ANALYST", {}).get("market_report", {}).get("preview", "")
    social_report = stages.get("SOCIAL_ANALYST", {}).get("sentiment_report", {}).get("preview", "")
    news_report = stages.get("NEWS_ANALYST", {}).get("news_report", {}).get("preview", "")
    fund_report = stages.get("FUNDAMENTALS_ANALYST", {}).get("fundamentals_report", {}).get("preview", "")

    a_market = market_analyst(market_report)
    a_social = social_analyst("\n".join(social_report if isinstance(social_report, list) else [social_report]))
    a_news = news_analyst(news_report)
    a_fund = fundamentals_analyst(fund_report)

    analyst_reports = {
        "market": a_market,
        "social": a_social,
        "news": a_news,
        "fundamentals": a_fund,
    }

    bull = researcher_stance(analyst_reports, stance="BULL")
    bear = researcher_stance(analyst_reports, stance="BEAR")
    disc = discussion(bull, bear)

    risk = risk_analyst(disc.get("consensus_summary", disc.get("consensus", "")))
    manager_decision = risk_manager(disc.get("consensus", ""))

    trader = trader_proposal(manager_decision, disc.get("consensus", ""))

    final = {
        "analysts": analyst_reports,
        "researchers": {"bull": bull, "bear": bear, "discussion": disc},
        "risk": risk,
        "manager_decision": manager_decision,
        "trader_proposal": trader,
    }

    with open("flow_output.json", "w", encoding="utf-8") as fout:
        json.dump(final, fout, ensure_ascii=False, indent=2)

    return final


if __name__ == "__main__":
    out = run_flow()
    print("Flow completed. Output written to flow_output.json")
