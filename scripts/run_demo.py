import json
import os

from chains.langchain_chains import run_langchain_flow


def _compact_stage(name: str, obj: dict) -> dict:
    if not isinstance(obj, dict):
        return {"stage": name, "value": obj}

    # Common fields
    stage_name = obj.get("stage") if isinstance(obj.get("stage"), str) else name
    compact = {"stage": stage_name}
    for k in ("role", "ticker", "timestamp"):
        if k in obj:
            compact[k] = obj.get(k)

    # Stage-specific fields
    if name in ("MARKET_ANALYST", "SOCIAL_ANALYST", "NEWS_ANALYST", "FUNDAMENTALS_ANALYST"):
        for k in ("summary", "key_points", "confidence", "recommendation_hint", "tool_call", "error"):
            if k in obj:
                compact[k] = obj.get(k)
    elif name in ("BULL_RESEARCHER", "BEAR_RESEARCHER") or stage_name.startswith("BULL_DEBATE_R") or stage_name.startswith("BEAR_DEBATE_R"):
        for k in (
            "stance",
            "final_label",
            "consensus_summary",
            "confidence",
            "evidence",
            "counterarguments",
            "tool_call",
            "error",
        ):
            if k in obj:
                compact[k] = obj.get(k)
    elif name == "RISK_ANALYST":
        for k in ("risk_score", "breach_flags", "explainers", "tool_call", "error"):
            if k in obj:
                compact[k] = obj.get(k)
    elif name == "RISK_MANAGER":
        for k in ("decision", "reason", "next_steps", "tool_call", "tool", "function", "symbol", "error"):
            if k in obj:
                compact[k] = obj.get(k)
    elif name == "TRADER":
        for k in (
            "ticker",
            "side",
            "size",
            "entry",
            "stop",
            "target",
            "rationale",
            "confidence",
            "created_by",
            "tool_call",
            "error",
        ):
            if k in obj:
                compact[k] = obj.get(k)

    # If model returned only a text wrapper, keep it short.
    if isinstance(obj.get("text"), str) and "text" not in compact:
        t = obj.get("text", "")
        compact["text_snippet"] = (t[:500] + "...") if len(t) > 500 else t

    # If we didn't capture anything useful, fall back to whole object (but keep stage label)
    if len(compact.keys()) <= 2:
        compact["raw"] = obj
    return compact


def _print_json(title: str, data: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    model = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL") or "deepseek-r1:latest"
    ticker = os.getenv("TICKER") or "TSLA"
    result = run_langchain_flow("OUTPUT_TEMPLATE.TXT", model=model, ticker=ticker)

    # Print each agent output first (compact), then final summary
    analysts = result.get("analysts", {}) or {}
    researchers = result.get("researchers", {}) or {}
    bull_rounds = researchers.get("bull_rounds", []) or []
    bear_rounds = researchers.get("bear_rounds", []) or []
    risk = result.get("risk", {}) or {}
    manager = result.get("manager_decision", {}) or {}
    trader = result.get("trader_proposal", {}) or {}
    trace = result.get("trace", []) or []

    _print_json("MARKET_ANALYST", _compact_stage("MARKET_ANALYST", analysts.get("market", {})))
    _print_json("SOCIAL_ANALYST", _compact_stage("SOCIAL_ANALYST", analysts.get("social", {})))
    _print_json("NEWS_ANALYST", _compact_stage("NEWS_ANALYST", analysts.get("news", {})))
    _print_json("FUNDAMENTALS_ANALYST", _compact_stage("FUNDAMENTALS_ANALYST", analysts.get("fundamentals", {})))

    # Debate rounds (expected: 3 rounds each)
    if bull_rounds:
        for i, r in enumerate(bull_rounds, start=1):
            _print_json(f"BULL_DEBATE_R{i}", _compact_stage(f"BULL_DEBATE_R{i}", r if isinstance(r, dict) else {"value": r}))
    if bear_rounds:
        for i, r in enumerate(bear_rounds, start=1):
            _print_json(f"BEAR_DEBATE_R{i}", _compact_stage(f"BEAR_DEBATE_R{i}", r if isinstance(r, dict) else {"value": r}))

    # Last stance snapshot (final round output)
    _print_json("BULL_RESEARCHER", _compact_stage("BULL_RESEARCHER", researchers.get("bull", {})))
    _print_json("BEAR_RESEARCHER", _compact_stage("BEAR_RESEARCHER", researchers.get("bear", {})))
    _print_json("RISK_ANALYST", _compact_stage("RISK_ANALYST", risk))
    _print_json("RISK_MANAGER", _compact_stage("RISK_MANAGER", manager))
    _print_json("TRADER", _compact_stage("TRADER", trader))

    if trace:
        _print_json("TRACE", {"events": trace[-25:], "total": len(trace)})

    print("\n=== FINAL SUMMARY ===")
    print(trader.get("rationale", "(no rationale)"))


if __name__ == "__main__":
    main()
