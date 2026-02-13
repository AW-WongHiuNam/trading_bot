from prompts import RISK_ANALYST_PROMPT, RISK_MANAGER_PROMPT
from tools.ollama_client import generate as ollama_generate

def risk_analyst(discussion_summary: str) -> dict:
    prompt = RISK_ANALYST_PROMPT.replace('{input}', discussion_summary)
    resp = ollama_generate(prompt)
    # Demo scoring: simple heuristic fallback
    return {
        "risk_score": 60,
        "breach_flags": [],
        "explainers": [resp[:800]],
    }

def risk_manager(aggregated_info: str) -> dict:
    prompt = RISK_MANAGER_PROMPT.replace('{input}', aggregated_info)
    resp = ollama_generate(prompt)
    return {
        "decision": "approve" if "SELL" in aggregated_info or "HOLD" in aggregated_info else "require_manual",
        "reason": resp[:800],
        "next_steps": "execute" if "approve" else "manual_review",
    }
