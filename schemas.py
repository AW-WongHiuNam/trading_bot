ANALYST_SCHEMA = {
    "role": str,
    "ticker": str,
    "timestamp": str,
    "summary": str,
    "key_points": list,
    "confidence": float,
    "recommendation_hint": str,
}

DISCUSSION_SCHEMA = {
    "stance": str,
    "evidence": list,
    "counterarguments": list,
    "consensus_summary": str,
    "final_label": str,
    "confidence": float,
}

RISK_SCHEMA = {
    "risk_score": int,
    "breach_flags": list,
    "explainers": list,
}

TRADING_PROPOSAL_SCHEMA = {
    "ticker": str,
    "side": str,
    "size": (int, float),
    "entry": float,
    "stop": float,
    "target": float,
    "rationale": str,
    "confidence": float,
    "created_by": str,
}
