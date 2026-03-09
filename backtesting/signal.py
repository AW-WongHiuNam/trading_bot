from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeSignal:
    ticker: str
    side: str
    size: float
    confidence: float
    rationale: str
    blocked_reason: str | None = None


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_side(value: str | None) -> str:
    text = (value or "").strip().upper()
    if text in {"BUY", "LONG"}:
        return "BUY"
    if text in {"SELL", "SHORT"}:
        return "SELL"
    return "NO_TRADE"


def derive_signal(result: dict, ticker_fallback: str = "UNKNOWN", decision_policy: str = "strict") -> TradeSignal:
    manager = result.get("manager_decision") or {}
    trader = result.get("trader_proposal") or {}
    risk = result.get("risk") or {}
    researchers = result.get("researchers") or {}
    bull = researchers.get("bull") if isinstance(researchers, dict) else {}
    bear = researchers.get("bear") if isinstance(researchers, dict) else {}

    manager_decision = str(manager.get("decision") or "").strip().lower()
    risk_score = _to_float(risk.get("risk_score"), 0.0)

    side = _normalize_side(trader.get("side"))
    size = _to_float(trader.get("size"), 0.0)
    confidence = _to_float(trader.get("confidence"), 0.0)
    ticker = str(trader.get("ticker") or ticker_fallback)
    rationale = str(trader.get("rationale") or manager.get("reason") or "")

    policy = (decision_policy or "strict").strip().lower()
    blocked_reason = None

    if risk_score >= 80:
        side = "NO_TRADE"
        size = 0.0
        blocked_reason = f"risk_score={risk_score} >= 80"
    elif policy == "strict":
        if manager_decision == "require_manual":
            side = "NO_TRADE"
            size = 0.0
            blocked_reason = "manager_decision=require_manual"
        elif side == "NO_TRADE":
            blocked_reason = "trader_proposal.side=NO_TRADE"
    elif policy == "backtest":
        needs_override = (manager_decision == "require_manual") or (side == "NO_TRADE")
        if needs_override:
            bull_conf = _to_float((bull or {}).get("confidence"), 0.0) if isinstance(bull, dict) else 0.0
            bear_conf = _to_float((bear or {}).get("confidence"), 0.0) if isinstance(bear, dict) else 0.0
            spread = bull_conf - bear_conf

            if spread > 0.05:
                side = "BUY"
            elif spread < -0.05:
                side = "SELL"
            else:
                side = "NO_TRADE"

            if side == "NO_TRADE":
                size = 0.0
                blocked_reason = "backtest_policy_no_edge"
            else:
                base_size = min(0.25, max(0.05, abs(spread)))
                if risk_score >= 70:
                    base_size *= 0.5
                size = round(base_size, 4)
                confidence = round(max(bull_conf, bear_conf), 4)
                blocked_reason = f"backtest_policy_override({manager_decision or 'no_manager'})"
                rationale = (
                    rationale
                    + f" | policy=backtest spread={spread:.4f} bull={bull_conf:.4f} bear={bear_conf:.4f}"
                ).strip(" |")
    elif policy == "auto":
        bull_conf = _to_float((bull or {}).get("confidence"), 0.0) if isinstance(bull, dict) else 0.0
        bear_conf = _to_float((bear or {}).get("confidence"), 0.0) if isinstance(bear, dict) else 0.0
        spread = bull_conf - bear_conf

        # In auto mode we explicitly do NOT block on manager_decision=require_manual.
        # Prefer trader output if it's directional, otherwise fallback to confidence spread.
        if side not in {"BUY", "SELL"}:
            if spread > 0.03:
                side = "BUY"
            elif spread < -0.03:
                side = "SELL"
            else:
                side = "NO_TRADE"

        if side == "NO_TRADE":
            size = 0.0
            blocked_reason = "auto_policy_no_edge"
        else:
            # Size policy: cap at small/medium exposure with risk-aware scaling.
            # Base from trader size when it is a fraction (0~1), otherwise derive from confidence spread.
            if 0.0 < size <= 1.0:
                base_size = size
            else:
                base_size = min(0.35, max(0.08, abs(spread)))

            if risk_score >= 85:
                base_size *= 0.0
                side = "NO_TRADE"
                blocked_reason = f"auto_policy_risk_block({risk_score})"
            elif risk_score >= 75:
                base_size *= 0.35
            elif risk_score >= 65:
                base_size *= 0.60
            elif risk_score >= 55:
                base_size *= 0.80

            size = round(min(0.40, max(0.0, base_size)), 4)
            if side == "NO_TRADE" or size <= 0:
                size = 0.0
            else:
                confidence = round(max(confidence, bull_conf, bear_conf), 4)
                blocked_reason = f"auto_policy_execute(risk={risk_score},spread={spread:.4f})"
                rationale = (
                    rationale
                    + f" | policy=auto manager={manager_decision or 'none'} spread={spread:.4f} bull={bull_conf:.4f} bear={bear_conf:.4f} risk={risk_score:.2f}"
                ).strip(" |")
    else:
        raise ValueError(f"Unsupported decision_policy: {decision_policy}")

    return TradeSignal(
        ticker=ticker,
        side=side,
        size=size,
        confidence=confidence,
        rationale=rationale,
        blocked_reason=blocked_reason,
    )
