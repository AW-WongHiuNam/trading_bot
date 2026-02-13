from prompts import TRADER_PROMPT
from tools.ollama_client import generate as ollama_generate
from datetime import datetime

def trader_proposal(decision: dict, consensus: str, ticker: str = "TSLA") -> dict:
    input_text = f"Decision: {decision}\nConsensus: {consensus}"
    prompt = TRADER_PROMPT.replace('{input}', input_text)
    resp = ollama_generate(prompt)
    # For demo, parse minimal fields heuristically
    return {
        "ticker": ticker,
        "side": "SELL",
        "size": 100,
        "entry": 243.75,
        "stop": 255.0,
        "target": 210.0,
        "rationale": resp[:800],
        "confidence": 0.7,
        "created_by": "TRADER",
        "timestamp": datetime.utcnow().isoformat(),
    }
