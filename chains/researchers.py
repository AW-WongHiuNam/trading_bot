from prompts import RESEARCHER_PROMPT
from tools.ollama_client import generate as ollama_generate
from datetime import datetime

def researcher_stance(analyst_reports: dict, stance: str = "BULL") -> dict:
    merged = "\n\n".join([f"{k}: {v.get('summary','')[:800]}" for k,v in analyst_reports.items()])
    prompt = RESEARCHER_PROMPT.replace('{stance}', stance).replace('{input}', merged)
    resp = ollama_generate(prompt)
    # For demo, put resp into fields conservatively
    return {
        "stance": stance,
        "evidence": [resp[:800]],
        "counterarguments": [],
        "consensus_summary": resp[:1200],
        "final_label": "HOLD",
        "confidence": 0.7,
    }

def discussion(bull: dict, bear: dict) -> dict:
    # Combine and ask Ollama to synthesize
    prompt = (
        "Two researchers with opposing views. BULL:\n" + bull.get('consensus_summary','')[:800] +
        "\n\nBEAR:\n" + bear.get('consensus_summary','')[:800] +
        "\n\nPlease produce a short synthesis and state final_label and confidence."
    )
    resp = ollama_generate(prompt)
    return {
        "consensus": resp[:1500],
        "final_label": "HOLD",
        "confidence": 0.75,
    }
