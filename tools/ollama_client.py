import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# prefer chat model env var if provided
OLLAMA_MODEL = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL") or "deepseek-r1:latest"
OLLAMA_COMPLETION_PATH = os.getenv("OLLAMA_COMPLETION_PATH", "/api/generate")


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}

def generate(prompt: str, model: str | None = None, timeout: int = 30) -> str:
    model = model or OLLAMA_MODEL
    try:
        payload: dict = {"model": model, "prompt": prompt}

        # Encourage strict JSON-only outputs when prompts require it.
        # Ollama supports `format: "json"` for /api/generate; models may still fail,
        # but this usually improves compliance.
        if _env_flag("OLLAMA_FORCE_JSON", "1"):
            payload["format"] = "json"

        if _env_flag("OLLAMA_DISABLE_STREAM", "0"):
            payload["stream"] = False

        # use streaming so we can handle newline-delimited JSON from Ollama
        resp = requests.post(f"{OLLAMA_URL}{OLLAMA_COMPLETION_PATH}", json=payload, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return resp.text

        # accumulate text from each JSON line in the stream
        out_parts = []
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            # Ollama often sends NDJSON per line
            try:
                j = None
                import json as _json
                j = _json.loads(line)
                # Prefer 'response'. Do NOT append chain-of-thought fields like 'thinking'.
                if isinstance(j, dict):
                    if j.get("response"):
                        out_parts.append(str(j.get("response")))
                    elif j.get("output"):
                        out_parts.append(str(j.get("output")))
                    # stop if done flag is true
                    if j.get("done") is True:
                        break
                else:
                    out_parts.append(line)
            except Exception:
                # not JSON, append raw
                out_parts.append(line)

        return "".join(out_parts)
    except Exception as e:
        return f"[OLLAMA_ERROR] {e}\nPROMPT:\n{prompt}"
