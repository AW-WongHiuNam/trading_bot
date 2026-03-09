import os
import time
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# prefer chat model env var if provided
OLLAMA_MODEL = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL") or "deepseek-r1:latest"
OLLAMA_COMPLETION_PATH = os.getenv("OLLAMA_COMPLETION_PATH", "/api/generate")


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.getenv(name, default).strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

def generate(prompt: str, model: str | None = None, timeout: int = 30) -> str:
    model = model or OLLAMA_MODEL
    completion_retries = max(0, _env_int("OLLAMA_COMPLETION_RETRIES", 3))
    completion_backoff = max(0.1, _env_float("OLLAMA_COMPLETION_BACKOFF_SEC", 1.0))

    payload: dict = {"model": model, "prompt": prompt}

    # Encourage strict JSON-only outputs when prompts require it.
    # Ollama supports `format: "json"` for /api/generate; models may still fail,
    # but this usually improves compliance.
    if _env_flag("OLLAMA_FORCE_JSON", "1"):
        payload["format"] = "json"

    if _env_flag("OLLAMA_DISABLE_STREAM", "0"):
        payload["stream"] = False

    last_error: Exception | None = None
    for attempt in range(completion_retries + 1):
        try:
            # use streaming so we can handle newline-delimited JSON from Ollama
            resp = requests.post(f"{OLLAMA_URL}{OLLAMA_COMPLETION_PATH}", json=payload, timeout=timeout, stream=True)
            if resp.status_code != 200:
                # Retry only transient server-side failures.
                if resp.status_code >= 500 and attempt < completion_retries:
                    time.sleep(completion_backoff * (2 ** attempt))
                    continue
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
            last_error = e
            if attempt >= completion_retries:
                break
            time.sleep(completion_backoff * (2 ** attempt))

    return f"[OLLAMA_ERROR] {last_error}\nPROMPT:\n{prompt}"
