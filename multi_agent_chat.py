"""Simple multi-agent conversation demo using ollama Python API.
Each agent has its own `system` role prompt and a private memory list.
Agents take turns calling `ollama.chat` (or `ollama.generate` if preferred).
"""
import json
import time
import ollama
from config import get_settings
from vector_store_sqlite import VectorStore
from alpha_fetch import fetch_av

_cfg = get_settings()
DEFAULT_CHAT_MODEL = _cfg.ollama_chat_model or "qwen2.5:14b"

# Shared vector store for retrieval and grounding
_vector_store = VectorStore(
    table_name=_cfg.sqlite_table,
    sqlite_path=_cfg.sqlite_path,
    index_path=_cfg.vector_index_path,
    vector_dim=_cfg.vector_dim,
    ann_space=_cfg.ann_index_space,
    ann_ef=_cfg.ann_ef,
    ann_m=_cfg.ann_m,
    ollama_model=_cfg.ollama_embed_model,
    ollama_url=_cfg.ollama_embed_url,
    force_mock_embed=False,
)


def _print_tool_result(name: str, payload, max_chars: int | None = 800) -> None:
    try:
        rendered = json.dumps(payload, ensure_ascii=False)
    except Exception:
        rendered = str(payload)
    if max_chars is not None and max_chars > 0 and len(rendered) > max_chars:
        rendered = rendered[:max_chars] + " ...(truncated)"
    print(f"[tool] {name}: {rendered}")

class Agent:
    def __init__(self, name: str, system_prompt: str, model: str | None = None):
        self.name = name
        self.system = system_prompt
        self.private_memory = []  # list[str]
        self.model = model or DEFAULT_CHAT_MODEL

    def make_messages(self, transcript):
        # transcript: list of (speaker, text)
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        # include private memory as additional system entries
        for mem in self.private_memory:
            msgs.append({"role": "system", "content": mem})
        # convert transcript: if speaker == self -> assistant else user
        for speaker, text in transcript:
            role = 'assistant' if speaker == self.name else 'user'
            content = f"{speaker}: {text}"
            msgs.append({"role": role, "content": content})
        return msgs


def extract_response_text(resp):
    # Try common shapes: attribute `response`, mapping-like items, or iterator of pairs
    # 1) attribute
    if hasattr(resp, 'response'):
        return getattr(resp, 'response')
    # 1b) attribute message.content (ollama.chat)
    if hasattr(resp, 'message'):
        try:
            msg = getattr(resp, 'message')
            if hasattr(msg, 'content'):
                return getattr(msg, 'content')
        except Exception:
            pass
    # 2) try dict-like
    try:
        if isinstance(resp, dict) and 'response' in resp:
            return resp['response']
        if isinstance(resp, dict) and 'message' in resp:
            msg = resp.get('message')
            if isinstance(msg, dict) and 'content' in msg:
                return msg['content']
    except Exception:
        pass
    # 3) iterate and look for tuple pairs
    try:
        parts = []
        for part in resp:
            # dataclass-like iteration may yield (key, value)
            try:
                if isinstance(part, tuple) and len(part) == 2 and part[0] == 'response':
                    return part[1]
                parts.append(str(part))
            except Exception:
                parts.append(str(part))
        if parts:
            return '\n'.join(parts)
    except TypeError:
        pass
    return str(resp)


def search_qdrant(query: str, top_k: int | None = None):
    """Vector search against the configured vector store and return compact rows for the agent."""
    if not query:
        return {"error": "missing query"}
    k = top_k or _cfg.vector_top_k or 5
    try:
        items = _vector_store.retrieve(query, top_k=k)
    except Exception as e:
        return {"error": f"vector query failed: {e}"}

    results = []
    for doc, payload, score in items:
        results.append(
            {
                "score": round(score, 4),
                "source": payload.get("source", "unknown"),
                "symbol": payload.get("symbol") or payload.get("tickers"),
                "chunk": payload.get("chunk"),
                "snippet": (doc or "")[:280],
            }
        )
    return {"query": query, "hits": results}


def _preview_feed(feed, limit: int | None = 3):
    preview = []
    items = feed if limit is None or limit <= 0 else feed[:limit]
    for item in items:
        preview.append(
            {
                "title": item.get("title"),
                "summary": item.get("summary"),
                "url": item.get("url"),
                "sentiment": item.get("overall_sentiment_label"),
                "time": item.get("time_published"),
            }
        )
    return preview


def fetch_and_store_news(
    topic_or_ticker: str,
    *,
    return_feed: bool = False,
    preview_n: int | None = 3,
    do_search: bool = False,
    return_full_feed: bool = False,
):
    """Call Alpha Vantage NEWS_SENTIMENT, store to PostgreSQL, optionally return feed preview & immediate search hits."""
    api_key = _cfg.alphavantage_api_key
    if not api_key:
        return {"error": "ALPHAVANTAGE_API_KEY missing"}

    params = {"tickers": topic_or_ticker} if topic_or_ticker else {}
    try:
        data = fetch_av("NEWS_SENTIMENT", api_key=api_key, params=params)
    except Exception as e:
        return {"error": f"alpha fetch failed: {e}"}

    feed = []
    if isinstance(data, dict) and isinstance(data.get("feed"), list):
        feed = data.get("feed", [])

    try:
        metadata = {
            "source": "alpha_fetch",
            "function": "NEWS_SENTIMENT",
            "tickers": topic_or_ticker,
            "timestamp": int(time.time()),
            "type": "api_response",
        }
        _vector_store.store_response(
            json.dumps(data, ensure_ascii=False),
            metadata=metadata,
            chunk_size=_cfg.chunk_size,
            overlap=_cfg.chunk_overlap,
        )
    except Exception as e:
        return {"warning": f"stored fetch failed: {e}", "items": len(feed)}

    resp = {"stored_items": len(feed) or "unknown", "tickers": topic_or_ticker}

    if return_feed:
        if return_full_feed:
            resp["feed"] = _preview_feed(feed, limit=None)
        else:
            resp["feed_preview"] = _preview_feed(feed, limit=preview_n)

    if do_search:
        resp["search_hits"] = search_qdrant(topic_or_ticker)

    return resp


def handle_tool_request(text):
    """Parse any line starting with 'TOOL:'; support multiple directives per message, return on first supported."""
    if text is None:
        return None

    raw = text if isinstance(text, str) else str(text)

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("TOOL:"):
            continue

        body = stripped.split(":", 1)[1].strip()
        if not body:
            continue

        if "=" in body:
            name, arg = body.split("=", 1)
        else:
            name, arg = body, ""

        name = name.strip().lower().replace("-", "_")
        arg = arg.strip()

        if name in {"search_news", "search"}:
            return name, search_qdrant(arg), stripped
        if name in {"fetch_news", "fetch_latest"}:
            return name, fetch_and_store_news(arg), stripped
        if name in {"fetch_news_full", "fetch_and_read"}:
            return name, fetch_and_store_news(arg, return_feed=True, preview_n=None, do_search=True, return_full_feed=True), stripped

        print(f"[tool] unknown tool '{name}' (arg='{arg}'). Supported: fetch_news, fetch_news_full, search_news")

    return None


def run_conversation(agent_a: Agent, agent_b: Agent, turns: int = 4, pause: float = 0.5):
    transcript = []  # list of (speaker, text)

    agents = [agent_a, agent_b]
    for i in range(turns):
        speaker = agents[i % 2]
        # prepare messages for this speaker
        messages = speaker.make_messages(transcript)
        print(f"\n--- {speaker.name} generating (turn {i+1}) ---")
        try:
            resp = ollama.chat(model=speaker.model, messages=messages)
            text = extract_response_text(resp)
        except Exception as e:
            print('chat failed:', e)
            last_text = transcript[-1][1] if transcript else 'Start the discussion.'
            prompt = f"{speaker.name}, reply to: {last_text}"
            try:
                resp = ollama.generate(model=speaker.model, prompt=prompt)
                text = extract_response_text(resp)
            except Exception as e2:
                print('generate fallback failed:', e2)
                text = f"(error generating reply: {e2})"

        text = text.strip() if isinstance(text, str) else str(text)
        print(f"{speaker.name}: {text}\n")

        tool_result = handle_tool_request(text)
        if tool_result is not None:
            tool_name, tool_payload, tool_line = tool_result
            first_line = tool_line or (text.splitlines()[0] if isinstance(text, str) else str(text))
            print(f"[tool] detected {tool_name} from {speaker.name}: {first_line}")
            max_chars = None if tool_name == "fetch_news_full" else 800
            _print_tool_result(tool_name, tool_payload, max_chars=max_chars)
            # Record tool output in transcript to ground the next turn.
            as_text = json.dumps(tool_payload, ensure_ascii=False)
            transcript.append((f"Tool:{tool_name}", as_text))
            speaker.private_memory.append(f"Tool {tool_name} returned: {as_text}")
            other = agents[(i + 1) % 2]
            other.private_memory.append(f"I saw tool {tool_name}: {as_text}")
            time.sleep(pause)
            continue

        transcript.append((speaker.name, text))

        speaker.private_memory.append(f"I said: {text}")
        other = agents[(i + 1) % 2]
        other.private_memory.append(f"I heard {speaker.name} say: {text}")

        time.sleep(pause)

    return transcript


if __name__ == '__main__':
    a = Agent('Alice', system_prompt='You are Alice, a concise analyst of stock. please try to call tools and tell me what you got. When you need data, issue TOOL commands like "TOOL: fetch_news_full=AAPL" or "TOOL: search_news=AI chips". After calling, summarize what you got. ')
    b = Agent('Bob', system_prompt='You are Bob, a a concise analyst of stack. discuss the stock with alice')
    conv = run_conversation(a, b, turns=6, pause=0.3)
    print('\n--- Final transcript ---')
    for s, t in conv:
        print(f"{s}: {t}")
