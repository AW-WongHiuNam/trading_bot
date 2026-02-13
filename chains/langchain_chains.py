try:
    from langchain.chains import LLMChain  # type: ignore
    from langchain.prompts import PromptTemplate  # type: ignore
    USE_LANGCHAIN = True
except Exception:
    LLMChain = None
    PromptTemplate = None
    USE_LANGCHAIN = False

from llms.ollama_llm import OllamaLLM
from tools.ollama_client import generate as ollama_generate
from tools.alpha_tool import call_alpha
from datetime import datetime
import json
from typing import TypedDict, NotRequired, Any
import os
import time

try:
    from langgraph.graph import StateGraph, END  # type: ignore
    USE_LANGGRAPH = True
except Exception:
    StateGraph = None
    END = None
    USE_LANGGRAPH = False

from prompts import MARKET_PROMPT, ANALYST_PROMPT, RESEARCHER_PROMPT, RISK_ANALYST_PROMPT, RISK_MANAGER_PROMPT, TRADER_PROMPT

try:
    from vector_store_sqlite import VectorStore  # type: ignore
except Exception:
    VectorStore = None  # type: ignore

try:
    from tools.rag_tool import rag_search  # type: ignore
except Exception:
    rag_search = None  # type: ignore


class FlowState(TypedDict, total=False):
    template_path: str
    model: NotRequired[str | None]
    ticker: str

    market_report: str
    social_report: str
    news_report: str
    fund_report: str

    trace: list[dict]
    baseline: dict
    baseline_ctx: str

    analysts: dict
    bull: dict
    bear: dict
    debate_round: int
    bull_last: str
    bear_last: str
    bull_rounds: list[dict]
    bear_rounds: list[dict]
    discussion_summary: str
    risk: dict
    manager_decision: dict
    trader_proposal: dict

    final: dict

    # Shared work-pipeline fields (graph-level tool_call loop + schema retry)
    work_stage: str
    work_kind: str
    work_input: str
    work_output: str
    work_parsed: dict
    work_required: list[str]
    work_schema_hint: str
    work_auto_tool_calls: list[dict]
    work_schema_pass: int
    work_tool_iter: int
    work_missing: list[str]
    work_stance: NotRequired[str]
    work_next: str


_RAG_STORE = None
_ALPHA_LAST_CALL_AT = 0.0
_ALPHA_MIN_INTERVAL_SEC = 1.1


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_rag_store():
    global _RAG_STORE
    if _RAG_STORE is not None:
        return _RAG_STORE
    if VectorStore is None:
        return None
    try:
        from config import get_settings

        cfg = get_settings()
        _RAG_STORE = VectorStore(
            table_name=cfg.sqlite_table,
            sqlite_path=cfg.sqlite_path,
            index_path=cfg.vector_index_path,
            vector_dim=cfg.vector_dim,
            ann_space=cfg.ann_index_space,
            ann_ef=cfg.ann_ef,
            ann_m=cfg.ann_m,
            ollama_model=cfg.ollama_embed_model,
            ollama_url=cfg.ollama_embed_url,
            force_mock_embed=False,
        )
        return _RAG_STORE
    except Exception:
        return None


def _rag_store_json(payload: Any, *, metadata: dict[str, Any], trace_event: dict | None = None) -> bool:
    vs = _get_rag_store()
    if vs is None:
        return False
    md = dict(metadata)
    md.setdefault("is_test", _env_truthy("RAG_IS_TEST"))
    try:
        vs.store_json(payload, metadata=md)
        return True
    except Exception:
        if isinstance(trace_event, dict):
            try:
                import traceback

                trace_event["rag_error_type"] = "store_failed"
                trace_event["rag_error"] = traceback.format_exc(limit=5)
            except Exception:
                trace_event["rag_error_type"] = "store_failed"
                trace_event["rag_error"] = "unknown"
        # best-effort: never break the main flow because storage failed
        return False


def _alpha_throttle() -> None:
    """Best-effort AlphaVantage throttling to reduce per-second rate limiting."""
    global _ALPHA_LAST_CALL_AT
    try:
        now = time.time()
        wait = (_ALPHA_LAST_CALL_AT + float(_ALPHA_MIN_INTERVAL_SEC)) - now
        if wait > 0:
            time.sleep(wait)
        _ALPHA_LAST_CALL_AT = time.time()
    except Exception:
        pass


def _alpha_cache_lookup(
    *,
    function: str,
    symbol: str | None,
    tickers: str | None,
    max_age_sec: float | None,
) -> tuple[object | None, float | None]:
    vs = _get_rag_store()
    if vs is None:
        return (None, None)

    min_created_at = None
    if max_age_sec is not None and float(max_age_sec) > 0:
        min_created_at = time.time() - float(max_age_sec)

    hit = None
    try:
        hit = vs.get_latest(
            source="alpha_tool",
            tool="alpha_fetch",
            function=function,
            symbol=symbol if symbol else None,
            tickers=tickers if tickers else None,
            type="tool_result",
            meta_equals={"alpha_ok": True},
            min_created_at=min_created_at,
            max_scan=800,
        )
    except Exception:
        hit = None

    if not hit:
        # fallback: try matching only by function (useful if symbol/tickers mismatch)
        try:
            hit = vs.get_latest(
                source="alpha_tool",
                tool="alpha_fetch",
                function=function,
                type="tool_result",
                meta_equals={"alpha_ok": True},
                min_created_at=min_created_at,
                max_scan=800,
            )
        except Exception:
            hit = None

    if not hit:
        return (None, None)

    doc, _md, created_at = hit
    age = None
    try:
        age = max(0.0, time.time() - float(created_at))
    except Exception:
        age = None

    try:
        return (json.loads(doc), age)
    except Exception:
        return ({"text": doc}, age)


def _run_llm_prompt_once(
    prompt_template: str,
    *,
    values: dict[str, str],
    model: str | None,
    input_variables: list[str],
) -> str:
    """Run a single LLM call for a given prompt template.

    Looping, tool calls, and schema repair are handled at the graph layer.
    """
    if USE_LANGCHAIN:
        llm = OllamaLLM(model=model)
        prompt = PromptTemplate(template=prompt_template, input_variables=input_variables)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(values)

    prompt_text = prompt_template
    for k, v in values.items():
        prompt_text = prompt_text.replace("{" + k + "}", v)
    return ollama_generate(prompt_text, model=model)


def _safe_parse_json(text: str):
    def _try_load(s: str):
        return json.loads(s)

    if not isinstance(text, str):
        return {"text": str(text)}

    raw = text.strip()
    if not raw:
        return {"text": text}

    # 1) direct JSON
    try:
        return _try_load(raw)
    except Exception:
        pass

    # 2) fenced ```json ...``` or ``` ... ```
    try:
        import re

        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            try:
                return _try_load(candidate)
            except Exception:
                pass
    except Exception:
        pass

    # 3) extract balanced JSON object/array candidates from mixed text.
    # Prefer the "best" candidate (e.g. contains tool_call/expected keys), since
    # models sometimes include JSON examples before the final answer.
    def _extract_balanced_from(s: str, start: int, open_ch: str, close_ch: str) -> str | None:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return s[start : i + 1]
        return None

    def _score_json(obj) -> int:
        if not isinstance(obj, dict):
            return 0
        score = 0
        # tool calls are usually the "real" payload
        if isinstance(obj.get("tool_call"), dict):
            score += 100
        key_weights = {
            "role": 10,
            "ticker": 10,
            "summary": 10,
            "key_points": 8,
            "stance": 12,
            "final_label": 12,
            "consensus_summary": 12,
            "confidence": 10,
            "risk_score": 12,
            "breach_flags": 8,
            "explainers": 8,
            "decision": 12,
            "reason": 10,
            "next_steps": 8,
            "side": 12,
            "entry": 8,
            "stop": 8,
            "target": 8,
            "rationale": 12,
            "created_by": 8,
        }
        for k, w in key_weights.items():
            if k in obj:
                score += w
        # Penalize pure text wrappers
        if set(obj.keys()) == {"text"}:
            score -= 20
        return score

    candidates: list[tuple[object, int, int]] = []
    for idx, ch in enumerate(raw):
        if ch == "{":
            candidate = _extract_balanced_from(raw, idx, "{", "}")
        elif ch == "[":
            candidate = _extract_balanced_from(raw, idx, "[", "]")
        else:
            continue
        if not candidate:
            continue
        try:
            obj = _try_load(candidate)
            candidates.append((obj, _score_json(obj), idx))
        except Exception:
            continue

    if candidates:
        # prefer highest score; break ties by choosing the last occurrence
        best_obj, _, _ = max(candidates, key=lambda x: (x[1], x[2]))
        return best_obj

    return {"text": text}


def _is_nonempty_str(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_schema(obj: dict, required: list[str]) -> list[str]:
    if not isinstance(obj, dict):
        return ["not_a_dict"]
    missing = [k for k in required if k not in obj]
    return missing


def _repair_instruction(missing: list[str], schema_hint: str) -> str:
    missing_txt = ", ".join(missing) if missing else "(unknown)"
    return (
        "\n\nVALIDATION_ERROR: Your previous output was invalid. "
        f"Missing required fields: {missing_txt}. "
        "Return ONLY a single JSON object matching the schema exactly. "
        "No markdown, no prose, no explanations.\n"
        f"SCHEMA_HINT: {schema_hint}\n"
    )


def _extract_tool_call(parsed) -> dict | None:
    """Return a normalized tool_call dict if present, else None.

    Supports either:
      {"tool_call": {"tool": "alpha_fetch", ...}}
    or (less strict model output):
      {"tool": "alpha_fetch", "function": "...", ...}
    """
    if not isinstance(parsed, dict):
        return None

    tc = parsed.get("tool_call")
    if isinstance(tc, dict):
        return tc

    # tolerate top-level tool call fields
    if isinstance(parsed.get("tool"), str) and parsed.get("tool"):
        # Keep this generic so new tools (e.g. rag_search) don't get their
        # parameters dropped during normalization.
        allow = {
            "tool",
            "function",
            "symbol",
            "tickers",
            "params",
            "query",
            "types",
            "stage",
            "source",
            "run_id",
            "days",
            "top_k",
            "include_test",
        }
        out = {k: parsed.get(k) for k in allow if k in parsed}
        out.setdefault("tool", parsed.get("tool"))
        return out

    return None


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _trace_append(trace: list[dict] | None, event: dict) -> None:
    if trace is None:
        return
    trace.append(event)


def _call_alpha_traced(
    *,
    function: str,
    symbol: str | None = None,
    tickers: str | None = None,
    params: dict | None = None,
    trace: list[dict] | None = None,
    stage: str = "",
) -> dict:
    # Normalize common Alpha Vantage quirks.
    if function == "NEWS_SENTIMENT" and not tickers and symbol:
        tickers = symbol
        symbol = None

    ev = {
        "at": _now_iso(),
        "stage": stage,
        "tool": "alpha_fetch",
        "function": function,
        "symbol": symbol,
        "tickers": tickers,
        "params": params or {},
    }

    def _alpha_api_issue(payload: object) -> tuple[str, str] | None:
        if not isinstance(payload, dict):
            return None
        # Alpha Vantage typically returns these keys for throttling / errors.
        for k in ("Error Message", "Note", "Information"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return k, v.strip()
        return None

    # Prefer using cached tool results when available to avoid rate limits.
    # (You can still force fresh calls by clearing the SQLite table.)
    cached_any, cached_age = _alpha_cache_lookup(function=function, symbol=symbol, tickers=tickers, max_age_sec=None)
    cached_fresh, cached_fresh_age = _alpha_cache_lookup(function=function, symbol=symbol, tickers=tickers, max_age_sec=86400)

    if cached_fresh is not None:
        ev["status"] = "cache"
        ev["tool_called"] = False
        ev["cache_hit"] = True
        ev["cache_age_sec"] = cached_fresh_age
        if isinstance(cached_fresh, dict):
            ev["response_keys"] = list(cached_fresh.keys())[:25]
        ev["response_preview"] = _trim_json(cached_fresh, max_chars=900)
        ev["rag_stored"] = True
        _trace_append(trace, ev)
        return cached_fresh  # type: ignore[return-value]

    try:
        _alpha_throttle()
        res = call_alpha(function, symbol=symbol, tickers=tickers, params=params)

        issue = _alpha_api_issue(res)
        if issue is not None:
            ev["status"] = "api_error"
            ev["api_issue"] = {"kind": issue[0], "message": issue[1][:500]}
        else:
            ev["status"] = "ok"

        ev["tool_called"] = True
        ev["cache_hit"] = False

        # On rate-limit responses, fall back to any cached copy (even if older).
        if issue is not None and cached_any is not None:
            ev["fallback_cache"] = True
            ev["cache_age_sec"] = cached_age
            if isinstance(cached_any, dict):
                ev["response_keys"] = list(cached_any.keys())[:25]
            ev["response_preview"] = _trim_json(cached_any, max_chars=900)
            ev["rag_stored"] = True
            _trace_append(trace, ev)
            return cached_any  # type: ignore[return-value]

        # Add a compact response preview to the trace so it's easy to verify.
        if isinstance(res, dict):
            ev["response_keys"] = list(res.keys())[:25]
        ev["response_preview"] = _trim_json(res, max_chars=900)

        # Best-effort: store tool result to RAG DB (one JSON == one row)
        stored = False
        if issue is None:
            stored = _rag_store_json(
                res,
                metadata={
                    "source": "alpha_tool",
                    "tool": "alpha_fetch",
                    "function": function,
                    "symbol": symbol,
                    "tickers": tickers,
                    "params": params or {},
                    "stage": stage,
                    "timestamp": int(time.time()),
                    "type": "tool_result",
                    "alpha_ok": True,
                },
                trace_event=ev,
            )
        ev["rag_stored"] = bool(stored)
        _trace_append(trace, ev)
        return res
    except Exception as e:
        ev["status"] = "error"
        ev["error_type"] = type(e).__name__
        ev["error"] = str(e)
        _trace_append(trace, ev)
        raise


def _trim_json(obj: object, max_chars: int = 1800) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) > max_chars:
        return s[:max_chars] + "..."
    return s


def _ensure_json_schema_with_auto_tools(
    *,
    stage: str,
    base_input: str,
    out_text: str,
    run_once,
    required: list[str],
    schema_hint: str,
    auto_tool_calls: list[dict] | None = None,
    trace: list[dict] | None = None,
) -> tuple[dict, str]:
    parsed = _safe_parse_json(out_text)
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}

    missing = _validate_schema(parsed, required)
    if not missing:
        return parsed, out_text

    # Retry 1: explicit validation error + schema hint
    retry_input = base_input + _repair_instruction(missing, schema_hint)
    _trace_append(trace, {"at": _now_iso(), "stage": stage, "type": "retry", "reason": "schema_missing", "missing": missing, "pass": 1})
    out2 = run_once(retry_input)
    parsed2 = _safe_parse_json(out2)
    if isinstance(parsed2, dict):
        parsed = parsed2
    missing = _validate_schema(parsed, required)
    if not missing:
        return parsed, out2

    # Retry 2: auto tool calls + explicit validation error
    tool_blob: dict = {}
    for tc in auto_tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        sym = tc.get("symbol")
        ticks = tc.get("tickers")
        params = tc.get("params") if isinstance(tc.get("params"), dict) else None
        if not fn:
            continue
        try:
            tool_blob[fn] = _call_alpha_traced(function=fn, symbol=sym, tickers=ticks, params=params, trace=trace, stage=stage)
        except Exception as e:
            tool_blob[fn] = {"error": str(e)}

    tool_ctx = "\n\nAUTO_TOOL_CONTEXT:\n" + _trim_json(tool_blob)
    retry_input2 = base_input + tool_ctx + _repair_instruction(missing, schema_hint)
    _trace_append(trace, {"at": _now_iso(), "stage": stage, "type": "retry", "reason": "schema_missing_after_tools", "missing": missing, "pass": 2})
    out3 = run_once(retry_input2)
    parsed3 = _safe_parse_json(out3)
    if isinstance(parsed3, dict):
        parsed = parsed3
    return parsed, out3


def market_chain(raw_market_text: str, ticker: str = "TSLA", model: str = None, trace: list[dict] | None = None) -> dict:
    # run once, then check for tool_call instructions in returned JSON; allow up to 2 iterations
    def _run_once(input_text: str) -> str:
        if USE_LANGCHAIN:
            llm = OllamaLLM(model=model)
            prompt = PromptTemplate(template=MARKET_PROMPT, input_variables=["input"]) 
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run({"input": input_text})
        else:
            prompt_text = MARKET_PROMPT.replace('{input}', input_text)
            return ollama_generate(prompt_text, model=model)

    out = _run_once(raw_market_text)
    # check for tool_call in model output
    for _ in range(2):
        try:
            import json as _json
            j = _safe_parse_json(out)
            tc = _extract_tool_call(j)
            if isinstance(tc, dict):
                if tc.get("tool") == "alpha_fetch":
                    func = tc.get("function")
                    sym = tc.get("symbol")
                    t = tc.get("tickers")
                    params = tc.get("params") if isinstance(tc.get("params"), dict) else None
                    tool_res = _call_alpha_traced(function=func, symbol=sym, tickers=t, params=params, trace=trace, stage="MARKET_ANALYST")
                    # append tool result to input and rerun
                    raw_market_text = raw_market_text + "\n\nTOOL_RESULT:\n" + _json.dumps(tool_res)
                    out = _run_once(raw_market_text)
                    continue
        except Exception:
            pass
        break
    parsed, _ = _ensure_json_schema_with_auto_tools(
        stage="MARKET_ANALYST",
        base_input=raw_market_text,
        out_text=out,
        run_once=_run_once,
        required=["summary", "key_points", "confidence", "recommendation_hint"],
        schema_hint='{ "role":"MARKET_ANALYST","ticker":"","timestamp":"","summary":"","key_points":[],"confidence":0.0,"recommendation_hint":"" }',
        auto_tool_calls=[
            {"function": "GLOBAL_QUOTE", "symbol": ticker},
            {"function": "TIME_SERIES_DAILY", "symbol": ticker, "params": {"outputsize": "compact"}},
        ],
        trace=trace,
    )
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    parsed.update({"role": "MARKET_ANALYST", "ticker": ticker, "timestamp": datetime.utcnow().isoformat()})
    return parsed


def social_chain(raw_social_text: str, ticker: str = "TSLA", model: str = None, trace: list[dict] | None = None) -> dict:
    def _run_once(input_text: str) -> str:
        if USE_LANGCHAIN:
            llm = OllamaLLM(model=model)
            prompt = PromptTemplate(template=ANALYST_PROMPT, input_variables=["input"]) 
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run({"input": input_text})
        else:
            prompt_text = ANALYST_PROMPT.replace('{input}', input_text)
            return ollama_generate(prompt_text, model=model)

    out = _run_once(raw_social_text)
    for _ in range(2):
        try:
            import json as _json
            j = _safe_parse_json(out)
            tc = _extract_tool_call(j)
            if isinstance(tc, dict):
                if tc.get("tool") == "alpha_fetch":
                    func = tc.get("function")
                    sym = tc.get("symbol")
                    t = tc.get("tickers")
                    params = tc.get("params") if isinstance(tc.get("params"), dict) else None
                    tool_res = _call_alpha_traced(function=func, symbol=sym, tickers=t, params=params, trace=trace, stage="SOCIAL_ANALYST")
                    raw_social_text = raw_social_text + "\n\nTOOL_RESULT:\n" + _json.dumps(tool_res)
                    out = _run_once(raw_social_text)
                    continue
        except Exception:
            pass
        break
    parsed, _ = _ensure_json_schema_with_auto_tools(
        stage="SOCIAL_ANALYST",
        base_input=raw_social_text,
        out_text=out,
        run_once=_run_once,
        required=["summary", "key_points", "confidence", "recommendation_hint"],
        schema_hint='{ "role":"SOCIAL_ANALYST","ticker":"","timestamp":"","summary":"","key_points":[],"confidence":0.0,"recommendation_hint":"" }',
        auto_tool_calls=[{"function": "NEWS_SENTIMENT", "tickers": ticker}],
        trace=trace,
    )
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    parsed.update({"role": "SOCIAL_ANALYST", "ticker": ticker, "timestamp": datetime.utcnow().isoformat()})
    return parsed


def news_chain(raw_news_text: str, ticker: str = "TSLA", model: str = None, trace: list[dict] | None = None) -> dict:
    def _run_once(input_text: str) -> str:
        if USE_LANGCHAIN:
            llm = OllamaLLM(model=model)
            prompt = PromptTemplate(template=ANALYST_PROMPT, input_variables=["input"]) 
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run({"input": input_text})
        else:
            prompt_text = ANALYST_PROMPT.replace('{input}', input_text)
            return ollama_generate(prompt_text, model=model)

    out = _run_once(raw_news_text)
    for _ in range(2):
        try:
            import json as _json
            j = _safe_parse_json(out)
            tc = _extract_tool_call(j)
            if isinstance(tc, dict):
                if tc.get("tool") == "alpha_fetch":
                    func = tc.get("function")
                    sym = tc.get("symbol")
                    t = tc.get("tickers")
                    params = tc.get("params") if isinstance(tc.get("params"), dict) else None
                    tool_res = _call_alpha_traced(function=func, symbol=sym, tickers=t, params=params, trace=trace, stage="NEWS_ANALYST")
                    raw_news_text = raw_news_text + "\n\nTOOL_RESULT:\n" + _json.dumps(tool_res)
                    out = _run_once(raw_news_text)
                    continue
        except Exception:
            pass
        break
    parsed, _ = _ensure_json_schema_with_auto_tools(
        stage="NEWS_ANALYST",
        base_input=raw_news_text,
        out_text=out,
        run_once=_run_once,
        required=["summary", "key_points", "confidence", "recommendation_hint"],
        schema_hint='{ "role":"NEWS_ANALYST","ticker":"","timestamp":"","summary":"","key_points":[],"confidence":0.0,"recommendation_hint":"" }',
        auto_tool_calls=[{"function": "NEWS_SENTIMENT", "tickers": ticker}],
        trace=trace,
    )
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    parsed.update({"role": "NEWS_ANALYST", "ticker": ticker, "timestamp": datetime.utcnow().isoformat()})
    return parsed


def fundamentals_chain(raw_fund_text: str, ticker: str = "TSLA", model: str = None, trace: list[dict] | None = None) -> dict:
    def _run_once(input_text: str) -> str:
        if USE_LANGCHAIN:
            llm = OllamaLLM(model=model)
            prompt = PromptTemplate(template=ANALYST_PROMPT, input_variables=["input"]) 
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run({"input": input_text})
        else:
            prompt_text = ANALYST_PROMPT.replace('{input}', input_text)
            return ollama_generate(prompt_text, model=model)

    out = _run_once(raw_fund_text)
    for _ in range(2):
        try:
            import json as _json
            j = _safe_parse_json(out)
            tc = _extract_tool_call(j)
            if isinstance(tc, dict):
                if tc.get("tool") == "alpha_fetch":
                    func = tc.get("function")
                    sym = tc.get("symbol")
                    t = tc.get("tickers")
                    params = tc.get("params") if isinstance(tc.get("params"), dict) else None
                    tool_res = _call_alpha_traced(function=func, symbol=sym, tickers=t, params=params, trace=trace, stage="FUNDAMENTALS_ANALYST")
                    raw_fund_text = raw_fund_text + "\n\nTOOL_RESULT:\n" + _json.dumps(tool_res)
                    out = _run_once(raw_fund_text)
                    continue
        except Exception:
            pass
        break
    parsed, _ = _ensure_json_schema_with_auto_tools(
        stage="FUNDAMENTALS_ANALYST",
        base_input=raw_fund_text,
        out_text=out,
        run_once=_run_once,
        required=["summary", "key_points", "confidence", "recommendation_hint"],
        schema_hint='{ "role":"FUNDAMENTALS_ANALYST","ticker":"","timestamp":"","summary":"","key_points":[],"confidence":0.0,"recommendation_hint":"" }',
        auto_tool_calls=[{"function": "OVERVIEW", "symbol": ticker}],
        trace=trace,
    )
    if not isinstance(parsed, dict):
        parsed = {"value": parsed}
    parsed.update({"role": "FUNDAMENTALS_ANALYST", "ticker": ticker, "timestamp": datetime.utcnow().isoformat()})
    return parsed


def researcher_chain(analyst_reports: dict, stance: str = "BULL", model: str = None, trace: list[dict] | None = None) -> dict:
    # merge short summaries
    merged = "\n\n".join([f"{k}: {v.get('summary','')[:1000]}" for k, v in analyst_reports.items()])
    def _run_once(input_text: str) -> str:
        if USE_LANGCHAIN:
            llm = OllamaLLM(model=model)
            prompt = PromptTemplate(template=RESEARCHER_PROMPT, input_variables=["input", "stance"]) 
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run({"input": input_text, "stance": stance})
        else:
            prompt_text = RESEARCHER_PROMPT.replace('{input}', input_text).replace('{stance}', stance)
            return ollama_generate(prompt_text, model=model)

    out = _run_once(merged)
    for _ in range(2):
        try:
            import json as _json
            j = _safe_parse_json(out)
            tc = _extract_tool_call(j)
            if isinstance(tc, dict) and tc.get("tool") == "alpha_fetch":
                func = tc.get("function")
                sym = tc.get("symbol")
                t = tc.get("tickers")
                params = tc.get("params") if isinstance(tc.get("params"), dict) else None
                try:
                    tool_res = _call_alpha_traced(function=func, symbol=sym, tickers=t, params=params, trace=trace, stage=f"{stance}_RESEARCHER")
                    merged = merged + "\n\nTOOL_RESULT:\n" + _json.dumps(tool_res)
                except Exception as e:
                    merged = merged + "\n\nTOOL_ERROR:\n" + str(e)
                out = _run_once(merged)
                continue
        except Exception:
            pass
        break
    parsed, _ = _ensure_json_schema_with_auto_tools(
        stage=f"{stance}_RESEARCHER",
        base_input=merged,
        out_text=out,
        run_once=_run_once,
        required=["final_label", "consensus_summary", "evidence", "counterarguments", "confidence"],
        schema_hint='{ "stance":"BULL|BEAR","final_label":"BULLISH|BEARISH|NEUTRAL","consensus_summary":"","evidence":[],"counterarguments":[],"confidence":0.0 }',
        auto_tool_calls=[{"function": "NEWS_SENTIMENT", "tickers": analyst_reports.get("market", {}).get("ticker") or "TSLA"}],
        trace=trace,
    )
    parsed["stance"] = stance
    return parsed


def researcher_debate_turn(
    analyst_reports: dict,
    *,
    stance: str,
    opponent_last: str,
    round_n: int,
    ticker: str,
    model: str | None = None,
    trace: list[dict] | None = None,
) -> dict:
    """A debate turn for BULL/BEAR researcher with schema enforcement.

    This reuses the same RESEARCHER_PROMPT but injects opponent's previous summary
    and the round number into the input.
    """
    merged = "\n\n".join([f"{k}: {v.get('summary','')[:1000]}" for k, v in (analyst_reports or {}).items()])
    debate_ctx = (
        f"\n\nDEBATE_ROUND: {round_n}\n"
        f"YOUR_ROLE: {stance}_RESEARCHER\n"
        "TASK: Respond to the opponent's last points, add new evidence/counterarguments, "
        "and update your consensus_summary and final_label if needed.\n"
        f"OPPONENT_LAST:\n{opponent_last or '(none)'}\n"
    )
    base_input = merged + debate_ctx

    def _run_once(input_text: str) -> str:
        if USE_LANGCHAIN:
            llm = OllamaLLM(model=model)
            prompt = PromptTemplate(template=RESEARCHER_PROMPT, input_variables=["input", "stance"]) 
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run({"input": input_text, "stance": stance})
        prompt_text = RESEARCHER_PROMPT.replace('{input}', input_text).replace('{stance}', stance)
        return ollama_generate(prompt_text, model=model)

    out = _run_once(base_input)
    for _ in range(2):
        try:
            import json as _json

            j = _safe_parse_json(out)
            tc = _extract_tool_call(j)
            if isinstance(tc, dict) and tc.get("tool") == "alpha_fetch":
                func = tc.get("function")
                sym = tc.get("symbol")
                t = tc.get("tickers")
                params = tc.get("params") if isinstance(tc.get("params"), dict) else None
                try:
                    tool_res = _call_alpha_traced(function=func, symbol=sym, tickers=t, params=params, trace=trace, stage=f"{stance}_DEBATE_R{round_n}")
                    base_input = base_input + "\n\nTOOL_RESULT:\n" + _json.dumps(tool_res)
                except Exception as e:
                    base_input = base_input + "\n\nTOOL_ERROR:\n" + str(e)
                out = _run_once(base_input)
                continue
        except Exception:
            pass
        break

    parsed, _ = _ensure_json_schema_with_auto_tools(
        stage=f"{stance}_DEBATE_R{round_n}",
        base_input=base_input,
        out_text=out,
        run_once=_run_once,
        required=["final_label", "consensus_summary", "evidence", "counterarguments", "confidence"],
        schema_hint='{ "stance":"BULL|BEAR","final_label":"BULLISH|BEARISH|NEUTRAL","consensus_summary":"","evidence":[],"counterarguments":[],"confidence":0.0 }',
        auto_tool_calls=[{"function": "NEWS_SENTIMENT", "tickers": ticker}],
        trace=trace,
    )
    parsed["stance"] = stance
    return parsed


def risk_chain(discussion_summary: str, model: str = None, trace: list[dict] | None = None) -> dict:
    def _run_once(input_text: str) -> str:
        if USE_LANGCHAIN:
            llm = OllamaLLM(model=model)
            prompt = PromptTemplate(template=RISK_ANALYST_PROMPT, input_variables=["input"]) 
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run({"input": input_text})
        else:
            prompt_text = RISK_ANALYST_PROMPT.replace('{input}', input_text)
            return ollama_generate(prompt_text, model=model)

    out = _run_once(discussion_summary)
    for _ in range(2):
        try:
            import json as _json
            j = _safe_parse_json(out)
            tc = _extract_tool_call(j)
            if isinstance(tc, dict) and tc.get("tool") == "alpha_fetch":
                func = tc.get("function")
                sym = tc.get("symbol")
                t = tc.get("tickers")
                params = tc.get("params") if isinstance(tc.get("params"), dict) else None
                try:
                    tool_res = _call_alpha_traced(function=func, symbol=sym, tickers=t, params=params, trace=trace, stage="RISK_ANALYST")
                    discussion_summary = discussion_summary + "\n\nTOOL_RESULT:\n" + _json.dumps(tool_res)
                except Exception as e:
                    discussion_summary = discussion_summary + "\n\nTOOL_ERROR:\n" + str(e)
                out = _run_once(discussion_summary)
                continue
        except Exception:
            pass
        break
    parsed, _ = _ensure_json_schema_with_auto_tools(
        stage="RISK_ANALYST",
        base_input=discussion_summary,
        out_text=out,
        run_once=_run_once,
        required=["risk_score", "breach_flags", "explainers"],
        schema_hint='{ "risk_score":0, "breach_flags":[], "explainers":[] }',
        auto_tool_calls=[],
        trace=trace,
    )

    if not _is_number(parsed.get("risk_score")):
        parsed["risk_score"] = 60
    if not isinstance(parsed.get("breach_flags"), list):
        parsed["breach_flags"] = []
    if not isinstance(parsed.get("explainers"), list):
        parsed["explainers"] = []
    return parsed


def risk_manager_chain(aggregated_info: str, model: str = None, trace: list[dict] | None = None) -> dict:
    # run once, then check for tool_call instructions in returned JSON; allow up to 2 iterations
    def _run_once(input_text: str) -> str:
        if USE_LANGCHAIN:
            llm = OllamaLLM(model=model)
            prompt = PromptTemplate(template=RISK_MANAGER_PROMPT, input_variables=["input"]) 
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run({"input": input_text})
        else:
            prompt_text = RISK_MANAGER_PROMPT.replace('{input}', input_text)
            return ollama_generate(prompt_text, model=model)

    out = _run_once(aggregated_info)
    # check for tool_call in model output
    for _ in range(2):
        try:
            import json as _json
            j = _safe_parse_json(out)
            tc = _extract_tool_call(j)
            if isinstance(tc, dict) and tc.get("tool") == "alpha_fetch":
                func = tc.get("function")
                sym = tc.get("symbol")
                t = tc.get("tickers")
                params = tc.get("params") if isinstance(tc.get("params"), dict) else None
                try:
                    tool_res = _call_alpha_traced(function=func, symbol=sym, tickers=t, params=params, trace=trace, stage="RISK_MANAGER")
                    aggregated_info = aggregated_info + "\n\nTOOL_RESULT:\n" + _json.dumps(tool_res)
                except Exception as e:
                    aggregated_info = aggregated_info + "\n\nTOOL_ERROR:\n" + str(e)
                out = _run_once(aggregated_info)
                continue
        except Exception:
            pass
        break
    parsed, _ = _ensure_json_schema_with_auto_tools(
        stage="RISK_MANAGER",
        base_input=aggregated_info,
        out_text=out,
        run_once=_run_once,
        required=["decision", "reason", "next_steps"],
        schema_hint='{ "decision":"approve|reject|require_manual", "reason":"", "next_steps":"" }',
        auto_tool_calls=[{"function": "NEWS_SENTIMENT", "tickers": "TSLA"}],
        trace=trace,
    )

    parsed.update({"role": "RISK_MANAGER", "timestamp": datetime.utcnow().isoformat()})
    return parsed


def trader_chain(manager_decision: dict, discussion_summary: str, model: str = None, trace: list[dict] | None = None) -> dict:
    payload = {
        "manager_decision": manager_decision,
        "discussion_summary": discussion_summary,
    }
    raw_input = json.dumps(payload, ensure_ascii=False)

    def _run_once(input_text: str) -> str:
        if USE_LANGCHAIN:
            llm = OllamaLLM(model=model)
            prompt = PromptTemplate(template=TRADER_PROMPT, input_variables=["input"]) 
            chain = LLMChain(llm=llm, prompt=prompt)
            return chain.run({"input": input_text})
        else:
            prompt_text = TRADER_PROMPT.replace('{input}', input_text)
            return ollama_generate(prompt_text, model=model)

    out = _run_once(raw_input)
    for _ in range(2):
        try:
            import json as _json
            j = _safe_parse_json(out)
            tc = _extract_tool_call(j)
            if isinstance(tc, dict) and tc.get("tool") == "alpha_fetch":
                func = tc.get("function")
                sym = tc.get("symbol")
                t = tc.get("tickers")
                params = tc.get("params") if isinstance(tc.get("params"), dict) else None
                try:
                    tool_res = _call_alpha_traced(function=func, symbol=sym, tickers=t, params=params, trace=trace, stage="TRADER")
                    raw_input = raw_input + "\n\nTOOL_RESULT:\n" + _json.dumps(tool_res)
                except Exception as e:
                    raw_input = raw_input + "\n\nTOOL_ERROR:\n" + str(e)
                out = _run_once(raw_input)
                continue
        except Exception:
            pass
        break

    parsed, _ = _ensure_json_schema_with_auto_tools(
        stage="TRADER",
        base_input=raw_input,
        out_text=out,
        run_once=_run_once,
        required=["ticker", "side", "size", "entry", "stop", "target", "rationale", "confidence"],
        schema_hint='{ "ticker":"", "side":"BUY|SELL|NO_TRADE", "size":0.0, "entry":0.0, "stop":0.0, "target":0.0, "rationale":"", "confidence":0.0, "created_by":"TRADER" }',
        auto_tool_calls=[{"function": "GLOBAL_QUOTE", "symbol": "TSLA"}],
        trace=trace,
    )

    parsed["created_by"] = "TRADER"
    parsed.setdefault("timestamp", datetime.utcnow().isoformat())
    if not _is_nonempty_str(parsed.get("rationale")):
        parsed["rationale"] = "No actionable trade rationale produced."
    return parsed


def run_langchain_flow(template_path="OUTPUT_TEMPLATE.TXT", model: str = None, ticker: str = "TSLA") -> dict:
    # Backward-compatible entrypoint name.
    # Now prefers LangGraph for orchestration; falls back to legacy sequential flow
    # if LangGraph isn't installed.
    if USE_LANGGRAPH:
        return run_langgraph_flow(template_path=template_path, model=model, ticker=ticker)
    return _run_legacy_sequential_flow(template_path=template_path, model=model, ticker=ticker)


def _summary_fallback(obj: dict) -> str:
    if not isinstance(obj, dict):
        return str(obj)
    text = obj.get("consensus_summary") or obj.get("summary") or obj.get("text") or json.dumps(obj, ensure_ascii=False)
    if isinstance(text, str) and len(text) > 2500:
        return text[:2500] + "..."
    return text


def _run_legacy_sequential_flow(template_path: str, model: str | None, ticker: str) -> dict:
    """Legacy non-LangGraph orchestration kept as a safe fallback."""
    import json as _json

    with open(template_path, "r", encoding="utf-8") as f:
        data = _json.load(f)

    stages = data.get("stages", {})
    market_report = stages.get("MARKET_ANALYST", {}).get("market_report", {}).get("preview", "")
    social_report = stages.get("SOCIAL_ANALYST", {}).get("sentiment_report", {}).get("preview", "")
    news_report = stages.get("NEWS_ANALYST", {}).get("news_report", {}).get("preview", "")
    fund_report = stages.get("FUNDAMENTALS_ANALYST", {}).get("fundamentals_report", {}).get("preview", "")

    trace: list[dict] = []
    baseline: dict[str, Any] = {}
    for fn, kwargs in [
        ("GLOBAL_QUOTE", {"symbol": ticker}),
        ("OVERVIEW", {"symbol": ticker}),
        ("NEWS_SENTIMENT", {"tickers": ticker}),
    ]:
        try:
            baseline[fn] = _call_alpha_traced(function=fn, trace=trace, stage="PREFETCH", **kwargs)
        except Exception as e:
            baseline[fn] = {"error": str(e)}

    baseline_ctx = "\n\nBASELINE_TOOL_CONTEXT:\n" + _trim_json(baseline)

    a_market = market_chain(market_report + baseline_ctx, ticker=ticker, model=model, trace=trace)
    a_social = social_chain(social_report + baseline_ctx, ticker=ticker, model=model, trace=trace)
    a_news = news_chain(news_report + baseline_ctx, ticker=ticker, model=model, trace=trace)
    a_fund = fundamentals_chain(fund_report + baseline_ctx, ticker=ticker, model=model, trace=trace)

    analyst_reports = {"market": a_market, "social": a_social, "news": a_news, "fundamentals": a_fund}
    bull = researcher_chain(analyst_reports, stance="BULL", model=model, trace=trace)
    bear = researcher_chain(analyst_reports, stance="BEAR", model=model, trace=trace)
    discussion_summary = _summary_fallback(bull) + "\n\n" + _summary_fallback(bear)

    risk = risk_chain(discussion_summary, model=model, trace=trace)
    aggregated_info = json.dumps(
        {"discussion_summary": discussion_summary, "bull": bull, "bear": bear, "risk": risk},
        ensure_ascii=False,
    )
    manager_decision = risk_manager_chain(aggregated_info, model=model, trace=trace)
    trader = trader_chain(manager_decision, discussion_summary, model=model, trace=trace)

    final = {
        "analysts": analyst_reports,
        "researchers": {"bull": bull, "bear": bear, "discussion": discussion_summary},
        "risk": risk,
        "manager_decision": manager_decision,
        "trader_proposal": trader,
        "trace": trace,
    }

    with open("flow_output_langchain.json", "w", encoding="utf-8") as fout:
        _json.dump(final, fout, ensure_ascii=False, indent=2)
    return final


def run_langgraph_flow(template_path: str = "OUTPUT_TEMPLATE.TXT", model: str | None = None, ticker: str = "TSLA") -> dict:
    """LangGraph-orchestrated version of the flow (Flow #2).

    In this version, the *tool_call loop* and *schema retry* are implemented at the graph layer
    via conditional edges (shared work pipeline). The old per-stage Python loops remain for
    the legacy sequential flow.
    """
    if not USE_LANGGRAPH:
        raise RuntimeError("langgraph is not installed. Add 'langgraph' to requirements.txt and reinstall.")

    def _prompt_for_kind(kind: str) -> tuple[str, list[str]]:
        if kind == "MARKET":
            return MARKET_PROMPT, ["input"]
        if kind == "ANALYST":
            return ANALYST_PROMPT, ["input"]
        if kind == "RESEARCHER":
            return RESEARCHER_PROMPT, ["input", "stance"]
        if kind == "RISK_ANALYST":
            return RISK_ANALYST_PROMPT, ["input"]
        if kind == "RISK_MANAGER":
            return RISK_MANAGER_PROMPT, ["input"]
        if kind == "TRADER":
            return TRADER_PROMPT, ["input"]
        raise ValueError(f"Unknown work kind: {kind}")

    def node_load_template(state: FlowState) -> FlowState:
        import json as _json

        with open(template_path, "r", encoding="utf-8") as f:
            data = _json.load(f)

        stages = data.get("stages", {})
        return {
            **state,
            "ticker": ticker,
            "model": model,
            "market_report": stages.get("MARKET_ANALYST", {}).get("market_report", {}).get("preview", ""),
            "social_report": stages.get("SOCIAL_ANALYST", {}).get("sentiment_report", {}).get("preview", ""),
            "news_report": stages.get("NEWS_ANALYST", {}).get("news_report", {}).get("preview", ""),
            "fund_report": stages.get("FUNDAMENTALS_ANALYST", {}).get("fundamentals_report", {}).get("preview", ""),
            "trace": [],
        }

    def node_prefetch(state: FlowState) -> FlowState:
        trace = state.get("trace") or []
        baseline: dict[str, Any] = {}
        for fn, kwargs in [
            ("GLOBAL_QUOTE", {"symbol": ticker}),
            ("OVERVIEW", {"symbol": ticker}),
            ("NEWS_SENTIMENT", {"tickers": ticker}),
        ]:
            try:
                baseline[fn] = _call_alpha_traced(function=fn, trace=trace, stage="PREFETCH", **kwargs)
            except Exception as e:
                baseline[fn] = {"error": str(e)}

        baseline_ctx = "\n\nBASELINE_TOOL_CONTEXT:\n" + _trim_json(baseline)
        return {**state, "trace": trace, "baseline": baseline, "baseline_ctx": baseline_ctx}

    # -------- Shared work pipeline nodes --------
    def node_work_llm(state: FlowState) -> FlowState:
        prompt_template, input_vars = _prompt_for_kind(state.get("work_kind") or "")
        values: dict[str, str] = {"input": state.get("work_input") or ""}
        if "stance" in input_vars:
            values["stance"] = str(state.get("work_stance") or "")
        out = _run_llm_prompt_once(prompt_template, values=values, model=model, input_variables=input_vars)
        return {**state, "work_output": out}

    def node_work_parse(state: FlowState) -> FlowState:
        parsed = _safe_parse_json(state.get("work_output") or "")
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        return {**state, "work_parsed": parsed}

    def route_tool_or_validate(state: FlowState) -> str:
        tc = _extract_tool_call(state.get("work_parsed") or {})
        tool_iter = int(state.get("work_tool_iter") or 0)
        if isinstance(tc, dict) and tc.get("tool") in {"alpha_fetch", "rag_search"} and tool_iter < 2:
            return "work_tool"
        return "work_validate"

    def node_work_tool(state: FlowState) -> FlowState:
        parsed = state.get("work_parsed") or {}
        tc = _extract_tool_call(parsed) or {}
        tool_name = tc.get("tool")

        tool_res: Any
        if tool_name == "rag_search":
            if rag_search is None:
                tool_res = {"error": "rag_search tool unavailable"}
            else:
                try:
                    tool_res = rag_search(
                        str(tc.get("query") or ""),
                        symbol=tc.get("symbol"),
                        types=tc.get("types") if isinstance(tc.get("types"), list) else None,
                        stage=tc.get("stage"),
                        source=tc.get("source"),
                        run_id=tc.get("run_id"),
                        days=tc.get("days"),
                        top_k=int(tc.get("top_k") or 5),
                        include_test=tc.get("include_test"),
                    )
                except Exception as e:
                    tool_res = {"error": str(e)}
        else:
            fn = tc.get("function")
            sym = tc.get("symbol")
            ticks = tc.get("tickers")
            params = tc.get("params") if isinstance(tc.get("params"), dict) else None
            try:
                tool_res = _call_alpha_traced(function=fn, symbol=sym, tickers=ticks, params=params, trace=state.get("trace"), stage=state.get("work_stage") or "")
            except Exception as e:
                tool_res = {"error": str(e)}

        appended = (state.get("work_input") or "") + "\n\nTOOL_RESULT:\n" + json.dumps(tool_res, ensure_ascii=False)
        return {**state, "work_input": appended, "work_tool_iter": int(state.get("work_tool_iter") or 0) + 1}

    def node_work_validate(state: FlowState) -> FlowState:
        required = state.get("work_required") or []
        parsed = state.get("work_parsed") or {}
        missing = _validate_schema(parsed, required)
        return {**state, "work_missing": missing}

    def route_after_validate(state: FlowState) -> str:
        missing = state.get("work_missing") or []
        if not missing:
            return "work_commit"
        schema_pass = int(state.get("work_schema_pass") or 0)
        if schema_pass == 0:
            return "work_repair1"
        if schema_pass == 1:
            return "work_auto_tools"
        return "work_commit"

    def node_work_repair1(state: FlowState) -> FlowState:
        missing = state.get("work_missing") or []
        hint = state.get("work_schema_hint") or "{}"
        _trace_append(state.get("trace"), {"at": _now_iso(), "stage": state.get("work_stage") or "", "type": "retry", "reason": "schema_missing", "missing": missing, "pass": 1})
        return {
            **state,
            "work_input": (state.get("work_input") or "") + _repair_instruction(missing, hint),
            "work_schema_pass": 1,
        }

    def node_work_auto_tools(state: FlowState) -> FlowState:
        missing = state.get("work_missing") or []
        hint = state.get("work_schema_hint") or "{}"

        tool_blob: dict[str, Any] = {}
        for tc in state.get("work_auto_tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            sym = tc.get("symbol")
            ticks = tc.get("tickers")
            params = tc.get("params") if isinstance(tc.get("params"), dict) else None
            if not fn:
                continue
            try:
                tool_blob[fn] = _call_alpha_traced(function=fn, symbol=sym, tickers=ticks, params=params, trace=state.get("trace"), stage=state.get("work_stage") or "")
            except Exception as e:
                tool_blob[fn] = {"error": str(e)}

        tool_ctx = "\n\nAUTO_TOOL_CONTEXT:\n" + _trim_json(tool_blob)
        _trace_append(state.get("trace"), {"at": _now_iso(), "stage": state.get("work_stage") or "", "type": "retry", "reason": "schema_missing_after_tools", "missing": missing, "pass": 2})
        return {
            **state,
            "work_input": (state.get("work_input") or "") + tool_ctx + _repair_instruction(missing, hint),
            "work_schema_pass": 2,
        }

    def node_work_commit(state: FlowState) -> FlowState:
        stage = state.get("work_stage") or ""
        parsed = state.get("work_parsed")
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}

        # Best-effort: store stage output into RAG DB (one JSON == one row)
        _rag_store_json(
            parsed,
            metadata={
                "source": "flow_stage",
                "symbol": ticker,
                "stage": stage,
                "timestamp": int(time.time()),
                "type": "stage_output",
            },
        )

        # Analysts
        if stage in {"MARKET_ANALYST", "SOCIAL_ANALYST", "NEWS_ANALYST", "FUNDAMENTALS_ANALYST"}:
            parsed.update({"role": stage, "ticker": ticker, "timestamp": datetime.utcnow().isoformat()})
            analysts = state.get("analysts") or {}
            mapping = {"MARKET_ANALYST": "market", "SOCIAL_ANALYST": "social", "NEWS_ANALYST": "news", "FUNDAMENTALS_ANALYST": "fundamentals"}
            analysts[mapping[stage]] = parsed
            return {**state, "analysts": analysts}

        # Debate rounds
        if stage.startswith("BULL_DEBATE_R"):
            parsed["stance"] = "BULL"
            bull_rounds = list(state.get("bull_rounds") or [])
            bull_rounds.append(parsed)
            return {**state, "bull": parsed, "bull_last": _summary_fallback(parsed), "bull_rounds": bull_rounds}

        if stage.startswith("BEAR_DEBATE_R"):
            parsed["stance"] = "BEAR"
            bear_rounds = list(state.get("bear_rounds") or [])
            bear_rounds.append(parsed)
            return {**state, "bear": parsed, "bear_last": _summary_fallback(parsed), "bear_rounds": bear_rounds}

        # Risk
        if stage == "RISK_ANALYST":
            if not _is_number(parsed.get("risk_score")):
                parsed["risk_score"] = 60
            if not isinstance(parsed.get("breach_flags"), list):
                parsed["breach_flags"] = []
            if not isinstance(parsed.get("explainers"), list):
                parsed["explainers"] = []
            return {**state, "risk": parsed}

        # Manager
        if stage == "RISK_MANAGER":
            parsed.update({"role": "RISK_MANAGER", "timestamp": datetime.utcnow().isoformat()})
            return {**state, "manager_decision": parsed}

        # Trader
        if stage == "TRADER":
            parsed["created_by"] = "TRADER"
            parsed.setdefault("timestamp", datetime.utcnow().isoformat())
            if not _is_nonempty_str(parsed.get("rationale")):
                parsed["rationale"] = "No actionable trade rationale produced."
            return {**state, "trader_proposal": parsed}

        return state

    def route_next_from_commit(state: FlowState) -> str:
        return state.get("work_next") or "finalize"

    # -------- Prepare nodes (configure work_*) --------
    def _reset_work(state: FlowState) -> dict:
        return {"work_schema_pass": 0, "work_tool_iter": 0, "work_missing": [], "work_output": "", "work_parsed": {}}

    def node_prepare_market(state: FlowState) -> FlowState:
        return {
            **state,
            **_reset_work(state),
            "work_stage": "MARKET_ANALYST",
            "work_kind": "MARKET",
            "work_input": (state.get("market_report") or "") + (state.get("baseline_ctx") or ""),
            "work_required": ["summary", "key_points", "confidence", "recommendation_hint"],
            "work_schema_hint": '{ "role":"MARKET_ANALYST","ticker":"","timestamp":"","summary":"","key_points":[],"confidence":0.0,"recommendation_hint":"" }',
            "work_auto_tool_calls": [
                {"function": "GLOBAL_QUOTE", "symbol": ticker},
                {"function": "TIME_SERIES_DAILY", "symbol": ticker, "params": {"outputsize": "compact"}},
            ],
            "work_next": "prepare_social",
        }

    def node_prepare_social(state: FlowState) -> FlowState:
        return {
            **state,
            **_reset_work(state),
            "work_stage": "SOCIAL_ANALYST",
            "work_kind": "ANALYST",
            "work_input": (state.get("social_report") or "") + (state.get("baseline_ctx") or ""),
            "work_required": ["summary", "key_points", "confidence", "recommendation_hint"],
            "work_schema_hint": '{ "role":"SOCIAL_ANALYST","ticker":"","timestamp":"","summary":"","key_points":[],"confidence":0.0,"recommendation_hint":"" }',
            "work_auto_tool_calls": [{"function": "NEWS_SENTIMENT", "tickers": ticker}],
            "work_next": "prepare_news",
        }

    def node_prepare_news(state: FlowState) -> FlowState:
        return {
            **state,
            **_reset_work(state),
            "work_stage": "NEWS_ANALYST",
            "work_kind": "ANALYST",
            "work_input": (state.get("news_report") or "") + (state.get("baseline_ctx") or ""),
            "work_required": ["summary", "key_points", "confidence", "recommendation_hint"],
            "work_schema_hint": '{ "role":"NEWS_ANALYST","ticker":"","timestamp":"","summary":"","key_points":[],"confidence":0.0,"recommendation_hint":"" }',
            "work_auto_tool_calls": [{"function": "NEWS_SENTIMENT", "tickers": ticker}],
            "work_next": "prepare_fundamentals",
        }

    def node_prepare_fundamentals(state: FlowState) -> FlowState:
        return {
            **state,
            **_reset_work(state),
            "work_stage": "FUNDAMENTALS_ANALYST",
            "work_kind": "ANALYST",
            "work_input": (state.get("fund_report") or "") + (state.get("baseline_ctx") or ""),
            "work_required": ["summary", "key_points", "confidence", "recommendation_hint"],
            "work_schema_hint": '{ "role":"FUNDAMENTALS_ANALYST","ticker":"","timestamp":"","summary":"","key_points":[],"confidence":0.0,"recommendation_hint":"" }',
            "work_auto_tool_calls": [{"function": "OVERVIEW", "symbol": ticker}],
            "work_next": "init_debate",
        }

    def node_init_debate(state: FlowState) -> FlowState:
        return {
            **state,
            "debate_round": 1,
            "bull_last": "",
            "bear_last": "",
            "bull_rounds": [],
            "bear_rounds": [],
        }

    def _merged_analyst_summaries(state: FlowState) -> str:
        analysts = state.get("analysts") or {}
        return "\n\n".join([f"{k}: {(v or {}).get('summary','')[:1000]}" for k, v in analysts.items()])

    def node_prepare_bull_debate(state: FlowState) -> FlowState:
        round_n = int(state.get("debate_round") or 1)
        base = _merged_analyst_summaries(state)
        debate_ctx = (
            f"\n\nDEBATE_ROUND: {round_n}\n"
            "YOUR_ROLE: BULL_RESEARCHER\n"
            "TASK: Respond to the opponent's last points, add new evidence/counterarguments, and update your JSON.\n"
            f"OPPONENT_LAST:\n{state.get('bear_last') or '(none)'}\n"
        )
        return {
            **state,
            **_reset_work(state),
            "work_stage": f"BULL_DEBATE_R{round_n}",
            "work_kind": "RESEARCHER",
            "work_stance": "BULL",
            "work_input": base + debate_ctx,
            "work_required": ["final_label", "consensus_summary", "evidence", "counterarguments", "confidence"],
            "work_schema_hint": '{ "stance":"BULL|BEAR","final_label":"BULLISH|BEARISH|NEUTRAL","consensus_summary":"","evidence":[],"counterarguments":[],"confidence":0.0 }',
            "work_auto_tool_calls": [{"function": "NEWS_SENTIMENT", "tickers": ticker}],
            "work_next": "prepare_bear_debate",
        }

    def node_prepare_bear_debate(state: FlowState) -> FlowState:
        round_n = int(state.get("debate_round") or 1)
        base = _merged_analyst_summaries(state)
        debate_ctx = (
            f"\n\nDEBATE_ROUND: {round_n}\n"
            "YOUR_ROLE: BEAR_RESEARCHER\n"
            "TASK: Respond to the opponent's last points, add new evidence/counterarguments, and update your JSON.\n"
            f"OPPONENT_LAST:\n{state.get('bull_last') or '(none)'}\n"
        )
        return {
            **state,
            **_reset_work(state),
            "work_stage": f"BEAR_DEBATE_R{round_n}",
            "work_kind": "RESEARCHER",
            "work_stance": "BEAR",
            "work_input": base + debate_ctx,
            "work_required": ["final_label", "consensus_summary", "evidence", "counterarguments", "confidence"],
            "work_schema_hint": '{ "stance":"BULL|BEAR","final_label":"BULLISH|BEARISH|NEUTRAL","consensus_summary":"","evidence":[],"counterarguments":[],"confidence":0.0 }',
            "work_auto_tool_calls": [{"function": "NEWS_SENTIMENT", "tickers": ticker}],
            "work_next": "route_after_bear",
        }

    def route_after_bear(state: FlowState) -> str:
        round_n = int(state.get("debate_round") or 1)
        if round_n >= 3:
            return "discussion"
        return "inc_round"

    def node_inc_round(state: FlowState) -> FlowState:
        round_n = int(state.get("debate_round") or 1)
        return {**state, "debate_round": round_n + 1}

    def node_discussion(state: FlowState) -> FlowState:
        discussion_summary = _summary_fallback(state.get("bull") or {}) + "\n\n" + _summary_fallback(state.get("bear") or {})
        return {**state, "discussion_summary": discussion_summary}

    def node_prepare_risk(state: FlowState) -> FlowState:
        return {
            **state,
            **_reset_work(state),
            "work_stage": "RISK_ANALYST",
            "work_kind": "RISK_ANALYST",
            "work_input": state.get("discussion_summary") or "",
            "work_required": ["risk_score", "breach_flags", "explainers"],
            "work_schema_hint": '{ "risk_score":0, "breach_flags":[], "explainers":[] }',
            "work_auto_tool_calls": [],
            "work_next": "prepare_manager",
        }

    def node_prepare_manager(state: FlowState) -> FlowState:
        aggregated_info = json.dumps(
            {
                "discussion_summary": state.get("discussion_summary") or "",
                "bull": state.get("bull") or {},
                "bear": state.get("bear") or {},
                "risk": state.get("risk") or {},
            },
            ensure_ascii=False,
        )
        return {
            **state,
            **_reset_work(state),
            "work_stage": "RISK_MANAGER",
            "work_kind": "RISK_MANAGER",
            "work_input": aggregated_info,
            "work_required": ["decision", "reason", "next_steps"],
            "work_schema_hint": '{ "decision":"approve|reject|require_manual", "reason":"", "next_steps":"" }',
            "work_auto_tool_calls": [{"function": "NEWS_SENTIMENT", "tickers": ticker}],
            "work_next": "prepare_trader",
        }

    def node_prepare_trader(state: FlowState) -> FlowState:
        payload = {"manager_decision": state.get("manager_decision") or {}, "discussion_summary": state.get("discussion_summary") or ""}
        raw_input = json.dumps(payload, ensure_ascii=False)
        return {
            **state,
            **_reset_work(state),
            "work_stage": "TRADER",
            "work_kind": "TRADER",
            "work_input": raw_input,
            "work_required": ["ticker", "side", "size", "entry", "stop", "target", "rationale", "confidence"],
            "work_schema_hint": '{ "ticker":"", "side":"BUY|SELL|NO_TRADE", "size":0.0, "entry":0.0, "stop":0.0, "target":0.0, "rationale":"", "confidence":0.0, "created_by":"TRADER" }',
            "work_auto_tool_calls": [{"function": "GLOBAL_QUOTE", "symbol": ticker}],
            "work_next": "finalize",
        }

    def node_route_after_bear(state: FlowState) -> FlowState:
        return state

    def node_finalize(state: FlowState) -> FlowState:
        import json as _json

        final = {
            "analysts": state.get("analysts") or {},
            "researchers": {
                "bull": state.get("bull") or {},
                "bear": state.get("bear") or {},
                "bull_rounds": state.get("bull_rounds") or [],
                "bear_rounds": state.get("bear_rounds") or [],
                "discussion": state.get("discussion_summary") or "",
            },
            "risk": state.get("risk") or {},
            "manager_decision": state.get("manager_decision") or {},
            "trader_proposal": state.get("trader_proposal") or {},
            "trace": state.get("trace") or [],
        }

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("outputs", ticker)
        os.makedirs(out_dir, exist_ok=True)
        archive_path = os.path.join(out_dir, f"flow_output_langchain_{ts}.json")

        with open("flow_output_langchain.json", "w", encoding="utf-8") as fout:
            _json.dump(final, fout, ensure_ascii=False, indent=2)
        with open(archive_path, "w", encoding="utf-8") as fout:
            _json.dump(final, fout, ensure_ascii=False, indent=2)

        final["output_paths"] = {"latest": "flow_output_langchain.json", "archive": archive_path.replace("\\", "/")}
        return {**state, "final": final}

    graph = StateGraph(FlowState)
    graph.add_node("load_template", node_load_template)
    graph.add_node("prefetch", node_prefetch)

    # prepares
    graph.add_node("prepare_market", node_prepare_market)
    graph.add_node("prepare_social", node_prepare_social)
    graph.add_node("prepare_news", node_prepare_news)
    graph.add_node("prepare_fundamentals", node_prepare_fundamentals)
    graph.add_node("init_debate", node_init_debate)
    graph.add_node("prepare_bull_debate", node_prepare_bull_debate)
    graph.add_node("prepare_bear_debate", node_prepare_bear_debate)
    graph.add_node("route_after_bear", node_route_after_bear)
    graph.add_node("inc_round", node_inc_round)
    graph.add_node("discussion", node_discussion)
    graph.add_node("prepare_risk", node_prepare_risk)
    graph.add_node("prepare_manager", node_prepare_manager)
    graph.add_node("prepare_trader", node_prepare_trader)
    graph.add_node("finalize", node_finalize)

    # shared pipeline
    graph.add_node("work_llm", node_work_llm)
    graph.add_node("work_parse", node_work_parse)
    graph.add_node("work_tool", node_work_tool)
    graph.add_node("work_validate", node_work_validate)
    graph.add_node("work_repair1", node_work_repair1)
    graph.add_node("work_auto_tools", node_work_auto_tools)
    graph.add_node("work_commit", node_work_commit)

    graph.set_entry_point("load_template")
    graph.add_edge("load_template", "prefetch")
    graph.add_edge("prefetch", "prepare_market")

    # Any prepare -> work pipeline
    for n in [
        "prepare_market",
        "prepare_social",
        "prepare_news",
        "prepare_fundamentals",
        "prepare_bull_debate",
        "prepare_bear_debate",
        "prepare_risk",
        "prepare_manager",
        "prepare_trader",
    ]:
        graph.add_edge(n, "work_llm")

    graph.add_edge("work_llm", "work_parse")
    graph.add_conditional_edges("work_parse", route_tool_or_validate, {"work_tool": "work_tool", "work_validate": "work_validate"})
    graph.add_edge("work_tool", "work_llm")
    graph.add_conditional_edges("work_validate", route_after_validate, {"work_commit": "work_commit", "work_repair1": "work_repair1", "work_auto_tools": "work_auto_tools"})
    graph.add_edge("work_repair1", "work_llm")
    graph.add_edge("work_auto_tools", "work_llm")

    # After commit, hop to next stage based on work_next
    graph.add_conditional_edges(
        "work_commit",
        route_next_from_commit,
        {
            "prepare_social": "prepare_social",
            "prepare_news": "prepare_news",
            "prepare_fundamentals": "prepare_fundamentals",
            "init_debate": "init_debate",
            "prepare_bull_debate": "prepare_bull_debate",
            "prepare_bear_debate": "prepare_bear_debate",
            "route_after_bear": "route_after_bear",
            "prepare_risk": "prepare_risk",
            "prepare_manager": "prepare_manager",
            "prepare_trader": "prepare_trader",
            "finalize": "finalize",
        },
    )

    # Debate control
    graph.add_edge("init_debate", "prepare_bull_debate")
    graph.add_conditional_edges("route_after_bear", route_after_bear, {"inc_round": "inc_round", "discussion": "discussion"})
    graph.add_edge("inc_round", "prepare_bull_debate")

    # Continue after debate
    graph.add_edge("discussion", "prepare_risk")

    graph.add_edge("finalize", END)

    app = graph.compile()
    out_state: FlowState = app.invoke({"template_path": template_path, "ticker": ticker, "model": model})
    return out_state.get("final") or {}
