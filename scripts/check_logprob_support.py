#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import requests


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _safe_json(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return {"_non_json": True, "text": resp.text[:2000]}


def _find_logprob_keys(obj: Any, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key_path = f"{prefix}.{k}" if prefix else str(k)
            kl = str(k).lower()
            if "logprob" in kl or "top_logprobs" in kl:
                hits.append(key_path)
            hits.extend(_find_logprob_keys(v, key_path))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key_path = f"{prefix}[{i}]"
            hits.extend(_find_logprob_keys(v, key_path))
    return hits


def _post(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    started = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.time() - started
        body = _safe_json(resp)
        return {
            "ok": resp.status_code == 200,
            "status_code": resp.status_code,
            "elapsed_sec": round(elapsed, 3),
            "response": body,
            "response_keys": list(body.keys())[:50] if isinstance(body, dict) else [],
            "logprob_paths": _find_logprob_keys(body),
        }
    except Exception as exc:
        elapsed = time.time() - started
        return {
            "ok": False,
            "status_code": None,
            "elapsed_sec": round(elapsed, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "response": None,
            "response_keys": [],
            "logprob_paths": [],
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase0: check whether the current Ollama setup returns token logprobs"
    )
    parser.add_argument("--base-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument(
        "--model",
        default=(os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL") or "deepseek-r1:latest"),
    )
    parser.add_argument("--prompt", default="Return JSON: {\"ok\":true}")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--out-dir", default="outputs/diagnostics")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    checks: dict[str, dict[str, Any]] = {}

    # 1) Native Ollama generate endpoint (non-streaming)
    checks["ollama_generate_plain"] = _post(
        f"{base}/api/generate",
        {
            "model": args.model,
            "prompt": args.prompt,
            "stream": False,
            "options": {"num_predict": 32},
        },
        timeout=args.timeout,
    )

    # 2) Native Ollama generate with explicit logprob-ish option
    # Not all servers support this option; this test helps detect support quickly.
    checks["ollama_generate_with_logprobs_option"] = _post(
        f"{base}/api/generate",
        {
            "model": args.model,
            "prompt": args.prompt,
            "stream": False,
            "options": {"logprobs": 5, "num_predict": 32},
        },
        timeout=args.timeout,
    )

    # 3) OpenAI-compatible chat endpoint (if available)
    checks["openai_chat_completions_logprobs"] = _post(
        f"{base}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt},
            ],
            "temperature": 0,
            "max_tokens": 64,
            "logprobs": True,
            "top_logprobs": 5,
        },
        timeout=args.timeout,
    )

    # 4) OpenAI-compatible completions endpoint (if available)
    checks["openai_completions_logprobs"] = _post(
        f"{base}/v1/completions",
        {
            "model": args.model,
            "prompt": args.prompt,
            "temperature": 0,
            "max_tokens": 64,
            "logprobs": 5,
        },
        timeout=args.timeout,
    )

    any_logprobs = any(bool(c.get("logprob_paths")) for c in checks.values())
    native_generate_has_logprobs = any(
        bool(checks[name].get("logprob_paths"))
        for name in ["ollama_generate_plain", "ollama_generate_with_logprobs_option"]
        if name in checks
    )
    openai_compat_has_logprobs = any(
        bool(checks[name].get("logprob_paths"))
        for name in ["openai_chat_completions_logprobs", "openai_completions_logprobs"]
        if name in checks
    )

    conclusion = {
        "phase": "Phase0",
        "base_url": base,
        "model": args.model,
        "native_generate_has_logprobs": native_generate_has_logprobs,
        "openai_compat_has_logprobs": openai_compat_has_logprobs,
        "any_logprobs_found": any_logprobs,
        "recommendation": (
            "Proceed with direct entropy implementation"
            if any_logprobs
            else "Current endpoints do not expose token logprobs. Use a logprob-capable provider/backend before Phase1."
        ),
    }

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "conclusion": conclusion,
        "checks": checks,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"logprob_capability_{_now()}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Phase0 Logprob Capability Check ===")
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))
    print(f"Report written: {str(out_path).replace('\\\\', '/')}")


if __name__ == "__main__":
    main()
