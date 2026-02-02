#!/usr/bin/env python3
"""Fetch various Alpha Vantage endpoints and save to JSON/JSONL files.

Usage examples:
  python alpha_fetch.py --apikey YOUR_KEY --function NEWS_SENTIMENT --tickers AAPL --out data/aapl_news.json
  python alpha_fetch.py --apikey YOUR_KEY --function TIME_SERIES_DAILY --symbol IBM --out data/ibm_daily.json
  python alpha_fetch.py --apikey YOUR_KEY --function NEWS_SENTIMENT --tickers AAPL --jsonl --out data/aapl_news.jsonl

The script writes the full API response by default. For news-like endpoints that return
an article `feed`, using `--jsonl` will write each feed item as one JSON object per line
which is convenient for backend ingestion and frontend streaming.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

import requests
from config import get_settings
from vector_store_sqlite import VectorStore

API_URL = "https://www.alphavantage.co/query"

def fetch_av(function: str, api_key: str, params: Dict[str, str], retries: int = 3, backoff: float = 1.0) -> Dict[str, Any]:
    params = dict(params)
    params.update({"function": function, "apikey": api_key})

    attempt = 0
    while True:
        try:
            resp = requests.get(API_URL, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            attempt += 1
            if attempt > retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(items, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Alpha Vantage data and save to JSON/JSONL")
    parser.add_argument("--apikey", required=False, help="Alpha Vantage API key (overrides .env)")
    parser.add_argument("--function", required=True, help="Alpha Vantage function name, e.g. NEWS_SENTIMENT")
    parser.add_argument("--symbol", help="Symbol for single-symbol endpoints (e.g. IBM)")
    parser.add_argument("--tickers", help="Comma-separated tickers for NEWS_SENTIMENT")
    parser.add_argument("--quarter", help="Quarter for EARNINGS_CALL_TRANSCRIPT (e.g. 2024Q1)")
    parser.add_argument("--out", required=False, help="Output file path (default data/<function>.json)")
    parser.add_argument("--jsonl", action="store_true", help="If set, write newline-delimited JSON when possible")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of items when writing feeds (0=all)")
    args = parser.parse_args()

    cfg = get_settings()
    api_key = args.apikey or cfg.alphavantage_api_key
    if not api_key:
        raise ValueError("Alpha Vantage API key is required. Set ALPHAVANTAGE_API_KEY in .env or pass --apikey.")

    params = {}
    if args.symbol:
        params["symbol"] = args.symbol
    if args.tickers:
        params["tickers"] = args.tickers
    if args.quarter:
        params["quarter"] = args.quarter

    out_path = args.out or cfg.default_output_path or f"{cfg.data_dir}/{args.function.lower()}"
    if not os.path.splitext(out_path)[1]:
        out_path += ".jsonl" if args.jsonl else ".json"

    # Fetch
    print(f"Fetching {args.function} with params={params}")
    data = fetch_av(args.function, api_key=api_key, params=params)

    # Store API response into vector DB (chunked + embeddings via Ollama)
    try:
        vs = VectorStore(
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
        text_for_store = json.dumps(data, ensure_ascii=False)
        metadata = {
            "source": "alpha_fetch",
            "function": args.function,
            "symbol": args.symbol,
            "tickers": args.tickers,
            "timestamp": int(time.time()),
            "type": "api_response",
        }
        vs.store_response(text_for_store, metadata=metadata, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        print("Stored API response to vector DB (SQLite + hnswlib)")
    except Exception as e:
        print(f"Warning: failed to store response to vector DB: {e}")

    # If JSONL requested and response contains a 'feed' list, write feed items as JSONL
    if args.jsonl and isinstance(data, dict) and "feed" in data and isinstance(data["feed"], list):
        items = data["feed"]
        if args.limit and args.limit > 0:
            items = items[: args.limit]
        save_jsonl(items, out_path)
        print(f"Wrote {len(items)} items to {out_path}")
        return

    # Otherwise, optionally trim list-like subkeys by limit if requested
    if args.limit and isinstance(data, dict):
        # apply simple trimming to top-level list-valued keys
        for k, v in list(data.items()):
            if isinstance(v, list):
                data[k] = v[: args.limit]

    save_json(data, out_path)
    print(f"Wrote response to {out_path}")


if __name__ == "__main__":
    main()
