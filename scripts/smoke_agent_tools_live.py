"""Live smoke test for multi_agent_chat tools using local SQLite + hnswlib + Ollama.

Requires:
- Writable `SQLITE_PATH` (default `vector_store.sqlite`)
- Ollama embeddings endpoint reachable at `cfg.ollama_embed_url`

Run:
    python smoke_agent_tools_live.py
"""

from __future__ import annotations

import os
import sqlite3
import time

from config import get_settings
from multi_agent_chat import Agent, handle_tool_request
from vector_store_sqlite import VectorStore


def main() -> None:
    cfg = get_settings()

    # Use a temp table so we don't touch the main one.
    table = f"smoke_agent_{int(time.time())}"
    vs = VectorStore(
        table_name=table,
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

    sample_text = "AAPL releases new product and market reacts positively."
    metadata = {"source": "smoke", "function": "manual", "tickers": "AAPL", "timestamp": int(time.time())}
    vs.store_response(sample_text, metadata=metadata, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    print(f"Stored sample text into table {table}")

    hits = vs.retrieve("AAPL product news", top_k=3)
    print("Retrieved hits:")
    for doc, payload, score in hits:
        print({"score": score, "source": payload.get("source"), "chunk": payload.get("chunk"), "snippet": doc[:120]})

    # Verify Agent message wiring (no network) and tool parser (search path is exercised above).
    agent = Agent("Tester", system_prompt="You are a concise assistant.")
    transcript = [("Tester", "hello"), ("Other", "run tool")]
    msgs = agent.make_messages(transcript)
    print(f"Agent messages roles: {[m['role'] for m in msgs]}")

    tool_out = handle_tool_request("TOOL: search_news=AAPL")
    print(f"handle_tool_request returned: {tool_out}")

    # Clean up the temp table and index file to keep state clean.
    try:
        with sqlite3.connect(cfg.sqlite_path) as conn:
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        if os.path.exists(cfg.vector_index_path):
            try:
                os.remove(cfg.vector_index_path)
            except Exception:
                pass
        print(f"Deleted temp table {table} and index file if present")
    except Exception as e:
        print(f"Warning: failed to delete temp table {table}: {e}")


if __name__ == "__main__":
    main()
