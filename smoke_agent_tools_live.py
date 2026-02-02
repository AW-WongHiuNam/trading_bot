"""Live smoke test for multi_agent_chat tools using real services (Qdrant + Ollama).

Requires:
- Qdrant reachable at cfg.qdrant_url (default http://localhost:7500)
- Ollama embeddings endpoint reachable at cfg.ollama_embed_url
- No mocks involved; uses a temporary collection to avoid polluting main data.

Run:
  python smoke_agent_tools_live.py
"""
from __future__ import annotations

import time
from config import get_settings
from vector_store import VectorStore
from multi_agent_chat import Agent, handle_tool_request


def main() -> None:
    cfg = get_settings()

    # Use a temp collection so we don't touch the main one.
    coll = f"smoke_agent_{int(time.time())}"
    vs = VectorStore(
        collection_name=coll,
        persist_path=cfg.qdrant_path,
        qdrant_url=cfg.qdrant_url,
        ollama_model=cfg.ollama_embed_model,
        ollama_url=cfg.ollama_embed_url,
        force_mock_embed=False,
    )

    sample_text = "AAPL releases new product and market reacts positively."
    metadata = {"source": "smoke", "function": "manual", "tickers": "AAPL", "timestamp": int(time.time())}
    vs.store_response(sample_text, metadata=metadata, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    print(f"Stored sample text into collection {coll}")

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

    # Clean up the temp collection to keep state clean.
    try:
        vs.client.delete_collection(coll)
        print(f"Deleted temp collection {coll}")
    except Exception as e:
        print(f"Warning: failed to delete temp collection {coll}: {e}")


if __name__ == "__main__":
    main()
