# Tests in this repo

This folder groups both fast logic tests and a live smoke check for the multi-agent + tool stack.

## tests/test_multi_agent_tools.py
- Purpose: logic-level validation without hitting external services.
- Strategy: stub vector store, Alpha Vantage fetch, and Ollama so the core wiring is exercised offline.
- Covers:
  - `search_qdrant`: formats hits (score/source/symbol/chunk/snippet).
  - `fetch_and_store_news`: calls fetch, stores into vector store with proper metadata, returns feed count.
  - `handle_tool_request`: parses `TOOL:` directives and routes to tools.
  - `Agent.make_messages`: builds system/user/assistant messages from transcript.
- Run: `python -m unittest tests.test_multi_agent_tools`

## tests/smoke_agent_tools_live.py
- Purpose: live end-to-end smoke using local SQLite + hnswlib + Ollama (no mocks).
- Strategy: write a sample doc to a temporary table, retrieve it, exercise tool parsing, and clean up.
- Prereqs: local SQLite file (or writable path) and an Ollama embeddings endpoint at `cfg.ollama_embed_url` with the model available.
- Covers:
  - `VectorStore.store_response`/`retrieve` against live services.
  - `Agent.make_messages` role wiring.
  - `handle_tool_request` search path with real vector hits.
- Run: `python tests/smoke_agent_tools_live.py`

## tests/test_ollama.py
- Purpose: sanity-check Ollama Python client connectivity/models.
- Strategy: try `ollama.generate` first, fallback to `ollama.chat`, print raw responses.
- Prereqs: Ollama daemon running and model `qwen2.5:14b` available (or edit the script to the model you have).
- Run: `python tests/test_ollama.py`
