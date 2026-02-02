"""Lightweight logic tests for multi_agent_chat tools and Agent wiring.

Uses in-process fakes for vector store, Alpha Vantage fetch, and Ollama so it runs
without network access.

Run:
    python -m unittest tests.test_multi_agent_tools
"""
from __future__ import annotations

import sys
import types
import unittest

# Provide a minimal ollama stub if not installed to keep imports happy.
if "ollama" not in sys.modules:
    stub = types.ModuleType("ollama")
    stub.chat = lambda *_, **__: {"response": "ok"}
    stub.generate = lambda *_, **__: {"response": "ok"}
    sys.modules["ollama"] = stub

import multi_agent_chat as mac  # noqa: E402  # after stub


class DummyVectorStore:
    def __init__(self):
        self.stored = []

    def retrieve(self, query: str, top_k: int = 5):
        # Return one fake hit to verify formatting.
        payload = {"source": "alpha_fetch", "symbol": "AAPL", "chunk": 0}
        return [("doc about AAPL", payload, 0.9876)]

    def store_response(self, text: str, metadata=None, chunk_size: int = 0, overlap: int = 0):
        self.stored.append((text, metadata, chunk_size, overlap))


class DummyFetch:
    calls = []

    @classmethod
    def fetch_av(cls, function: str, api_key: str, params: dict, retries: int = 3, backoff: float = 1.0):
        cls.calls.append((function, params))
        return {"feed": [{"title": "t1"}, {"title": "t2"}]}


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.orig_vs = mac._vector_store
        self.orig_fetch = mac.fetch_av
        mac._vector_store = DummyVectorStore()
        mac.fetch_av = DummyFetch.fetch_av

    def tearDown(self):
        mac._vector_store = self.orig_vs
        mac.fetch_av = self.orig_fetch

    def test_search_qdrant_returns_hits(self):
        res = mac.search_qdrant("AAPL news", top_k=3)
        self.assertIn("hits", res)
        self.assertEqual(len(res["hits"]), 1)
        hit = res["hits"][0]
        self.assertEqual(hit["source"], "alpha_fetch")
        self.assertAlmostEqual(hit["score"], 0.9876, places=4)

    def test_fetch_and_store_news_uses_store(self):
        res = mac.fetch_and_store_news("AAPL")
        self.assertEqual(res["stored_items"], 2)
        self.assertEqual(res["tickers"], "AAPL")
        self.assertEqual(len(mac._vector_store.stored), 1)
        stored_md = mac._vector_store.stored[0][1]
        self.assertEqual(stored_md.get("function"), "NEWS_SENTIMENT")

    def test_handle_tool_request_parses(self):
        out = mac.handle_tool_request("TOOL: search_news=ai chips")
        self.assertIsInstance(out, tuple)
        name, payload, _raw = out
        self.assertEqual(name, "search_news")
        self.assertIn("hits", payload)

    def test_agent_message_wiring(self):
        agent = mac.Agent("Alice", system_prompt="sys")
        transcript = [("Alice", "hi"), ("Bob", "yo")]
        msgs = agent.make_messages(transcript)
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles[0], "system")
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertIn("Alice: hi", msgs[1]["content"])


if __name__ == "__main__":
    unittest.main()
