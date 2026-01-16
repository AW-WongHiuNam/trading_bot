"""Offline smoke test for alpha_fetch.py.

- Patches alpha_fetch.fetch_av to avoid hitting the real API.
- Forces VectorStore into mock-embedding mode and writes output to a temp folder.
Run: python test_alpha_fetch_dummy.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import alpha_fetch


def fake_response() -> Dict[str, Any]:
    return {
        "feed": [
            {
                "title": "Test headline",
                "summary": "Lorem ipsum dolor sit amet",
                "url": "https://example.com/a",
            },
            {
                "title": "Another headline",
                "summary": "consectetur adipiscing elit",
                "url": "https://example.com/b",
            },
        ],
        "meta": {"symbol": "AAPL", "source": "stub"},
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "news.json"
        qdrant_dir = Path(tmpdir) / "qdrant"
        os.environ["QDRANT_PATH"] = str(qdrant_dir)
        os.environ["QDRANT_FORCE_MOCK_EMBED"] = "1"
        argv = [
            "alpha_fetch.py",
            "--apikey",
            "demo-key",
            "--function",
            "NEWS_SENTIMENT",
            "--tickers",
            "AAPL",
            "--out",
            str(out_file),
            "--limit",
            "2",
            "--jsonl",
        ]
        with patch.object(sys, "argv", argv):
            with patch("alpha_fetch.fetch_av", return_value=fake_response()):
                alpha_fetch.main()
        print("alpha_fetch wrote:", out_file)
        if out_file.exists():
            print(out_file.read_text(encoding="utf-8"))
        else:
            print("output file missing!")


if __name__ == "__main__":
    main()
