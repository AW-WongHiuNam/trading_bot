"""Minimal VectorStore test using mock embeddings.

Creates a temporary Qdrant folder, stores a sample document, and queries it.
Run: python test_vector_store_dummy.py
"""
from __future__ import annotations

import os
import tempfile

from vector_store import VectorStore


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        qdrant_dir = os.path.join(tmpdir, "qdrant")
        vs = VectorStore(persist_path=qdrant_dir, force_mock_embed=True)
        sample_text = "Tesla shares jumped after earnings."
        metadata = {"source": "unit-test", "symbol": "TSLA", "type": "note"}
        vs.store_response(sample_text, metadata=metadata, chunk_size=64, overlap=0)
        results = vs.retrieve("What happened to Tesla?", top_k=3)
        print("Retrieved", len(results), "items")
        for doc, meta, score in results:
            print("---")
            print("score:", score)
            print("metadata:", meta)
            print("doc:", doc)


if __name__ == "__main__":
    main()
