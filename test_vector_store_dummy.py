"""Minimal VectorStore test using mock embeddings.

Creates a temporary PostgreSQL table, stores a sample document, and queries it.
Run: python test_vector_store_dummy.py
"""
from __future__ import annotations

import time

from config import get_settings
from vector_store_sqlite import VectorStore
import sqlite3
import os


def main() -> None:
    cfg = get_settings()
    table = f"test_vector_store_{int(time.time())}"
    vs = VectorStore(
        table_name=table,
        sqlite_path=cfg.sqlite_path,
        index_path=cfg.vector_index_path,
        vector_dim=512,
        ann_space=cfg.ann_index_space,
        ann_ef=cfg.ann_ef,
        ann_m=cfg.ann_m,
        force_mock_embed=True,
    )
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
