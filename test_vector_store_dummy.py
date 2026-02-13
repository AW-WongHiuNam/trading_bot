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
import os as _os


def main() -> None:
    cfg = get_settings()
    table = f"test_vector_store_{int(time.time())}"

    # Mark writes as test data
    _os.environ["RAG_IS_TEST"] = "1"

    vs = VectorStore(
        table_name=table,
        sqlite_path=f"tmp_{table}.sqlite",
        index_path=f"tmp_{table}.bin",
        vector_dim=512,
        ann_space=cfg.ann_index_space,
        ann_ef=cfg.ann_ef,
        ann_m=cfg.ann_m,
        force_mock_embed=True,
    )
    sample_text = "Tesla shares jumped after earnings."
    metadata = {"source": "unit-test", "symbol": "TSLA", "type": "note"}
    vs.store_response(sample_text, metadata=metadata, chunk_size=64, overlap=0)

    # Default behavior: exclude is_test=true
    results = vs.retrieve("What happened to Tesla?", top_k=3)
    print("Retrieved (exclude test)", len(results), "items")

    # Explicitly include test items
    results_inc = vs.retrieve("What happened to Tesla?", top_k=3, include_test=True)
    print("Retrieved (include test)", len(results_inc), "items")

    if len(results) != 0:
        raise SystemExit("Expected 0 results when excluding test data")
    if len(results_inc) == 0:
        raise SystemExit("Expected results when include_test=True")

    for doc, meta, score in results_inc:
        print("---")
        print("score:", score)
        print("metadata:", meta)
        print("doc:", doc)

    try:
        with sqlite3.connect(f"tmp_{table}.sqlite") as conn:
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        if os.path.exists(f"tmp_{table}.sqlite"):
            try:
                os.remove(f"tmp_{table}.sqlite")
            except Exception:
                pass
        if os.path.exists(f"tmp_{table}.bin"):
            try:
                os.remove(f"tmp_{table}.bin")
            except Exception:
                pass
        print(f"Deleted temp table {table} and temp sqlite/index files")
    except Exception as e:
        print(f"Warning: failed to delete temp table {table}: {e}")


if __name__ == "__main__":
    main()
