"""SQLite + hnswlib quickstart using the project's VectorStore.

Creates a temporary SQLite table, inserts a few sample documents using
`force_mock_embed=True`, searches, prints results, and cleans up.
"""
import time
import os
from vector_store_sqlite import VectorStore


def main() -> None:
    table = f"sqlite_quickstart_{int(time.time())}"
    vs = VectorStore(table_name=table, sqlite_path="vector_store.sqlite", index_path="vector_index.bin", vector_dim=128, force_mock_embed=True)

    samples = [
        ("alpha text about apples", {"text": "alpha"}),
        ("bravo text about bananas", {"text": "bravo"}),
        ("charlie text about cherries", {"text": "charlie"}),
    ]

    for txt, md in samples:
        vs.store_response(txt, metadata=md, chunk_size=64, overlap=0)

    res = vs.retrieve("tell me about cherries", top_k=3)
    print("search results:")
    for doc, payload, score in res:
        print({"score": score, "payload": payload, "snippet": doc[:80]})

    # cleanup
    try:
        import sqlite3
        conn = sqlite3.connect("vector_store.sqlite")
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        conn.commit()
        conn.close()
        if os.path.exists("vector_index.bin"):
            try:
                os.remove("vector_index.bin")
            except Exception:
                pass
        print("cleaned up")
    except Exception as e:
        print("cleanup warning:", e)


if __name__ == "__main__":
    main()
