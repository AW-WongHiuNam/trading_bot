from __future__ import annotations

import time
from typing import Any, Optional

from config import get_settings
from vector_store_sqlite import VectorStore


_vector_store: Optional[VectorStore] = None


def _get_store() -> VectorStore:
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    cfg = get_settings()
    _vector_store = VectorStore(
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
    return _vector_store


def rag_search(
    query: str,
    *,
    symbol: str | None = None,
    types: list[str] | None = None,
    stage: str | None = None,
    source: str | None = None,
    run_id: str | None = None,
    days: int | None = 30,
    top_k: int = 5,
    include_test: bool | None = None,
) -> dict[str, Any]:
    """Search the vector DB and return both raw hits + a prompt-ready context block."""
    if not query or not str(query).strip():
        return {"error": "missing query"}

    min_created_at = None
    if days is not None and int(days) > 0:
        min_created_at = time.time() - (int(days) * 86400)

    vs = _get_store()
    hits = vs.retrieve(
        query,
        top_k=int(top_k),
        symbol=symbol,
        types=types,
        stage=stage,
        source=source,
        run_id=run_id,
        include_test=include_test,
        min_created_at=min_created_at,
        candidate_k=max(int(top_k) * 10, 50),
    )
    context = vs.build_context(hits)

    compact_hits = []
    for doc, md, score in hits:
        compact_hits.append(
            {
                "score": float(score),
                "source": md.get("source"),
                "symbol": md.get("symbol") or md.get("tickers"),
                "type": md.get("type"),
                "stage": md.get("stage"),
                "timestamp": md.get("timestamp"),
                "is_test": bool(md.get("is_test", False)),
                "snippet": (doc or "")[:400],
            }
        )

    return {"query": query, "top_k": int(top_k), "hits": compact_hits, "context": context}
