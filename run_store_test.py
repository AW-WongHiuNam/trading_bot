from config import get_settings
from vector_store_sqlite import VectorStore


def main():
    cfg = get_settings()
    vs = VectorStore(
        table_name="smoke_demo",
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

    text = "Sample test document: AAPL product launch caused market reaction."
    md = {"source": "manual_test", "tickers": "AAPL", "type": "demo"}
    vs.store_response(text, metadata=md, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    print("Stored sample text into SQLite vector store using Ollama embeddings.")
    res = vs.retrieve("AAPL product news", top_k=3)
    print("Retrieved:")
    for doc, meta, score in res:
        print({"score": score, "meta": meta, "snippet": doc[:120]})


if __name__ == '__main__':
    main()
