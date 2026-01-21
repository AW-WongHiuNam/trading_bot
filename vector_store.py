from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams


class OllamaEmbeddings:
    """Simple Ollama embedding wrapper.

    Expects an Ollama embedding endpoint that accepts JSON:
      POST {url} -> {"model": "model-name", "input": ["text1", ...]}
    and returns JSON containing an "embeddings" list of lists.
    Adjust `ollama_url` if your Ollama API is exposed elsewhere.
    """

    def __init__(self, model: str = "nomic-embed-text", ollama_url: Optional[str] = None):
        self.model = model
        self.ollama_url = ollama_url or "http://127.0.0.1:11434/api/embeddings"

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Call Ollama embeddings API. Uses per-item calls because the API expects a single prompt."""
        if not texts:
            return []

        vectors: List[List[float]] = []
        for t in texts:
            payload = {"model": self.model, "prompt": t or ""}
            resp = requests.post(self.ollama_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict):
                if "embedding" in data and isinstance(data["embedding"], list) and data["embedding"]:
                    vectors.append(data["embedding"])
                    continue
                if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
                    # Some variants return a list-of-vectors under "embeddings" even for single prompt.
                    if isinstance(data["embeddings"][0], list):
                        vectors.extend(data["embeddings"])
                        continue
            raise ValueError("Unexpected response from Ollama embed endpoint: %r" % (data,))

        return vectors


class MockEmbeddings:
    """Deterministic mock embedder for local testing without an embed service."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _text_to_vector(self, text: str) -> List[float]:
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec: List[float] = []
        i = 0
        while len(vec) < self.dim:
            b = h[i % len(h)]
            vec.append(((b / 255.0) * 2.0) - 1.0)
            i += 1
        return vec[: self.dim]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(t or "") for t in texts]


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 400) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    L = len(text)
    while i < L:
        end = min(i + chunk_size, L)
        chunks.append(text[i:end])
        if end == L:
            break
        i = end - overlap
        if i <= 0:
            i = end
    return chunks


class VectorStore:
    def __init__(
        self,
        collection_name: str = "api_calls",
        persist_path: str = "qdrant_db",
        qdrant_url: Optional[str] = "http://localhost:6333",
        ollama_model: str = "nomic-embed-text:latest",
        ollama_url: Optional[str] = None,
        force_mock_embed: bool = False,
    ) -> None:
        self.collection_name = collection_name
        self.persist_path = persist_path or "qdrant_db"
        if qdrant_url:
            # Default to Docker/HTTP endpoint; override via constructor if needed.
            self.client = QdrantClient(url=qdrant_url)
        else:
            # Fallback to embedded/local folder for tests or offline use.
            self.client = QdrantClient(path=self.persist_path)

        self.embedder = OllamaEmbeddings(model=ollama_model, ollama_url=ollama_url)
        if force_mock_embed or os.environ.get("QDRANT_FORCE_MOCK_EMBED"):
            print("VectorStore: using MockEmbeddings (forced)")
            self.embedder = MockEmbeddings()

    def _ensure_collection(self, dim: int) -> None:
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            vectors_cfg = getattr(info.config.params, "vectors", None)
            existing_size: Optional[int] = None
            if hasattr(vectors_cfg, "size"):
                existing_size = vectors_cfg.size
            elif isinstance(vectors_cfg, dict):
                existing_size = vectors_cfg.get("size")
            if existing_size and existing_size != dim:
                raise ValueError(
                    f"Collection {self.collection_name} exists with dim={existing_size}, incoming dim={dim}. "
                    "Drop or recreate the collection manually to proceed."
                )
            return

        print(f"VectorStore: creating Qdrant collection {self.collection_name} with dim={dim}")
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        except UnexpectedResponse as e:
            # Handle race/dup-create cases gracefully.
            if getattr(e, "status_code", None) == 409 or "already exists" in str(e):
                return
            raise

    def store_response(self, text: str, metadata: Optional[Dict[str, Any]] = None, chunk_size: int = 2000, overlap: int = 400) -> None:
        metadata = metadata or {}
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return
        print(f"VectorStore.store_response: chunked into {len(chunks)} chunks")
        try:
            embeddings = self.embedder.embed_batch(chunks)
            print(f"VectorStore.store_response: got {len(embeddings)} embeddings")
        except Exception:
            import traceback

            print("VectorStore.store_response: embed_batch raised an exception:")
            traceback.print_exc()
            if not isinstance(self.embedder, MockEmbeddings):
                try:
                    print("VectorStore.store_response: falling back to MockEmbeddings and retrying")
                    self.embedder = MockEmbeddings()
                    embeddings = self.embedder.embed_batch(chunks)
                    print(f"VectorStore.store_response: got {len(embeddings)} mock embeddings")
                except Exception:
                    traceback.print_exc()
                    raise
            else:
                raise

        if not embeddings or not embeddings[0]:
            raise ValueError("No embeddings returned; aborting store_response")

        dim = len(embeddings[0])
        self._ensure_collection(dim)

        points: List[PointStruct] = []
        for idx, (doc, emb) in enumerate(zip(chunks, embeddings)):
            payload = metadata.copy()
            payload.setdefault("chunk", idx)
            payload["document"] = doc
            points.append(PointStruct(id=str(uuid.uuid4()), vector=emb, payload=payload))

        print("VectorStore.store_response: upserting into Qdrant (wait=True) ...")
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        print("VectorStore.store_response: upsert returned")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        emb = self.embedder.embed_batch([query])[0]
        self._ensure_collection(len(emb))
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=emb,
            limit=top_k,
            with_payload=True,
        )
        items: List[Tuple[str, Dict[str, Any], float]] = []
        for point in res.points:
            payload = point.payload or {}
            doc = payload.get("document", "")
            items.append((doc, payload, float(point.score)))
        return items

    def build_context(self, items: List[Tuple[str, Dict[str, Any], float]], max_chars: int = 3000) -> str:
        parts: List[str] = []
        total = 0
        for i, (doc, md, score) in enumerate(items, start=1):
            src = md.get("source", "unknown")
            line = f"[{i}] source={src} chunk={md.get('chunk')} score={score}\n{doc}\n"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)
        return "\n".join(parts)

    def answer_query(
        self,
        query: str,
        top_k: int = 5,
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
    ) -> Any:
        items = self.retrieve(query, top_k=top_k)
        context = self.build_context(items)
        system_prompt = system_prompt or "你是一個有用的助理，使用下面的參考資料回答使用者問題。"
        user_prompt_template = user_prompt_template or "參考資料：\n{context}\n\n問題：{query}\n請依據參考資料回答。"
        prompt = f"{system_prompt}\n\n" + user_prompt_template.format(query=query, context=context)

        url = ollama_url or "http://127.0.0.1:11434/completion"
        model = ollama_model or "gpt-4o-mini" if ollama_model is None else ollama_model
        payload = {"model": model, "prompt": prompt, "max_tokens": 512}
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            try:
                data = resp.json()
                if isinstance(data, dict):
                    if "output" in data:
                        return data["output"]
                    if "choices" in data and data["choices"]:
                        return data["choices"][0].get("text") or data["choices"][0].get("message")
                return data
            except ValueError:
                return resp.text
        except Exception as e:
            return {"error": str(e), "prompt": prompt}


if __name__ == "__main__":
    print("vector_store.py: Qdrant-backed helper. Import VectorStore in your code.")
