from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import hnswlib


def _env_truthy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class OllamaEmbeddings:
    def __init__(self, model: str = "nomic-embed-text", ollama_url: Optional[str] = None):
        self.model = model
        self.ollama_url = ollama_url or "http://127.0.0.1:11434/api/embeddings"

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
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
                    if isinstance(data["embeddings"][0], list):
                        vectors.extend(data["embeddings"])
                        continue
            raise ValueError("Unexpected response from Ollama embed endpoint: %r" % (data,))
        return vectors


class MockEmbeddings:
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
        table_name: str = "api_calls",
        sqlite_path: str = "vector_store.sqlite",
        index_path: str = "vector_index.bin",
        vector_dim: int = 768,
        ann_space: str = "ip",
        ann_ef: int = 200,
        ann_m: int = 16,
        ollama_model: str = "nomic-embed-text:latest",
        ollama_url: Optional[str] = None,
        force_mock_embed: bool = False,
    ) -> None:
        self.table_name = table_name
        self.sqlite_path = sqlite_path
        self.index_path = index_path
        self.vector_dim = vector_dim
        self.ann_space = ann_space
        self.ann_ef = ann_ef
        self.ann_m = ann_m

        self._conn_lock = threading.Lock()
        self._ensure_db()

        self.index = None
        self._max_elements = 1000000
        self._load_or_create_index()

        self.embedder = OllamaEmbeddings(model=ollama_model, ollama_url=ollama_url)
        if force_mock_embed or _env_truthy("VECTOR_FORCE_MOCK_EMBED"):
            print("VectorStore: using MockEmbeddings (forced)")
            self.embedder = MockEmbeddings(dim=self.vector_dim)

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.sqlite_path) or ".", exist_ok=True)
        with self._conn() as conn:
            cur = conn.cursor()
            # if table doesn't exist, create with idx as INTEGER PRIMARY KEY AUTOINCREMENT
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.table_name,))
            if not cur.fetchone():
                cur.execute(
                    f"""
                    CREATE TABLE {self.table_name} (
                        idx INTEGER PRIMARY KEY AUTOINCREMENT,
                        id TEXT UNIQUE,
                        document TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
                conn.commit()
                return

            # table exists: inspect schema and migrate if idx is not an INTEGER PRIMARY KEY
            cur.execute(f"PRAGMA table_info({self.table_name})")
            cols = cur.fetchall()
            # PRAGMA table_info returns rows: (cid, name, type, notnull, dflt_value, pk)
            idx_col = None
            for c in cols:
                if c[1] == "idx":
                    idx_col = c
                    break

            need_migration = False
            if idx_col is None:
                need_migration = True
            else:
                # if idx exists but is not declared as primary key, migrate
                if idx_col[5] == 0:
                    need_migration = True

            if not need_migration:
                return

            # perform migration: create new temp table with desired schema and copy rows
            temp = f"{self.table_name}__tmp"
            cur.execute(
                f"""
                CREATE TABLE {temp} (
                    idx INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT UNIQUE,
                    document TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            # copy rows: if old table had idx column, preserve values; otherwise let SQLite assign
            if idx_col is not None:
                cur.execute(f"INSERT INTO {temp} (idx, id, document, metadata, created_at) SELECT idx, id, document, metadata, created_at FROM {self.table_name} ORDER BY idx")
            else:
                cur.execute(f"INSERT INTO {temp} (id, document, metadata, created_at) SELECT id, document, metadata, created_at FROM {self.table_name}")
            cur.execute(f"DROP TABLE {self.table_name}")
            cur.execute(f"ALTER TABLE {temp} RENAME TO {self.table_name}")
            conn.commit()

    def _conn(self):
        return sqlite3.connect(self.sqlite_path)

    def _load_or_create_index(self):
        space = self.ann_space
        dim = self.vector_dim
        self.index = hnswlib.Index(space=space, dim=dim)
        if os.path.exists(self.index_path):
            try:
                self.index.load_index(self.index_path)
                self.index.set_ef(self.ann_ef)
            except Exception:
                # recreate
                self.index = hnswlib.Index(space=space, dim=dim)
                self.index.init_index(max_elements=self._max_elements, ef_construction=self.ann_ef, M=self.ann_m)
        else:
            self.index.init_index(max_elements=self._max_elements, ef_construction=self.ann_ef, M=self.ann_m)

    def _next_idx(self) -> int:
        with self._conn_lock:
            with self._conn() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT MAX(idx) FROM {self.table_name}")
                r = cur.fetchone()
                if not r or r[0] is None:
                    return 0
                return int(r[0]) + 1

    def _normalize(self, v: List[float]) -> List[float]:
        a = np.array(v, dtype="float32")
        norm = np.linalg.norm(a)
        if norm == 0:
            return a.tolist()
        return (a / norm).tolist()

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
                    self.embedder = MockEmbeddings(dim=self.vector_dim)
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
        if dim != self.vector_dim:
            raise ValueError(f"Embedding dim {dim} does not match configured vector_dim {self.vector_dim}.")

        rows = []
        vecs = []
        for idx_local, (doc, emb) in enumerate(zip(chunks, embeddings)):
            payload = metadata.copy()
            payload.setdefault("chunk", idx_local)
            uid = str(uuid.uuid4())
            rows.append((uid, doc, json.dumps(payload), time.time()))
            if self.ann_space == "ip":
                vec = np.array(self._normalize(emb), dtype="float32")
            else:
                vec = np.array(emb, dtype="float32")
            vecs.append(vec)

        with self._conn_lock:
            idxs = []
            with self._conn() as conn:
                cur = conn.cursor()
                # insert rows one-by-one to capture lastrowid (the assigned idx)
                for uid, doc, md_json, created in rows:
                    cur.execute(f"INSERT INTO {self.table_name} (id, document, metadata, created_at) VALUES (?,?,?,?)", (uid, doc, md_json, created))
                    idxs.append(cur.lastrowid)
                conn.commit()
            if vecs:
                arr = np.vstack(vecs)
                ids = np.array(idxs, dtype="int64")
                self.index.add_items(arr, ids)
                self.index.set_ef(self.ann_ef)
                try:
                    self.index.save_index(self.index_path)
                except Exception:
                    pass

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        emb = self.embedder.embed_batch([query])[0]
        if len(emb) != self.vector_dim:
            raise ValueError(f"Query embedding dim {len(emb)} does not match configured vector_dim {self.vector_dim}.")

        if self.ann_space == "ip":
            q = np.array(self._normalize(emb), dtype="float32")
        else:
            q = np.array(emb, dtype="float32")

        # Ensure index has elements and adjust k to available elements.
        try:
            n = self.index.get_current_count()
        except Exception:
            n = 0
        if n == 0:
            return []
        k = min(top_k, int(n))
        # ensure ef is sufficiently large for the query
        try:
            self.index.set_ef(max(self.ann_ef, k * 2))
        except Exception:
            pass

        try:
            labels, distances = self.index.knn_query(q, k=k)
        except Exception:
            return []

        items: List[Tuple[str, Dict[str, Any], float]] = []
        ids = labels[0].tolist()
        dists = distances[0].tolist()
        with self._conn() as conn:
            cur = conn.cursor()
            for idx_val, dist in zip(ids, dists):
                try:
                    idx_int = int(idx_val)
                except Exception:
                    # skip non-convertible labels
                    continue
                # sqlite INTEGER is signed 64-bit; skip out-of-range values
                if idx_int < -2**63 or idx_int > 2**63 - 1:
                    continue
                cur.execute(f"SELECT id, document, metadata FROM {self.table_name} WHERE idx=?", (idx_int,))
                row = cur.fetchone()
                if not row:
                    continue
                uid, doc, md = row
                try:
                    payload = json.loads(md)
                except Exception:
                    payload = {}
                score = float(dist)
                if self.ann_space != "ip":
                    score = -score
                items.append((doc or "", payload or {}, score))
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

    def answer_query(self, query: str, top_k: int = 5, ollama_url: Optional[str] = None, ollama_model: Optional[str] = None, system_prompt: Optional[str] = None, user_prompt_template: Optional[str] = None) -> Any:
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
    print("vector_store_sqlite.py: SQLite + hnswlib-backed helper. Import VectorStore in your code.")
