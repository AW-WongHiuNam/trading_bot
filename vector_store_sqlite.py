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
        conn = sqlite3.connect(self.sqlite_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        return conn

    def get_latest(
        self,
        *,
        source: Optional[str] = None,
        tool: Optional[str] = None,
        function: Optional[str] = None,
        symbol: Optional[str] = None,
        tickers: Optional[str] = None,
        type: Optional[str] = None,
        meta_equals: Optional[Dict[str, Any]] = None,
        min_created_at: Optional[float] = None,
        max_scan: int = 500,
    ) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """Return the most recent (document, metadata, created_at) matching metadata filters.

        This does a small reverse scan over SQLite rows and filters in Python,
        so it doesn't depend on SQLite JSON extensions.
        """
        max_scan = max(int(max_scan), 1)
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"SELECT document, metadata, created_at FROM {self.table_name} ORDER BY idx DESC LIMIT ?",
                (max_scan,),
            )
            rows = cur.fetchall() or []

        for doc, md_raw, created_at in rows:
            if min_created_at is not None:
                try:
                    if float(created_at) < float(min_created_at):
                        continue
                except Exception:
                    pass

            try:
                md = json.loads(md_raw) if isinstance(md_raw, str) else {}
            except Exception:
                md = {}

            if source is not None and md.get("source") != source:
                continue
            if tool is not None and md.get("tool") != tool:
                continue
            if function is not None and md.get("function") != function:
                continue
            if symbol is not None and (md.get("symbol") or md.get("tickers")) != symbol:
                continue
            if tickers is not None and (md.get("tickers") or md.get("symbol")) != tickers:
                continue
            if type is not None and md.get("type") != type:
                continue

            if meta_equals:
                ok = True
                for k, v in meta_equals.items():
                    if md.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue

            try:
                return (str(doc or ""), dict(md or {}), float(created_at))
            except Exception:
                return (str(doc or ""), dict(md or {}), 0.0)
        return None

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

    def rebuild_index_from_sqlite(self, *, batch_size: int = 32) -> int:
        """Rebuild the ANN index from the SQLite table.

        This is useful if `index_path` has become out-of-sync with the SQLite
        table (e.g. shared index file reused across tables or dim changes).

        Returns the number of vectors indexed.
        """
        # Load all rows first to avoid holding the DB lock while embedding.
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT idx, document FROM {self.table_name} ORDER BY idx")
            rows = [(int(r[0]), (r[1] or "")) for r in cur.fetchall() if r and r[1]]

        space = self.ann_space
        dim = self.vector_dim
        new_index = hnswlib.Index(space=space, dim=dim)
        # Add headroom so future inserts don't immediately exceed capacity.
        headroom = max(int(len(rows) * 2), 1000, 1)
        new_index.init_index(max_elements=headroom, ef_construction=self.ann_ef, M=self.ann_m)
        new_index.set_ef(self.ann_ef)

        total = 0
        i = 0
        while i < len(rows):
            batch = rows[i : i + int(batch_size)]
            ids = [b[0] for b in batch]
            docs = [b[1] for b in batch]
            try:
                embs = self.embedder.embed_batch(docs)
            except Exception:
                import traceback

                traceback.print_exc()
                if not isinstance(self.embedder, MockEmbeddings):
                    self.embedder = MockEmbeddings(dim=self.vector_dim)
                    embs = self.embedder.embed_batch(docs)
                else:
                    raise

            vecs = []
            for emb in embs:
                if self.ann_space == "ip":
                    vecs.append(np.array(self._normalize(emb), dtype="float32"))
                else:
                    vecs.append(np.array(emb, dtype="float32"))

            if vecs:
                arr = np.vstack(vecs)
                new_index.add_items(arr, np.array(ids, dtype="int64"))
                total += len(vecs)
            i += int(batch_size)

        with self._conn_lock:
            self.index = new_index
            try:
                self.index.save_index(self.index_path)
            except Exception:
                pass
        return total

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

    def store_json(self, obj: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a single JSON object as one vector item (no chunking)."""
        try:
            text = json.dumps(obj, ensure_ascii=False)
        except Exception:
            text = str(obj)
        self.store_response(text, metadata=metadata, chunk_size=0, overlap=0)

    def store_response(self, text: Any, metadata: Optional[Dict[str, Any]] = None, chunk_size: int = 0, overlap: int = 0) -> None:
        """Store a single document as one vector item.

        Note: chunking is intentionally disabled (one JSON/text == one row).
        The chunk_size/overlap arguments are accepted for backward compatibility
        but ignored.
        """
        metadata = metadata or {}
        if not isinstance(text, str):
            try:
                text = json.dumps(text, ensure_ascii=False)
            except Exception:
                text = str(text)

        doc = (text or "").strip()
        if not doc:
            return

        payload = metadata.copy()
        payload.setdefault("chunk", 0)
        payload.setdefault("is_test", _env_truthy("RAG_IS_TEST"))

        try:
            embeddings = self.embedder.embed_batch([doc])
            print("VectorStore.store_response: got 1 embedding")
        except Exception:
            import traceback

            print("VectorStore.store_response: embed_batch raised an exception:")
            traceback.print_exc()
            if not isinstance(self.embedder, MockEmbeddings):
                try:
                    print("VectorStore.store_response: falling back to MockEmbeddings and retrying")
                    self.embedder = MockEmbeddings(dim=self.vector_dim)
                    embeddings = self.embedder.embed_batch([doc])
                    print("VectorStore.store_response: got 1 mock embedding")
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

        emb = embeddings[0]
        uid = str(uuid.uuid4())
        created = time.time()
        row = (uid, doc, json.dumps(payload, ensure_ascii=False), created)
        if self.ann_space == "ip":
            vec = np.array(self._normalize(emb), dtype="float32")
        else:
            vec = np.array(emb, dtype="float32")

        with self._conn_lock:
            assigned_idx = None
            last_exc: Exception | None = None
            for attempt in range(5):
                try:
                    with self._conn() as conn:
                        cur = conn.cursor()
                        cur.execute(
                            f"INSERT INTO {self.table_name} (id, document, metadata, created_at) VALUES (?,?,?,?)",
                            row,
                        )
                        assigned_idx = cur.lastrowid
                        conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    last_exc = e
                    # Common under concurrent writers / multi-instance stores.
                    if "locked" in str(e).lower():
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    raise

            if assigned_idx is None:
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("Failed to insert vector store row")
            arr = np.vstack([vec])
            ids = np.array([assigned_idx], dtype="int64")
            # Ensure index has capacity; hnswlib enforces a fixed max_elements.
            try:
                cur_n = int(self.index.get_current_count())
                max_n = int(self.index.get_max_elements())
                if cur_n + 1 > max_n:
                    new_max = max(max_n * 2, cur_n + 1, 1000)
                    self.index.resize_index(int(new_max))
            except Exception:
                pass
            self.index.add_items(arr, ids)
            self.index.set_ef(self.ann_ef)
            try:
                self.index.save_index(self.index_path)
            except Exception:
                pass

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        symbol: Optional[str] = None,
        types: Optional[List[str]] = None,
        stage: Optional[str] = None,
        source: Optional[str] = None,
        run_id: Optional[str] = None,
        include_test: Optional[bool] = None,
        min_created_at: Optional[float] = None,
        candidate_k: Optional[int] = None,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
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
        if include_test is None:
            include_test = _env_truthy("RAG_INCLUDE_TEST")
        if candidate_k is None:
            candidate_k = max(int(top_k) * 10, 50)

        k = min(int(candidate_k), int(n))
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
                cur.execute(f"SELECT id, document, metadata, created_at FROM {self.table_name} WHERE idx=?", (idx_int,))
                row = cur.fetchone()
                if not row:
                    continue
                uid, doc, md, created_at = row
                try:
                    payload = json.loads(md)
                except Exception:
                    payload = {}

                # default missing to False
                is_test = bool(payload.get("is_test", False))
                if not include_test and is_test:
                    continue
                if symbol is not None and (payload.get("symbol") or payload.get("tickers")) != symbol:
                    continue
                if types is not None and payload.get("type") not in set(types):
                    continue
                if stage is not None and payload.get("stage") != stage:
                    continue
                if source is not None and payload.get("source") != source:
                    continue
                if run_id is not None and payload.get("run_id") != run_id:
                    continue
                if min_created_at is not None:
                    try:
                        if float(created_at) < float(min_created_at):
                            continue
                    except Exception:
                        pass

                score = float(dist)
                if self.ann_space != "ip":
                    score = -score
                items.append((doc or "", payload or {}, score))
                if len(items) >= int(top_k):
                    break
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
