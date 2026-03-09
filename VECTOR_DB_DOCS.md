# 向量資料庫（SQLite + hnswlib）使用說明

這份文件說明專案目前的向量庫設計：使用本地 SQLite 儲存文件與 metadata，並用 `hnswlib` 儲存/查詢向量索引；嵌入來自 Ollama 或 `MockEmbeddings`（測試）。內容涵蓋儲存、檢索與檢視工具。

重要變更：本專案採用「一個 JSON / 一段文字 = 一筆資料」的方式儲存（不做 chunk）。

目錄
- 檔案與工具
- 資料結構（schema）
- 儲存流程
- 檢索與組裝 context
- 常用環境變數
- 查看資料
- 注意事項

**檔案與工具**
- `vector_store_sqlite.py`：SQLite + hnswlib 封裝，提供 `store_response()`、`retrieve()`、`build_context()`、`answer_query()`。
- `scripts/alpha_fetch.py`：抓取 Alpha Vantage 後呼叫 `VectorStore.store_response()` 儲存回應。
- `scripts/view_vectors.py`：列出 SQLite 內容的小工具。

**資料結構（schema）**
- `id`：UUID（在 `store_response` 內產生）。
- `idx`：整數索引對應 hnswlib element id。
- `document`：chunk 文字。
- `metadata`：JSON 字串，包含 `chunk` 與其他 metadata（`source`, `function`, `symbol`, `tickers`, `timestamp`, `type` 等）。
- `created_at`：建立時間。

**儲存流程（store_response / store_json）**
1. 直接以「整段文字 / 整個 JSON」作為 document（不切 chunk）。
2. 呼叫 Ollama embed 取得向量（預設模型 `nomic-embed-text`），或在測試中使用 `MockEmbeddings`。
3. 建立 SQLite 資料表與 hnswlib 索引（若不存在）。
4. 寫入 `document/metadata` 到 SQLite，並把向量加入 hnswlib 索引。

**維度一致性（重要）**
- `VECTOR_DIM` 必須等於嵌入模型輸出維度（`nomic-embed-text` 為 768）。
- 若維度不同，會拒絕寫入並提示錯誤。
- 建議：每種嵌入模型使用不同的 table 名稱（例如 `api_calls_nomic` vs `api_calls_minilm`）。

**檢索與組裝 context**
- `retrieve(query, top_k=5)`: 對 query 做嵌入並在 SQLite + hnswlib 搜尋，回傳 `(document, metadata, score)` 列表。
- `build_context(items)`: 將檢索結果轉成可直接放進 prompt 的文字區塊。
- `answer_query(...)`: 將檢索結果拼成 prompt，呼叫 Ollama completion API，回傳模型輸出。

**Agent Tool：rag_search**
模型可以透過 tool call 取得參考資料：
```json
{ "tool_call": { "tool": "rag_search", "query": "TSLA earnings", "symbol": "TSLA", "types": ["tool_result","stage_output"], "days": 30, "top_k": 5 } }
```
回傳包含 `hits`（精簡列表）與 `context`（可直接拼到 prompt）。

**常用環境變數**
- `SQLITE_PATH`: SQLite 檔案路徑（預設 `vector_store.sqlite`）
- `SQLITE_TABLE`: 資料表名稱（預設 `api_calls`）
- `VECTOR_INDEX_PATH`: hnswlib 索引檔案路徑（預設 `vector_index.bin`）
- `VECTOR_DIM`: 向量維度（預設 768）
- `ANN_INDEX_SPACE`: `ip`（內積，需正規化）或 `l2`
- `ANN_EF` / `ANN_M`: hnswlib 建構與查詢參數
- `VECTOR_FORCE_MOCK_EMBED`: 強制使用 MockEmbeddings（測試用）
- `VECTOR_ALLOW_MOCK_FALLBACK`: 當 Ollama embedding 失敗時，是否允許自動改用 MockEmbeddings（預設關閉）
- `OLLAMA_EMBED_URL`: Ollama 嵌入端點
- `RAG_IS_TEST`: 寫入資料時自動加上 `metadata.is_test=true`（測試資料標記）
- `RAG_INCLUDE_TEST`: 檢索時允許回傳 `is_test=true` 的資料（預設不回傳）

**查看資料**
- 列出向量：
```
python scripts/view_vectors.py --sqlite ./vector_store.sqlite --table api_calls --limit 200
```
- 依 metadata 過濾：
```
python scripts/view_vectors.py --sqlite ./vector_store.sqlite --filter-key type --filter-value api_response
```

**注意事項**
- 向量數量變大時，建議啟用 `ivfflat` 或 `hnsw` 索引。
- `ivfflat` 需要 `ANALYZE` 後才有較佳查詢品質。
- 若改嵌入模型，請更新 `VECTOR_DIM` 或改表名。

**檢索（retrieve）與過濾**
`retrieve()` 支援以 metadata 過濾（例如 `symbol/types/stage/source/run_id`）以及時間範圍（`min_created_at`）。

預設情況下，`is_test=true` 的資料不會被回傳；可用 `include_test=True` 或設定 `RAG_INCLUDE_TEST=1` 開啟。

