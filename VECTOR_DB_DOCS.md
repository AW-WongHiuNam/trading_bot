# 向量資料庫（Qdrant, 本機檔案模式）使用說明

這份文件說明專案目前的向量庫設計：使用 Qdrant 的內嵌（embedded）客戶端，將資料存成本機資料夾；嵌入來自 Ollama。內容涵蓋儲存、檢索與檢視工具。

目錄
- 檔案與工具
- 資料結構（schema）
- 儲存流程
- 檢索與組裝 context
- 常用環境變數
- 查看資料
- 注意事項

**檔案與工具**
- `vector_store.py`：Qdrant 封裝，提供 `store_response()`、`retrieve()`、`build_context()`、`answer_query()`。
- `alpha_fetch.py`：抓取 Alpha Vantage 後呼叫 `VectorStore.store_response()` 儲存回應。
- `view_vectors.py`：列出 Qdrant 內容的小工具。
- `test_chroma_debug.py`：已改成 Qdrant 內嵌 quickstart（新增/搜尋示例）。

**資料結構（schema）**
- `id`：UUID（在 `store_response` 內產生）。
- `vector`：浮點向量，來自 Ollama embed endpoint。
- `payload`：包含
  - `document`：chunk 文字
  - `chunk`：chunk 序號
  - 其他 metadata：`source`, `function`, `symbol`, `tickers`, `timestamp`, `type` 等

**儲存流程（store_response）**
1. 將文字切成 chunks（預設 `chunk_size=2000`, `overlap=400`）。
2. 呼叫 Ollama embed 取得向量（預設模型 `nomic-embed-text`）。
3. 若 collection 不存在，以向量維度建立 Qdrant collection（cosine）。
4. 以 `upsert` 將 (id, vector, payload) 寫入。

**檢索與組裝 context**
- `retrieve(query, top_k=5)`: 對 query 做嵌入並在 Qdrant 搜尋，回傳 `(document, metadata, score)` 列表。
- `build_context(items)`: 將檢索結果轉成可直接放進 prompt 的文字區塊。
- `answer_query(...)`: 將檢索結果拼成 prompt，呼叫 Ollama completion API，回傳模型輸出。

**常用環境變數**
- `QDRANT_PATH`: Qdrant 本機資料夾（預設 `qdrant_db`）。
- `OLLAMA_EMBED_URL`: Ollama 嵌入端點（預設 `http://127.0.0.1:11434/embed`）。
- `QDRANT_FORCE_MOCK_EMBED`: 設為任意值可強制使用內建 MockEmbeddings（不需要 Ollama）。

**查看資料**
- 列出向量：
```
python view_vectors.py --path qdrant_db --collection api_calls --limit 200
```
- 依 metadata 過濾：
```
python view_vectors.py --filter-key type --filter-value api_response
```

**注意事項**
- 目前使用本機檔案模式，資料夾應納入備份/排除版本控制（如需要）。
- 若更換嵌入模型導致維度不同，需重建 collection（Qdrant 需要固定維度）。
- 向量數量變大時，可改用 Qdrant 伺服器模式或雲端服務；封裝接口可沿用。
