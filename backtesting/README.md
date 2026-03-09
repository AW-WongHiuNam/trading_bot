# Backtesting System (Phase-by-Phase)

這個資料夾是給你的 trading-agent JSON 做回測，重點是：
- 可重現（同一份資料、同一參數，結果一致）
- 防作弊（anti-leakage）
- 明確限制資料來源（允許 news，禁止 price fetch）

---

## 1) 目前已實作（MVP）

### `backtesting/data.py`
- 讀取歷史價格 CSV（支援你現在的 `Date,Close/Last,Volume,Open,High,Low` 格式）
- 依 `target_date` 切出 snapshot（例如過去 252 天）
- 將 snapshot 輸出成新 CSV（給 agent 或策略特徵工程用）
- 讀取 agent JSON（同時支援：
  - API wrapper 格式：`{ticker,target_date,result}`
  - raw flow 格式：`{analysts,...,trader_proposal,...}`）

### `backtesting/sanity.py`
Sanity checks（現在就會擋）：
- 必要欄位存在：`trader_proposal`, `manager_decision`, `risk`, `trace`
- 時間防洩漏：所有 timestamp 不得晚於 `target_date`
- Tool policy：
  - 禁止 `alpha_fetch` 的 price 類函數（`GLOBAL_QUOTE`, `TIME_SERIES_*`）
  - 允許 `NEWS_SENTIMENT`（可透過參數關閉）

### `backtesting/signal.py`
- 把 agent JSON 轉成 `TradeSignal`
- 內建 gate：
  - `manager_decision=require_manual` → 強制 `NO_TRADE`
  - `risk_score >= 80` → 強制 `NO_TRADE`

### `backtesting/engine.py`
- 單筆訊號模擬：
  - 在 `target_date` 後下一根 bar 開盤入場
  - 持有 `N` 天後收盤出場
  - 支援手續費與滑點（bps）
- 輸出基本績效：交易數、勝率、平均淨報酬、總淨報酬、期末 equity

### `backtesting/runner.py`
- CLI 一鍵執行：
  - 載入 JSON + CSV
  - 產生 snapshot
  - 跑 sanity checks
  - 轉交易訊號 + 回測
  - 產出報告 JSON 到 `outputs/backtesting/`

---

## 2) 你的核心要求如何落地

### A. 「完整 CSV，snapshot 一部分給 agent」
已支援：
- 用 `--lookback-days 252` 代表只給 agent 看過去一年（交易日）資料
- 用 `--snapshot-out` 輸出切片 CSV

範例：
```bash
python -m backtesting.runner \
  --result-json outputs/TSLA/flow_output_langchain_20260220_103656.json \
  --price-csv stock_price/HistoricalData_TSLA.csv \
  --target-date 2026-02-12 \
  --lookback-days 252 \
  --snapshot-out backtesting/snapshots/TSLA_2026-02-12.csv
```

### B. 「檔案中不能有 future data」
已支援：
- 會檢查 `analysts.*.timestamp`, `manager_decision.timestamp`, `trader_proposal.timestamp`, `trace[*].at`
- 任何晚於 `target_date` 的 timestamp 會直接 fail

### C. 「model 不可以 call alpha_fetch 取得 stock price data，但可以得到 news data」
已支援（預設嚴格）：
- 預設禁止 price 函數（`GLOBAL_QUOTE`, `TIME_SERIES_*`）
- 允許 `NEWS_SENTIMENT`
- 如果想暫時放寬，才加 `--allow-price-fetch`

另外，backtesting 情境下當 `NEWS_SENTIMENT` 帶 `as_of_date`：
- 會先查 DB cache（同 ticker + 同 as_of_date）
- DB 無命中先 call API
- 回傳前會再過濾一次，確保不包含 `as_of_date` 之後的新聞（防 future leak）

---

## 3) 建議執行流程（Phase by Phase）

## Phase 1 — Data Contract & Snapshot
目標：確保每次回測輸入都一致
- [x] 固定 CSV schema
- [x] 固定 JSON schema（允許兩種格式）
- [x] snapshot 只保留 `target_date` 以前資料

## Phase 2 — Leakage Guardrails
目標：把作弊路徑先封掉
- [x] 時間戳 future-data 檢查
- [x] alpha price-fetch 禁止策略
- [x] news-fetch 可配置

## Phase 3 — Signal Mapping
目標：把 LLM decision 轉成 deterministic rule
- [x] `trader_proposal.side` 正規化
- [x] `manager_decision` gate
- [x] `risk_score` gate

## Phase 4 — Execution Simulator
目標：最小可用回測績效
- [x] 入場：next bar open
- [x] 出場：N days close
- [x] 成本：fee + slippage
- [x] 指標：win rate / return / equity

## Phase 5 — 後續擴充（你下一步可以做）
- [ ] 多訊號連續回測（portfolio timeline）
- [ ] walk-forward（rolling windows）
- [ ] baseline 對照（Buy&Hold、均線）
- [ ] 更完整風險指標（Sharpe, MDD, Calmar）
- [ ] CI gate：sanity fail 就拒絕產出結果

---

## 4) 快速命令

### 嚴格模式（建議，預設）
```bash
python -m backtesting.runner \
  --result-json outputs/TSLA/flow_output_langchain_20260220_103656.json \
  --price-csv stock_price/HistoricalData_TSLA.csv \
  --target-date 2026-02-12
```

### Backtest 決策模式（當 manager=require_manual 時仍可產生回測信號）
```bash
python -m backtesting.runner \
  --result-json outputs/TSLA/flow_output_langchain_20260220_103656.json \
  --price-csv stock_price/HistoricalData_TSLA.csv \
  --target-date 2026-02-12 \
  --decision-policy backtest
```

### Auto-trade 決策模式（完全自動，不做人手確認 gate）
```bash
python -m backtesting.runner \
  --result-json outputs/TSLA/flow_output_langchain_20260220_103656.json \
  --price-csv stock_price/HistoricalData_TSLA.csv \
  --target-date 2026-02-12 \
  --decision-policy auto
```

`decision-policy=auto` 規則：
- 忽略 `manager_decision=require_manual` 的阻擋
- 優先使用 trader 的方向（BUY/SELL）；若 trader 冇方向就用 bull/bear 信心差補方向
- 用風險分數動態縮放倉位（高風險會大幅降倉，極高風險可直接不交易）

### 放寬模式（只用於 debug）
```bash
python -m backtesting.runner \
  --result-json outputs/TSLA/flow_output_langchain_20260220_103656.json \
  --price-csv stock_price/HistoricalData_TSLA.csv \
  --target-date 2026-02-12 \
  --allow-price-fetch
```

### 指定輸出
```bash
python -m backtesting.runner \
  --result-json outputs/TSLA/flow_output_langchain_20260220_103656.json \
  --price-csv stock_price/HistoricalData_TSLA.csv \
  --target-date 2026-02-12 \
  --snapshot-out backtesting/snapshots/TSLA_2026-02-12.csv \
  --report-out backtesting/reports/tsla_report.json
```

---

## 5) 你給的這份 JSON 會看到什麼
- 因為 `trace` 裡有 `GLOBAL_QUOTE`，在嚴格模式下會被判定為 policy violation（sanity fail）
- 這是刻意設計，避免模型在回測時偷拿價格資料
- 如果你只是要驗證引擎可執行，可用 `--allow-price-fetch` 暫時放寬

---

## 6) Agent 情境化交易（你剛要求的）

現在 orchestration 已支援以下交易情境欄位，會注入 `SCENARIO_CONTEXT` 給 agent：
- `now_utc`
- `target_date`
- `account_cash`
- `account_shares`

此外，`alpha_fetch` 現在支援 `params.as_of_date`。當呼叫 `GLOBAL_QUOTE` 且帶 `as_of_date` 時，系統會改用 `TIME_SERIES_DAILY` 回推該日價格，不再默認用「即日」。

---

## 7) Phase 5 — 批次回測

新增：`backtesting/phase5_batch.py`

用途：
- 一次跑整個 `outputs/TSLA/*.json`
- 輸出 aggregate：
  - 檔案數
  - sanity 通過/失敗數
  - executed trades
  - win rate
  - total net return
  - top sanity errors

範例：
```bash
python -m backtesting.phase5_batch \
  --results-dir outputs/TSLA \
  --price-csv stock_price/HistoricalData_TSLA.csv \
  --target-date 2026-02-12 \
  --decision-policy auto \
  --report-out backtesting/reports/phase5_tsla.json
```

---

## 8) 指定區間逐日回測（你現在要的）

新增：`backtesting/day_by_day.py`

功能：
- 你指定 `start-date` 到 `end-date`
- 每個交易日都跑一次 agent orchestration
- 每天都會存完整輸出 JSON（包含 `bull_rounds`, `bear_rounds`, `discussion`, `trace`）
- 另外輸出 `daily_summary.csv` 與總報告 `report.json`

範例（你提的區間）：
```bash
python -m backtesting.day_by_day \
  --ticker TSLA \
  --start-date 2025-10-01 \
  --end-date 2026-01-31 \
  --price-csv stock_price/HistoricalData_TSLA.csv \
  --initial-cash 100000 \
  --initial-shares 0 \
  --decision-policy auto \
  --report-out outputs/backtesting/day_by_day_tsla_2025-10_to_2026-01.json
```

產物位置：
- 每日完整討論 JSON：`outputs/backtesting/day_by_day_<...>/daily_outputs/*.json`
- 每日摘要：`outputs/backtesting/day_by_day_<...>/daily_summary.csv`
- 區間總報告：`.../report.json` 或你指定的 `--report-out`

> 注意：逐日模式會呼叫 LLM 很多次（每個交易日一次完整 flow），執行時間會明顯較長。

---

## 9) Output JSON 位置地圖（整理）

### Agent flow 輸出（每次 orchestration）
- 最新：`flow_output_langchain.json`
- 歸檔：`outputs/<TICKER>/flow_output_langchain_<timestamp>.json`

### 單檔 backtest (`backtesting.runner`)
- 若有 `--report-out`：寫到你指定路徑
- 若無 `--report-out`：`outputs/backtesting/single_run_<timestamp>/report.json`

### Phase 5 批次 (`backtesting.phase5_batch`)
- 若有 `--report-out`：寫到你指定路徑
- 若無 `--report-out`：`outputs/backtesting/phase5_run_<timestamp>/report.json`

---

## 10) 如何看到執行 process（即時進度）

現在 `day_by_day` 與 `phase5_batch` 已內建即時進度輸出：
- `day_by_day` 會顯示：`目前第幾日 / 總日數 / 百分比 / 當日結果（sanity, side, executed, cash）`
- `phase5_batch` 會顯示：`目前第幾個檔案 / 總檔數 / 百分比 / 該檔結果（sanity, executed）`

範例輸出：
```text
[day_by_day] 12/84 (14.3%) running TSLA @ 2025-10-17
[day_by_day] done 2025-10-17 | sanity=True | side=BUY | executed=True | cash=101245.33
```

另外你可同時查看產物：
- 每日完整 JSON：`outputs/backtesting/day_by_day_<...>/daily_outputs/`
- 每日摘要 CSV：`outputs/backtesting/day_by_day_<...>/daily_summary.csv`

- 若無 `--out-dir`：`outputs/backtesting/day_by_day_<ticker>_<start>_to_<end>_<timestamp>/`

### Day-by-day 區間 (`backtesting.day_by_day`)
- 若有 `--out-dir`：用你指定資料夾
- 若無 `--out-dir`：`outputs/backtesting/day_by_day_<ticker>_<start>_to_<end>_<timestamp>/`
  - `daily_outputs/*.json`（每日完整討論）
  - `daily_summary.csv`
  - `report.json`（或你指定的 `--report-out`）
