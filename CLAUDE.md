# CLAUDE.md — forecast-teller 工作規則

台灣流感（主線）、登革熱預警（`dengue_ewarn/`）、腸病毒（`ev_forecast/`）三條工作線，以 Google TimesFM 3.0 零樣本預測，GitHub Pages 發布（`main:/docs`）。
**接手或不確定狀態時，先讀 `HANDOFF.md`**（狀態快照、決定事項、SOP、踩過的坑），再讀 `README.md` 與各 `PLAN.md` 的狀態章節。

## 溝通
- 回覆、文件、commit 說明的敘述用繁體中文；使用者是公衛 / 流病研究者，熟悉監測資料定義。
- 新聞稿或網路文章得到的閾值、規則、個案定義一律標「待確認」並向使用者確認，不直接寫死。

## 環境與執行
- `source .venv/bin/activate`（Python 3.11、`timesfm[mlx]` 3.0.2、MLX 後端，Apple M1 16 GB）。
- 檢查：`python tests/test_basics.py`（無輸出即通過）。
- 流感流程：`scripts/fetch_resp_lab.py` → `scripts/build_panel.py` → `scripts/forecast_now.py --joint nhi_out_ili nhi_er_ili rods_ili --covariates cny holiday` → `scripts/log_forecast.py` → `scripts/evaluate_forecasts.py` → `scripts/build_site.py`；一鍵 `scripts/weekly_update.sh [--push]`。
- 腸病毒：`ev_forecast/scripts/weekly_update.sh [--push]`。登革熱：見 `dengue_ewarn/README.md`。
- 本機預覽站台：`python3 -m http.server 8765 --directory docs`（或 `.claude/launch.json` 的 `docs-site`）。頁面用 fetch，必須走 HTTP。
- 改了網頁要在瀏覽器實際驗證（本機或線上加 `?v=時間戳` 避免快取）。內建瀏覽器對 Chart.js 頁面固定會報 `Cannot convert object to primitive value`，不是頁面錯誤。

## 資料
- `data/` 的流感原始檔是 cp950（Big5）的內部檔，由使用者手動更新，不進 git；`tw_holiday.csv`、`date_week_mapping.csv` 與 `RESP_LAB.csv` 為 UTF-8。`RESP_LAB.csv` 由 `scripts/fetch_resp_lab.py` 從 NIDSS 網站抓，每次抓到的值留存於 `data/resp_lab_history.csv`（進 git）。
- 每週預測值累積在 `outputs/forecast_log/forecast_log.csv`（進 git），不要刪；`evaluate_forecasts.py` 用它評估實際預測表現。
- 疫情週用 `forecast_teller.weeks`；頭尾不完整週由 `panel.detect_edges` 自動剔除，更新後先看 `data_processed/coverage.json`。
- 實驗室 A/B 型資料來源是「實驗室自動通報系統（LARS）」，不是合約實驗室。
- RODS 急診類流感%的流行閾值 11%（`backtest.DEFAULT_THRESHOLDS`）。腸病毒閾值依年記錄在 `ev_forecast/data/thresholds.csv`。
- 回測只用 2016w01–2025w53，暖機 104 週，h = 1–4，分 COVID 前 / 期 / 後三段。

## 版控
- **只 `git add` 明確路徑，禁止 `git add -A` 或 `git add .`**；commit 前 `git status` 確認沒有其他工作線的檔案混入。
- 原始資料與大型中間檔不進 git（見 `.gitignore`）；新增子專案的 `data/` 要一併加進去。
- 工作直接在 `main` 上進行並 push，push 後確認 GitHub Pages 已更新。
- Commit 訊息結尾加：`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`

## 風格
- 圖表依 https://github.com/drhao/epi-dataviz-styleguide ：主色 #739A6D、線 #5D7F58、區間 #B4C9B1、紅色 #BE373C 只用於閾值線、Noto Sans/Serif TC、tabular-nums、僅水平格線、直條圖 Y 軸從零。
- 報告、投影片、網頁的數字從 `outputs/` 與 `docs/data/*.json` 讀取，不手打。
- TimesFM 3.0 權重為非商業授權；談正式部署時要提醒改用 2.5 或另談授權。
