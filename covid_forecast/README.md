# covid_forecast — 台灣新冠（COVID-19）疫情預測（TimesFM 3.0 zero-shot）

`forecast-teller` 的獨立子專案：以 NIDSS 傳染病統計資料查詢系統的新冠健保門急診就診人次（疾管署新聞稿指標）、
疾管署資料開放平台的新冠門診／住院（健保）與急診（RODS）週資料，用 Google TimesFM 3.0 做每週滾動預測與
2024–2026 回測，並發布到 GitHub Pages（`docs/covid/`）。共用主專案的核心模組（`src/forecast_teller`：
模型包裝、指標、基準、回測引擎、週別與假日共變數、NIDSS 查詢客戶端 `nidss.py`）。

- 每週預測頁：https://drhao.github.io/forecast-teller/covid/
- 回測：https://drhao.github.io/forecast-teller/covid/backtest.html
- 報告：https://drhao.github.io/forecast-teller/covid/report.html
- 設計文件、先導測試、決定事項與結果：[PLAN.md](PLAN.md)

## 目標指標與規則

- **主指標 `covid_oe`**：全國新冠門急診就診人次 = NIDSS「健保門急診就診」查詢的門診 + 急診（2021w27 起）。
  開放資料的健保檔沒有急診、RODS 檔沒有分母，因此另以 `covid_oe_alt` = 開放資料門診 + RODS 急診作對照（最接近新聞稿首報值）。
- **流行閾值**：疾管署新聞稿只稱「達流行閾值，進入流行期」，數值未公布（`data/thresholds.csv` 記錄狀態）。
  取得前以**暫行規則**判定（`src/covid_forecast/thresholds.py`，使用者決定 2026-10-13）：連續 3 週上升且達本波低點 2 倍即進入；
  進入後連續 2 週下降且低於本波峰值一半即脫離。網頁不畫閾值線，改報「下週續升機率」與「達暫行進入／脫離水準的機率」。
- **回測**：context 自 2021w27（含 2022 大流行；RODS 相關設定自 2022w20），起點 2024w01 起至最新完整週，h = 1–4，依年與全期報告。

## 目錄

```
covid_forecast/
├── PLAN.md                 規劃、資料盤點、先導測試、決定事項、回測結果
├── data/                   開放資料 CSV 與 NIDSS 抓取檔（不進 git）；thresholds.csv、variants.csv（人工）、
│                           nidss_visits_history.csv、severe_history.csv（每次抓到的首報值，進 git）、twca_intermediate.pem
├── data_processed/         national_weekly.parquet/csv、age_nhi/age3/age_rods/age_nidss、county、region、coverage.json
├── outputs/backtest/       *_forecasts.parquet（不進 git）、*_summary.csv
├── outputs/latest/         latest_forecast(.csv|_joint.csv) 與圖
├── outputs/forecast_log/   forecast_log.csv、evaluation*.csv（主線 log_forecast.py / evaluate_forecasts.py --project covid_forecast）
├── outputs/design/         先導回測（2026-10-13）
├── scripts/                refresh_data.sh → fetch_nidss_visits.py → fetch_severe.py → build_panel.py → run_backtest.py →
│                           forecast_now.py → build_site.py；weekly_update.sh [--push]
└── src/covid_forecast/     io.py、panel.py、thresholds.py、site.py
```

## 使用

```bash
cd covid_forecast
scripts/refresh_data.sh                                   # 開放資料兩檔（TWCA 中繼憑證）
../.venv/bin/python scripts/fetch_nidss_visits.py         # NIDSS 門急診：全國 + 6 區 × 9 年齡層（約 40 秒）
../.venv/bin/python scripts/fetch_severe.py               # NIDSS 新冠併發重症週數
../.venv/bin/python scripts/build_panel.py                # 面板（需要主專案 data_processed/national_weekly.parquet 提供 rods_total、RESP_LAB）
../.venv/bin/python scripts/run_backtest.py --suite baselines stats layer1 layer2 layer3 core --target covid_oe
../.venv/bin/python scripts/run_backtest.py --suite baselines stats core --target covid_oe_alt covid_out
../.venv/bin/python scripts/forecast_now.py --targets covid_oe covid_rods_pct covid_inp covid_oe_alt --covariates cny holiday   # 主指標單變量（回測最佳）
../.venv/bin/python scripts/forecast_now.py --joint covid_out covid_rods --covariates cny holiday                            # 門診 + RODS 聯合
../.venv/bin/python scripts/build_site.py
```

或一次跑完每週流程：`scripts/weekly_update.sh [--push]`（先跑主線 `scripts/weekly_update.sh` 更新 RODS 分母）。

## 資料來源

| 檔案 | 來源 | 內容 |
|---|---|---|
| `nidss_visits.csv` | https://nidss.cdc.gov.tw/Cdcwnh/CDCWNH01?disease=11 （`forecast_teller.nidss`） | 週 × 全國/6 區 × 9 年齡層 × {門診, 門診+急診}，2021w01 起（有值自 2021w27） |
| `NHI_COVID-19.csv` | https://data.cdc.gov.tw/dataset/hi-visit-covid-19 | 週 × 門診/住院 × 9 年齡層 × 22 縣市，2021w27 起 |
| `RODS_COVID-19.csv` | https://data.cdc.gov.tw/dataset/rods-covid-19 | 週 × 5 年齡層 × 22 縣市，2022w20 起（無分母） |
| `severe_weekly.csv` | https://nidss.cdc.gov.tw/nndss/disease?id=19SC | 新冠併發重症週確定病例數（發病週），2024w34 起 |
| 主專案 `data_processed/national_weekly.parquet` | `data/RODS_RS.csv`、`data/RESP_LAB.csv` | RODS 急診總人次（分母）、RESP_LAB SARS-CoV-2 陽性數 |
| `variants.csv` | 疾管署新聞稿（人工填） | 主流變異株，只用於敘事 |
