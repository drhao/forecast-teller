# ev_forecast — 台灣腸病毒疫情預測（TimesFM 3.0 zero-shot）

`forecast-teller` 的獨立子專案：以疾管署資料開放平台的腸病毒門診（健保）與急診（RODS）週資料，
用 Google TimesFM 3.0 做每週滾動預測與 2016–2025 回測，並發布到 GitHub Pages（`docs/ev/`）。
共用主專案的核心模組（`src/forecast_teller`：模型包裝、指標、基準、回測引擎、週別與假日共變數）。

- 每週預測頁：https://drhao.github.io/forecast-teller/ev/
- 回測：https://drhao.github.io/forecast-teller/ev/backtest.html
- 報告：https://drhao.github.io/forecast-teller/ev/report.html
- 設計文件與結果：[PLAN.md](PLAN.md)

## 目標指標與規則

- **門急診合計 `ev_oe`** = 健保門診腸病毒就診人次 + RODS 急診腸病毒就診人次。開放資料的健保檔沒有急診，急診部分暫以 RODS 代替（2026-09-19 決定）。
- **流行閾值**由疾管署每年設定，依各年新聞稿記錄在 `data/thresholds.csv`（2016–2025 為 11,000；2026 為 12,000）。
- **流行期規則（暫定）**：單週達閾值即進入；連續 2 週低於閾值即脫離（第 2 週起為非流行期）。實作在 `src/ev_forecast/thresholds.py`。

## 目錄

```
ev_forecast/
├── PLAN.md                 設計、決定事項、回測結果
├── data/                   開放資料 CSV（不進 git）、thresholds.csv、twca_intermediate.pem
├── data_processed/         national_weekly.parquet/csv、age_*_weekly、county_weekly、coverage.json
├── outputs/backtest/       *_forecasts.parquet（不進 git）、*_summary.csv、run_log.txt
├── outputs/latest/         latest_forecast(.csv|_joint.csv) 與圖
├── outputs/design/         規劃階段的先導測試
├── scripts/                refresh_data.sh → build_panel.py → run_backtest.py → forecast_now.py → build_site.py；weekly_update.sh
└── src/ev_forecast/        io.py、panel.py、thresholds.py、covariates.py、site.py
```

## 使用

```bash
cd ev_forecast
scripts/refresh_data.sh                              # 下載開放資料（含 TWCA 中繼憑證）
../.venv/bin/python scripts/build_panel.py           # 週資料面板（需要主專案 data_processed/national_weekly.parquet 提供 RODS 分母）
../.venv/bin/python scripts/run_backtest.py --suite baselines core stats --target ev_oe
../.venv/bin/python scripts/run_backtest.py --suite baselines layer1 layer2 layer3 stats --target ev_out
../.venv/bin/python scripts/forecast_now.py --joint ev_oe ev_out ev_rods --covariates school holiday
../.venv/bin/python scripts/forecast_now.py --targets ev_rods_pct --covariates school holiday
../.venv/bin/python scripts/build_site.py
```

或一次跑完每週流程：`scripts/weekly_update.sh [--push]`。

## 資料來源

| 檔案 | 來源 | 內容 |
|---|---|---|
| `NHI_EnteroviralInfection.csv` | https://data.cdc.gov.tw/dataset/hi-outpatient-emergency-visit-enteroviral-infection | 週 × 門診/住院 × 9 年齡層 × 22 縣市，2016w01 起 |
| `RODS_EnteroviralInfection.csv` | https://data.cdc.gov.tw/dataset/rods-enteroviral-infection | 週 × 5 年齡層 × 22 縣市，2007w01 起（無分母） |
| 主專案 `data_processed/national_weekly.parquet` 的 `rods_total` | `data/RODS_RS.csv` | RODS 急診總人次（急診腸病毒%的分母） |
