# forecast-teller

以 Google **TimesFM 3.0** 對台灣流感監測資料做零樣本（zero-shot）預測。規劃文件見 [PLAN_timesfm3_flu.md](PLAN_timesfm3_flu.md)。

## 環境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "timesfm[mlx]" pandas pyarrow matplotlib statsforecast tabulate
```

Apple Silicon 用 MLX 後端（預設）。PyTorch 版：`pip install "timesfm[torch]"`，並在 `TimesFM3Model(backend="torch")` 使用。
第一次載入模型會從 Hugging Face 下載約 1.3 GB 權重到 `~/.cache/huggingface/`。

> TimesFM 3.0 權重授權為非商業、非生產用途；程式碼為 Apache-2.0。

## 流程

```bash
python scripts/build_panel.py          # data/ (Big5) → data_processed/ 面板 + qa_report.md
python scripts/smoke_test.py           # 載入模型並對全國類流感門診人次做 4 週預測
python scripts/run_backtest.py --target nhi_out_ili --suite layer1   # 滾動起點回測（2016–2025）
python scripts/run_backtest.py --target nhi_out_ili --suite layer2   # 加入已知未來共變數（layer2b 隔離假日效果）
python scripts/run_backtest.py --target nhi_out_ili --suite layer3   # 過去共變數與多變量聯合
python scripts/run_backtest.py --target nhi_out_ili --suite layer4   # 多變量 + 假日共變數（最佳）
python scripts/run_backtest.py --target nhi_out_ili --suite stats    # AutoETS / Theta 基準（CPU，約 3 分鐘）
python scripts/run_group_backtest.py --by county --indicator nhi_out_ili   # 22 縣市：單變量 vs 聯合
python scripts/make_report.py --tags nhi_out_ili_layer1 nhi_out_ili_layer2 nhi_out_ili_layer2b nhi_out_ili_layer3 nhi_out_ili_layer4 nhi_out_ili_stats --title 全國類流感門診人次 --name nhi_out_ili
python scripts/forecast_now.py --joint nhi_out_ili nhi_er_ili rods_ili --covariates cny holiday   # 即時預測（最佳設定；outputs/latest/）
```

## 目錄

| 路徑 | 內容 |
|---|---|
| `data/` | 原始 CSV（健保、RODS、NIDDS、合約實驗室；Big5 編碼） |
| `data_processed/` | `national_weekly.*`、`county_weekly.parquet`、`age_weekly.parquet`、`coverage.json`、`qa_report.md` |
| `src/forecast_teller/` | `io` 讀檔、`weeks` 疫情週、`panel` 面板、`covariates` 共變數、`model_timesfm3` 模型包裝、`baselines`、`metrics`（WIS）、`backtest`、`report` |
| `scripts/` | 可執行流程 |
| `outputs/backtest/` | 每個設定的逐起點預測（parquet）、summary、`*_REPORT.md` |
| `outputs/figures/`、`outputs/latest/` | 圖與即時預測 |

## 回測設計

- 資料窗 2016w01–2025w53（522 週），暖機 2016–2017，起點 2018w01 起每週一個，h = 1–4。
- 指標：WIS、MAE、MAPE、WAPE、MASE、80%/60% 區間涵蓋率，以及命中率（方向命中率、三分類命中率、±10% 容忍帶、閾值命中率/誤報率/準確率；`scripts/add_hit_rates.py` 可對既有結果補算，`make_report.py --threshold` 指定流行閾值；RODS 急診類流感%預設用流行閾值 10%）；基準：季節性 naive、last value、MA3（前 3 週移動平均，2 週以上遞迴代入）、AutoETS、Theta。
- 結果分 COVID 前（2018–2019）、COVID 期（2020–2022）、COVID 後（2023–2025）三段報告。

## 回測結論（2026-09-17）

最佳設定為四變量聯合（門診 + 急診 + RODS + 重症）加春節旗標與假日天數共變數：COVID 後 1–4 週的 WIS 比 last-value naive 低 32% / 29% / 23% / 22%，80% 區間涵蓋率 0.82。無共變數的 zero-shot TimesFM 已比 naive 低 16%、比 AutoETS 低 12%。詳見 `PLAN_timesfm3_flu.md` 第 12 節與 `outputs/backtest/*_REPORT.md`。

## Dashboard 與 GitHub Pages

`docs/` 是可直接發布到 GitHub Pages 的靜態站台（Chart.js，依《疫情資料視覺化指引》配色與規範）：

| 頁面 | 內容 |
|---|---|
| `docs/index.html` | 每週預測 dashboard：四項指標的 4 週扇形圖（60%/80% 區間）、KPI、分位數表 |
| `docs/backtest.html` | 回測 dashboard：排行榜（可排序）、WIS/命中率/涵蓋率依 horizon、預測 vs 實際、各年、縣市與年齡層 |
| `docs/report.html` | 一頁式解讀與評估報告（可列印） |
| `docs/data/*.json` | 由 `scripts/build_site.py` 從 `outputs/` 產生 |

本機預覽（頁面用 fetch 讀 JSON，需走 HTTP）：

```bash
python3 -m http.server 8765 --directory docs
```

每週更新一鍵流程：`scripts/weekly_update.sh`（加 `--push` 會 commit 並 push `docs/`）。

發布到 GitHub Pages（首次）：
1. `git init && git add -A && git commit -m "init"`，在 GitHub 建 repo 並 `git remote add origin ... && git push -u origin main`。`data/*.csv` 已在 `.gitignore`（RODS 檔 196 MB 超過 GitHub 單檔上限），只有假日與週對應表會進版控。
2. GitHub repo → Settings → Pages → Source 選「Deploy from a branch」→ Branch `main`、資料夾 `/docs` → Save。
3. 1–2 分鐘後網址為 `https://<帳號>.github.io/forecast-teller/`。之後每次 push `docs/` 的變動都會自動重新部署。
