# forecast-teller

以 Google **TimesFM 3.0** 對台灣流感監測資料做零樣本（zero-shot）預測。規劃文件見 [PLAN_timesfm3_flu.md](PLAN_timesfm3_flu.md)；接手或交接請先讀 [HANDOFF.md](HANDOFF.md)（狀態快照、決定事項、每週 SOP）與 [CLAUDE.md](CLAUDE.md)（工作規則）。

**線上網站（GitHub Pages）**：[總覽](https://drhao.github.io/forecast-teller/) · [類流感](https://drhao.github.io/forecast-teller/flu/) · [新冠](https://drhao.github.io/forecast-teller/covid/) · [腸病毒](https://drhao.github.io/forecast-teller/ev/) · [登革熱](https://drhao.github.io/forecast-teller/dengue/)

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
# 登革熱鄉鎮預警（CHG 情境一）是獨立子專案：見 dengue_ewarn/README.md
# 腸病毒疫情預測是獨立子專案：見 ev_forecast/README.md（網站 docs/ev/）
# 新冠疫情預測是獨立子專案：見 covid_forecast/README.md（網站 docs/covid/）
```

## 目錄

| 路徑 | 內容 |
|---|---|
| `data/` | 原始 CSV（健保、RODS、NIDDS、實驗室自動通報系統（LARS）；Big5 編碼）與 `RESP_LAB.csv`（社區合約實驗室呼吸道病原體 PCR，UTF-8，2025 起；由 `scripts/fetch_resp_lab.py` 從 NIDSS 網站抓取，每次抓到的值留存於 `resp_lab_history.csv`；目前用於病原體組成監測，回測顯示作為共變數無增益） |
| `data_processed/` | `national_weekly.*`、`county_weekly.parquet`、`age_weekly.parquet`、`coverage.json`、`qa_report.md` |
| `src/forecast_teller/` | `io` 讀檔、`weeks` 疫情週、`panel` 面板、`covariates` 共變數、`model_timesfm3` 模型包裝、`baselines`、`metrics`（WIS）、`backtest`、`report` |
| `scripts/` | 可執行流程 |
| `outputs/backtest/` | 每個設定的逐起點預測（parquet）、summary、`*_REPORT.md` |
| `outputs/figures/`、`outputs/latest/` | 圖與即時預測 |

## 回測設計

- 資料窗 2016w01–2025w53（522 週），暖機 2016–2017，起點 2018w01 起每週一個，h = 1–4。
- 指標：WIS、MAE、MAPE、WAPE、MASE、80%/60% 區間涵蓋率，以及命中率（方向命中率、三分類命中率、±10% 容忍帶、閾值命中率/誤報率/準確率；`scripts/add_hit_rates.py` 可對既有結果補算，`make_report.py --threshold` 指定流行閾值；RODS 急診類流感%預設用流行閾值 11%）；基準：季節性 naive、last value、MA3（前 3 週移動平均，2 週以上遞迴代入）、AutoETS、Theta。
- 結果分 COVID 前（2018–2019）、COVID 期（2020–2022）、COVID 後（2023–2025）三段報告。

## 回測結論（2026-09-17）

最佳設定為四變量聯合（門診 + 急診 + RODS + 重症）加春節旗標與假日天數共變數：COVID 後 1–4 週的 WIS 比 last-value naive 低 32% / 29% / 23% / 22%，80% 區間涵蓋率 0.82。無共變數的 zero-shot TimesFM 已比 naive 低 16%、比 AutoETS 低 12%。詳見 `PLAN_timesfm3_flu.md` 第 12 節與 `outputs/backtest/*_REPORT.md`。

## Dashboard 與 GitHub Pages

`docs/` 是可直接發布到 GitHub Pages 的靜態站台（Chart.js，依《疫情資料視覺化指引》配色與規範），**以疾病分類**（2026-10-19 改版）：上層導覽選疾病，下層選該疾病的頁面。

| 路徑 | 內容 | 產生方式 |
|---|---|---|
| `docs/index.html` | 總覽：四種疾病各一張卡片（最新週、判讀、主指標近半年與 4 週預測、4 週內超過閾值機率、流行期狀態） | 靜態，讀各疾病的 `latest.json` |
| `docs/flu/` | 類流感：`index.html` 預測、`backtest.html` 回測、`report.html` 報告、`method.html` 方法、`data/*.json` | `scripts/build_site.py` |
| `docs/covid/` | 新冠：同上四頁（門急診 = NIDSS 健保門診 + 急診；流行閾值待確認，以暫行規則標示） | `covid_forecast/scripts/build_site.py` |
| `docs/ev/` | 腸病毒：同上四頁（門急診合計 = 健保門診 + RODS 急診；流行閾值依年設定） | `ev_forecast/scripts/build_site.py` |
| `docs/dengue/` | 登革熱：`index.html` 每日預警、`report.html` 驗證報告、`method.html` 方法、`latest.json`、`data.json` | `dengue_ewarn/scripts/build_site.py`、`daily_update.sh` |
| `docs/assets/nav.js` | **導覽列唯一來源**：網站地圖與兩層頁首；各頁只放 `<header class="masthead" data-disease=… data-page=…>` | 手改 |
| `docs/assets/epi.css`、`charts.js` | 共用樣式與圖表（`window.EPI`） | 手改 |

產生的頁面（報告、方法、登革熱三頁）用 `src/forecast_teller/site_common.py` 的 `page_head`、`method_page`；手寫的預測頁與回測頁直接放 header 與 `nav.js`。新增疾病或頁面時改 `nav.js` 的 `SITE` 與 `site_common.DISEASE_NAME`。

本機預覽（頁面用 fetch 讀 JSON，需走 HTTP）：

```bash
python3 -m http.server 8765 --directory docs
```

每週更新一鍵流程：`scripts/weekly_update.sh`（抓 RESP_LAB → 面板 → 預測 → 記錄預測到 `outputs/forecast_log/` → 評估歷次預測 → 站台；加 `--push` 會 commit 並 push）。歷次即時預測與實際值的比對：`python scripts/evaluate_forecasts.py`。

發布到 GitHub Pages（首次）：
1. `git init && git add -A && git commit -m "init"`，在 GitHub 建 repo 並 `git remote add origin ... && git push -u origin main`。`data/*.csv` 已在 `.gitignore`（RODS 檔 196 MB 超過 GitHub 單檔上限），只有假日與週對應表會進版控。
2. GitHub repo → Settings → Pages → Source 選「Deploy from a branch」→ Branch `main`、資料夾 `/docs` → Save。
3. 1–2 分鐘後網址為 https://drhao.github.io/forecast-teller/ （總覽），類流感在 `/flu/`。之後每次 push `docs/` 的變動都會自動重新部署（已於 2026-09-17 設定完成）。
