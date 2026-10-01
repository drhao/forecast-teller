# forecast-teller 交接文件（HANDOFF）

更新日期：2026-10-01（HEAD 見 §12，工作樹乾淨，分支 `main`）。
目的：換 Claude 帳號 / Team 之後，新 session 讀完本文件就能無縫接續。所有要點都以 repo 內的檔案為準；本文件只做索引、狀態快照與決定事項的整理。

---

## 0. 新 session 開場（先讀這一節）

**建議的第一句話**（貼到新 session）：

> 請先讀 HANDOFF.md 與 CLAUDE.md，跑一次 §0 的接手檢查，回報目前狀態後等我指示。

**接手檢查**（約 1 分鐘）：

```bash
cd /Users/drhao/Documents/GitHub/forecast-teller
git status && git log --oneline | head -5          # 應為 clean、HEAD ddf9cda（或之後的 commit）
source .venv/bin/activate && python tests/test_basics.py   # 無輸出即通過
ls data/*.csv | wc -l                               # 11 個原始檔（不在 git 內，必須在本機）
python3 -c "import json;print(json.load(open('data_processed/coverage.json'))['nhi']['last_complete'])"  # 目前 202638
python scripts/evaluate_forecasts.py --quiet && head -3 outputs/forecast_log/evaluation_summary.csv  # 預測紀錄與評估可跑
```

**閱讀順序**：本文件 → `CLAUDE.md`（工作規則）→ `README.md` → `PLAN_timesfm3_flu.md` §11–§12 → `ev_forecast/PLAN.md` §10–§12 → `dengue_ewarn/PLAN.md` §7–§11。

**目前沒有進行中的半成品。** 三條工作線都已實作、回測、上線；2026-10-01 已完成流感與腸病毒的每週更新，並加入 RESP_LAB 網站抓取、預測存檔與評估。最可能的下一件事是下一次每週更新（§9）或 §10 的待辦。

---

## 1. 專案全貌

同一個 repo、同一個虛擬環境、共用核心模組 `src/forecast_teller/`，三條工作線：

| 工作線 | 目錄 | 內容 | 狀態 | 線上頁面 |
|---|---|---|---|---|
| 流感主線 | 根目錄（`src/`、`scripts/`、`docs/`） | 以 TimesFM 3.0 零樣本預測全國類流感門診、急診、RODS 急診類流感%、重症；2016–2025 回測；每週預測 dashboard、回測 dashboard、一頁報告、投影片 | 完成並上線（2026-09-17 起） | https://drhao.github.io/forecast-teller/ 、`/backtest.html`、`/report.html` |
| 登革熱預警 | `dengue_ewarn/` | CHG 情境一驗證：台南、高雄 74 鄉鎮每日病例的 7 日累計分布預測 → 14 天內突破 EWARN 閾值的機率、前置時間、假警報、機率校準、兩級門檻工作量 | 第一版完成並上線（2026-09-17）；一次性驗證，無每週流程。**第二版在 `dengue` 分支進行（2026-10-01 起）**：事件定義統一、面板 2008 起、回測 9 年 | https://drhao.github.io/forecast-teller/dengue/ |
| 腸病毒預測 | `ev_forecast/` | 開放資料（健保門診 + RODS 急診）的腸病毒每週預測與回測，流行閾值依年設定 | 完成並上線（2026-09-19） | https://drhao.github.io/forecast-teller/ev/ 、`/ev/backtest.html`、`/ev/report.html` |

- GitHub：公開 repo `drhao/forecast-teller`，GitHub Pages 來源 `main` 分支的 `/docs`，push 後約 1–2 分鐘自動部署。
- 模型：`timesfm` 3.0.2，checkpoint `google/timesfm-3.0-pytorch`（330M 參數，9 個分位數 0.1–0.9），MLX 後端。**權重授權為非商業、非生產用途**；若要正式部署需改 TimesFM 2.5（Apache-2.0）或另談授權。
- 使用者是公衛 / 流病背景，用繁體中文溝通，熟悉監測資料的操作定義；回覆請用繁體中文。

---

## 2. 環境

| 項目 | 內容 |
|---|---|
| 機器 | Apple M1、16 GB、macOS；Python 3.11.6（Homebrew）；無 uv / conda |
| 虛擬環境 | `.venv/`（不進 git）。重建：`python3 -m venv .venv && source .venv/bin/activate && pip install "timesfm[mlx]" pandas pyarrow matplotlib statsforecast tabulate python-pptx pymupdf` |
| 已裝版本 | timesfm 3.0.2、mlx 0.32.2、numpy 2.4.6、pandas 2.3.3、pyarrow 25.0.1、matplotlib 3.11.2、statsforecast 2.1.1、python-pptx 1.0.2、pymupdf 1.28.2、scipy 1.17.1 |
| 模型快取 | `~/.cache/huggingface/hub/models--google--timesfm-3.0-pytorch`（1.2 GB）。換機器時第一次載入會自動下載 |
| PyTorch 後端 | 選配：`pip install "timesfm[torch]"`，`TimesFM3Model(backend="torch")` |
| 本機預覽 | `.claude/launch.json` 有 `docs-site`（`python3 -m http.server 8765 --directory docs`）；頁面用 fetch 讀 JSON，必須走 HTTP |
| 疾管署開放資料 TLS | `od.cdc.gov.tw` 憑證鏈缺 TWCA 中繼憑證，`curl` 會報 `unable to get local issuer certificate`。解法在 `ev_forecast/scripts/refresh_data.sh`：`ev_forecast/data/twca_intermediate.pem` + certifi 合成 bundle 後 `--cacert`。流感版 `scripts/refresh_data.sh` 尚未套用同一修法 |
| 投影片 QA | 用 PowerPoint AppleScript 匯出 PDF：`tell application id "com.microsoft.Powerpoint"`，HFS 路徑，先 `close every presentation saving no`；再用 PyMuPDF 轉圖檢查 |

---

## 3. Repo 結構與版控範圍

```
forecast-teller/
├── CLAUDE.md                 新 session 自動載入的工作規則（本次交接新增）
├── HANDOFF.md                本文件
├── README.md                 使用方式、流程、Pages 連結
├── PLAN_timesfm3_flu.md      流感主線規劃 + §11 實作狀態 + §12 回測結果 + §13 dashboard
├── pyproject.toml            套件定義（src layout）
├── .claude/launch.json       docs 預覽伺服器
├── data/                     流感原始 CSV（不進 git，只有 tw_holiday.csv、date_week_mapping.csv 入版控）
├── data_processed/           面板（parquet 不進 git；national_weekly.csv、coverage.json、qa_report.md 入版控）
├── src/forecast_teller/      核心模組（三條工作線共用）
├── scripts/                  流感主線 CLI
├── tests/test_basics.py      基本檢查（無 pytest 依賴）
├── outputs/                  回測 summary / REPORT.md / 圖 / 即時預測 / 投影片（parquet 與 pdf 不進 git）
├── docs/                     GitHub Pages 靜態站（index、backtest、report、assets、data/*.json、dengue/、ev/）
├── dengue_ewarn/             登革熱子專案（自有 src/scripts/data/data_processed/outputs、PLAN.md、README.md）
└── ev_forecast/              腸病毒子專案（自有 src/scripts/data/data_processed/outputs、PLAN.md、README.md）
```

**不在 git 內、但本機必須存在的檔案**（換機器時要另外搬）：

| 檔案 | 大小 | 來源 |
|---|---|---|
| `data/NHI_PROCESSED_48X.csv`、`_48X_ER.csv`（類流感門診/住院、急診） | 35 MB、9 MB | 內部健保申報檔（Big5），使用者手動更新 |
| `data/NHI_PROCESSED_487.csv`、`_487_ER.csv`（流感診斷） | 35 MB、9 MB | 同上 |
| `data/RODS_RS.csv`（RODS 急診，醫院層級） | 193 MB（超過 GitHub 單檔上限，這是 data/ 不進版控的原因） | 內部 |
| `data/NIDDS_487A.csv`（流感併發重症，發病週） | 1.4 MB | 內部 |
| `data/INFLUENZA_TYPE_YW.csv`、`INFLUENZA_MON_TYPE_YW.csv`（LARS 型別陽性數、檢驗件數） | 12 MB、5 MB | 內部（實驗室自動通報系統） |
| `data/RESP_LAB.csv`（社區合約實驗室呼吸道病原體 PCR，UTF-8，2025 起） | 小 | 公開：NIDSS 首頁 `Home/Index?op=6`「全國每週呼吸道病原體分子生物學檢出情形」，`scripts/fetch_resp_lab.py` 抓取（累積檔，網站只顯示近兩年）；每次抓取的值另存 `data/resp_lab_history.csv`（進 git） |
| `ev_forecast/data/NHI_EnteroviralInfection.csv`、`RODS_EnteroviralInfection.csv` | 8 MB、2 MB | 開放資料，`ev_forecast/scripts/refresh_data.sh` 可重新下載 |
| `dengue_ewarn/data/Dengue_Daily.csv` | 22 MB | 疾管署官方檔已下架；用 Internet Archive 的 2025-07-29 快照 |
| `.venv/`、`~/.cache/huggingface/` | — | 可重建 |

可重建的中間檔：`data_processed/*.parquet`（`scripts/build_panel.py`，11 秒）、`outputs/backtest/*_forecasts.parquet`（各 suite 重跑，分鐘級）、`dengue_ewarn/outputs/backtest/*.parquet` 與 checkpoints（約 20 分鐘）、`ev_forecast/outputs/backtest/*.parquet`。回測的 summary CSV 與 REPORT.md 都在 git 內，不重跑也能看結果。

---

## 4. 資料與陷阱（寫程式前必讀）

### 4.1 流感主線的原始資料

| 來源鍵 | 檔案 | 編碼 | 原始範圍 → 判定完整（2026-10-01 更新） | 注意 |
|---|---|---|---|---|
| `nhi` | NHI_PROCESSED_48X.csv（類流感）、NHI_PROCESSED_487.csv（流感） | cp950 | 200501–202639 → 200502–202638 | 門診 + 住院；`nhi_out_ili`、`nhi_out_total` 等 |
| `nhi_er` | NHI_PROCESSED_48X_ER.csv、NHI_PROCESSED_487_ER.csv | cp950 | 201552–202639 → 201601–202638 | 急診 |
| `rods` | RODS_RS.csv | cp950 | 200901–202639 → 200901–202638 | 醫院層級，`rods_ili`、`rods_total`；`rods_ili_pct` = ili / total × 100 |
| `nidds` | NIDDS_487A.csv | cp950 | 200302–202639 → 200302–202635 | 個案清單、發病週；沒有列的週是真實 0（zero-fill）；回補延遲 → `DEFAULT_LAG["nidds"] = 3`，尾端多刪 3 週 |
| `lab` | INFLUENZA_TYPE_YW.csv（A/B/未分型）、INFLUENZA_MON_TYPE_YW.csv（檢驗件數） | cp950 | 201453–202639 → 201502–202638 | **來源名稱是「實驗室自動通報系統（LARS）」，不是合約實驗室**；合約實驗室資料日後再加 |
| `resp` | RESP_LAB.csv | UTF-8-sig | 202501–202637 → 202502–202636 | 社區合約實驗室多重 PCR，每週約 250 件檢體；**由 `scripts/fetch_resp_lab.py` 從 NIDSS 網站抓**（`hcJson.push` 內嵌資料，未來週是 `undefined` 要換成 `null`）；網站會回補修正過去週，所以整份重抓並以長格式記錄在 `data/resp_lab_history.csv`（`fetched_on` × `yw`）；檔內有預填的空白未來列要丟掉；`resp_*` 欄；`DEFAULT_LAG["resp"] = 1` |
| — | tw_holiday.csv | UTF-8 | 到 2026-12-27 | 2021 與 2023–2025 的節日名稱放在 `holidayCategory` 而非 `name`，春節判斷要同時比對兩欄；2027 以後只有春節日期外推（`covariates.LNY_DATES`，假日天數假設 5 天），**2027 官方行事曆公布後要延伸此檔並重跑** |
| — | date_week_mapping.csv | UTF-8 | — | 日期 ↔ 疫情週對應 |

- 疫情週：台灣疾管署定義，週日起算，部分年份有第 53 週；2009 以前以日曆錨定。工具在 `weeks.py`（`yw_range`、`next_weeks`、`week_start`、`yw_of_date`、`seasonal_phase`）。
- 頭尾不完整週自動偵測：`panel.detect_edges`，邊緣週總量低於鄰近 8 週中位數的 0.6 倍即視為不完整（NIDDS 用 0.5）。結果寫在 `data_processed/coverage.json`，每次更新後先看這個檔。
- COVID 期 2020–2022 流感幾乎消失，是結構斷點；2024–2025 為歷史高點季。
- RODS 急診類流感就診百分比的**流行閾值是 11%**（使用者指定；2026-09-17 由 10% 修正），設定在 `backtest.DEFAULT_THRESHOLDS`，網頁用紅色虛線標示。其他目標的閾值預設為回測窗內實際值第 75 百分位。
- 共變數：春節週旗標、每週平日假日天數（最有用）、季節 sin/cos（無幫助）；過去共變數依延遲平移後 bfill/ffill。
- 回測只用 2016–2025 十年資料（使用者 2026-09-17 決定）：2016w01–2025w53 共 522 週，暖機 104 週，起點 415 個，h = 1–4，分 COVID 前（2018–2019）、COVID 期（2020–2022）、COVID 後（2023–2025，決策依據）。

### 4.2 腸病毒（`ev_forecast/`）

- 健保開放資料檔只有門診與住院、沒有急診；RODS 檔只有人次、沒有分母（分母借用流感面板的 `rods_total`）。`週` 欄是整數，讀檔器會自動轉成 `YYYYWW` 並判斷 UTF-8-sig / cp950。
- 門急診合計 `ev_oe = ev_out + ev_rods`（急診暫以 RODS 代替，使用者 2026-09-19 決定），與新聞稿公布值差距在一成以內（早年偏低、2024 後偏高）。取得健保急診檔後要替換並重跑 `core` suite。
- 流行閾值每年由疾管署手動設定，記錄在 `ev_forecast/data/thresholds.csv`（2016–2025 為 11,000，其中 2017–2022 為推定；2026 為 12,000），**每年要手動更新**。流行期規則暫定「單週達閾值即進入、連續 2 週低於即脫離」（`thresholds.RULE_TEXT`）。
- 腸病毒一年兩波、波形逐年不同，季節性 naive 幾乎無用（WIS 約 naive 的 4 倍）；學校行事曆（暑假 7/1–8/29、寒假、開學前兩週）是腸病毒專屬的已知未來共變數。

### 4.3 登革熱（`dengue_ewarn/`）

- 疾管署「登革熱 1998 年起每日確定病例統計」已於 2026-03 下架，使用 Internet Archive 保存的官方檔（2025-07-29 快照，發病日至 2025-07-23，107,387 筆）。
- 本土病例、台南 + 高雄 74 鄉鎮 × 日；面板自 **2008-01-01** 起（`data.PANEL_START`，2026-10-01 由 2012 拉回，讓 2010–2012 成為測試年；原始檔對合併前年份已用現行「區」名，不需對照）；以通報日重建「起點當日可得」快照，通報延遲中位數 2 天、第 95 百分位 7 天。
- 閾值 = max(2 × 前 3 週週均值, 3 例)（EWARN 2 倍規則加最小 3 例下限，使用者決定）。
- **事件定義（2026-10-01 統一）**：警示機率用各 context 模式在起點當日可得資料算的閾值（`thr`）；事件（真值）一律用最終資料算的閾值（`thr_final`），三種模式共用同一組事件。第一版各模式各用自己的閾值定義事件，AUC 與 p* 掃描不可比。
- 流行年 / 平靜年由資料判定：該年面板內本土病例 ≥ 1,000（`data.epidemic_years`），目前為 2010、2011、2012、2014、2015、2023。
- 驗證用固定的歷史快照即可（使用者 2026-10-01 說明），資料來源斷線（開放平臺 2026-03 下架、NIDSS 只有全國週計數）暫不處理。

### 4.4 新聞稿數字一律「待確認」

使用者 2026-09-19 提醒：新聞稿上看似專業的數字或解讀不一定正確。凡是從新聞取得的閾值、流行期規則、個案定義，都要參數化、標示「待確認」並列在「待你確認」清單，由使用者確認後才當作既定事實。

---

## 5. 流感主線：方法、結果、產出

### 5.1 模組與指令

| 模組 / 腳本 | 用途 |
|---|---|
| `io.py` | cp950 讀檔器、`read_resp_lab`、`read_holidays`、`load_all` |
| `weeks.py`、`panel.py` | 疫情週；面板 `national_weekly.*`、`county_weekly.parquet`、`age_weekly.parquet`、`coverage.json`、`qa_report.md` |
| `covariates.py` | `holiday_weekly`、`seasonal_features`、`future_covariates(kinds=...)` |
| `model_timesfm3.py` | `TimesFM3Model(backend, batch_size).forecast(contexts, horizon, past_future, past_only, log1p, symmetric, positive)` → (median, quantiles) |
| `baselines.py` | snaive、naive、MA3（遞迴）、AutoETS、Theta；分位數 = 點 + 中位數置中的經驗誤差 |
| `metrics.py` | WIS、涵蓋率、MAE、MASE、方向命中、三分類命中、±10% 容忍、閾值命中/誤報 |
| `backtest.py` | `Config` dataclass、`run_config(..., panel=, covariate_fn=)`（子專案掛勾）、`summarize`、`SEGMENTS` |
| `scripts/run_backtest.py` | suites：baselines、ma3、stats、layer1、layer2、layer2b、layer3、layer4、core、resp、resp_smooth |
| `scripts/run_group_backtest.py` | `--by county|age`，逐序列單變量 vs 聯合多變量 |
| `scripts/make_report.py`、`add_hit_rates.py` | `*_REPORT.md`；對既有結果補算命中率 |
| `scripts/fetch_resp_lab.py` | 從 NIDSS 網站抓 RESP_LAB（累積合併 + `data/resp_lab_history.csv`）；`--dry-run` 只比對、`--import 舊檔 日期` 匯入舊匯出檔到 history |
| `nidss_cdcwnh.py`（根目錄） | 使用者提供的 NIDSS 健保門急診查詢（CDCWNH01/02/03/09 表單 POST）參考程式，尚未接進流程 |
| `scripts/forecast_now.py` | 即時預測：`--joint nhi_out_ili nhi_er_ili rods_ili --covariates cny holiday`；單變量 `--targets ...` |
| `scripts/log_forecast.py` | 把 `outputs/latest/latest_forecast*.csv` 累積到 `outputs/forecast_log/forecast_log.csv`（含 `run_at`，同 mode/target/origin/target_yw 以最新取代；另加門急診合計 `nhi_oe_ili`、mode `joint_sum`）；`--from-git` 回填 git 歷史的快照 |
| `scripts/evaluate_forecasts.py` | 把 forecast_log 與 `national_weekly.csv` 實際值對上 → `outputs/forecast_log/evaluation.csv`（逐列 abs_err / ape / in60 / in80 / wis）與 `evaluation_summary.csv`（mode × target × h）；`--since 起點週` 篩選 |
| `scripts/build_site.py` | `outputs/` → `docs/data/{latest,backtest,meta}.json` + `docs/report.html`；`build_narrative`（規則生成的趨勢判讀）、`combine_out_er`（門急診合計圖） |
| `scripts/weekly_update.sh [--push]` | 抓 RESP_LAB → 面板 → 聯合與單變量即時預測 → `log_forecast.py` → `evaluate_forecasts.py` → 站台；`--push` 只 add `docs`、`data_processed/{coverage.json,qa_report.md,national_weekly.csv}`、`outputs/latest`、`outputs/forecast_log`、`data/resp_lab_history.csv` |
| `scripts/make_slide_charts.py`、`build_deck.py` | 投影片（`outputs/slides/forecast-teller_成果報告_2026-09-17.pptx`，6 頁，PingFang TC） |

### 5.2 回測結果（COVID 後 2023–2025，h = 1–4 平均）

| 目標 | 最佳設定 | WIS | 相對 naive | 其他 |
|---|---|---:|---:|---|
| 全國類流感門診人次 `nhi_out_ili` | `tfm_best_mv4_cny_hol`：門診 + 急診 + RODS + 重症四變量聯合 + 春節旗標 + 假日天數 | 6,857 | 0.74（h1–4：−32% / −29% / −23% / −22%） | 80% 涵蓋 0.82、MAPE 10.5%、方向命中 0.68、閾值命中 0.85 / 誤報 0.17 |
| 全國類流感急診人次 `nhi_er_ili` | 同上 | 656 | 0.70 | MAPE 10.4%、涵蓋 0.81、方向命中 0.72 |
| RODS 急診類流感% `rods_ili_pct` | `tfm_expanding_sym`（單變量、對稱平均） | 0.63 pp | 0.88 | 11% 閾值命中 0.82 / 誤報 0.19（naive 0.78 / 0.24） |
| 22 縣市 | 聯合多變量 + 季節/春節 | — | 0.86 | 每縣市皆優於 naive（0.79–0.90） |
| 18 年齡層 | 聯合 + 共變數 | — | 0.83 | COVID 前 0.72 |

層結論：零樣本單變量已比 naive 低 16%、比 AutoETS 低 12%；假日天數是最有用的共變數；季節相位、log1p、對稱平均、滑動視窗無幫助；MA3 比 naive 差 6–13%；季節性 naive 失效（MASE > 1）；RESP_LAB 當共變數反而 +6–9% WIS，所以只用在敘事；LARS 當共變數中性。

### 5.3 最新預測（起點 202638，資料 2026-10-01 更新）

| 指標 | 202639 | 202640 | 202641 | 202642 |
|---|---:|---:|---:|---:|
| 門診人次（聯合） | 121,209 | 114,586 | 113,944 | 105,090 |
| 急診人次（聯合） | 13,051 | 12,522 | 11,412 | 10,675 |
| 門急診合計（兩者相加） | 134,260 | 127,108 | 125,356 | 115,765 |
| RODS 急診類流感%（單變量） | 12.2 | 12.2 | 11.8 | 11.6 |
| 重症（起點 202635，目標週 202636–202639） | 103 | 99 | 93 | 87 |

202638 實際：門診 122,060（就診率 2.37%）、急診 12,772、RODS 12.2%（連續 4 週高於 11% 閾值）、重症 202635 為 106 例。模型判讀為高原期、未來 4 週緩降。**202638 的健保門診總就診人次比前週少 12%（5.15M 對 5.84M），疑似申報未齊，就診率偏高，下次更新可能上修。** 這些數字會被下次更新覆蓋；歷次預測都在 `outputs/forecast_log/forecast_log.csv`，與實際值的比對在 `evaluation.csv`。

前一次（起點 202636）的即時預測對 202637 的 1 週前誤差：門診 1.7%、急診 1.5%、RODS% 4.4%，皆在 80% 區間內；2 週前對 202638 的門診高估 15%（受上述分母問題影響）。

### 5.4 網站

- `docs/index.html`：趨勢判讀（headline / 目前趨勢 / 模型判讀 / 注意事項，`build_narrative` 自動生成）、KPI、門急診合計總覽圖（門診 + 急診兩個聯合預測相加；分位數同向相加，區間偏保守）、四張扇形圖（歷史範圍 40 週 / 1.5 年 / 2.5 年）、分位數表。
- `docs/backtest.html`：目標切換、分段、可排序排行榜（含 MAPE、閾值註腳）、依 horizon 的 WIS / 方向命中 / 涵蓋 / 閾值命中、1–4 週前預測 vs 實際四張圖、各年、縣市與年齡層。
- `docs/report.html`：一頁式解讀與評估報告，含「使用的資料」一節與 RESP 實驗說明。
- 風格：`docs/assets/charts.js`（`window.EPI`：fanChart、linesChart、backtestTsChart、hbarChart、refLines）與 `epi.css`，依 https://github.com/drhao/epi-dataviz-styleguide （主色 #739A6D、線 #5D7F58、區間 #B4C9B1、警示紅 #BE373C 僅用於閾值線、Noto Sans/Serif TC、tabular-nums、僅水平格線）。Chart.js 4.4.1 由 cdnjs 載入；資產連結帶 `?v=` 版本號避免快取。
- 瀏覽器檢查時的已知假訊息：內建瀏覽器對任何 Chart.js 頁面都會報 `Cannot convert object to primitive value`，是工具儀器化造成，不是頁面錯誤。

---

## 6. 登革熱子專案 `dengue_ewarn/`

- 流程：`build_panel.py` → `run_backtest.py --years 2010 2011 2012 2014 2015 2016 2019 2023 2024 --step 2 --modes final asof_adj asof --batch 32 --chunk 1024 --context 730` → `make_report.py --tag dengue` → `calibrate.py` → `build_site.py`（→ `docs/dengue/index.html`）。回測約 50 分鐘，有 checkpoint 可續跑（改面板或事件定義要先刪 `outputs/backtest/*_checkpoints/`）。第三輪（全台 274 鄉鎮）的指令見 `dengue_ewarn/README.md`，所有腳本都吃 `--tag` / `--panel`。
- 機器會在長跑時睡眠：第二輪最後一個模式年花了 2.3 小時（其他 2 分鐘）。長回測前先用 keep-awake（Claude app 的 `request_keep_awake`），或在系統設定關閉睡眠。
- 使用者決定（2026-09-17）：用確定病例、鄉鎮層級、EWARN 閾值加 3 例下限、先不比 TimesFM 2.5、假警報成本以工作量 / 取捨曲線 / 配對比較呈現。
- 主要結果（第二版，2026-10-01，74 區 × 9 年，事件定義統一）：asof_adj 模式 AUC 0.89（naive 0.77）；p* = 0.5 時敏感度 0.35、每 100 鄉鎮週假警報 1.6、前置中位數 4 天（CUSUM 0 天、EWMA 2 天）；完整度校正是關鍵（不校正 AUC 0.86、前置幾乎消失）。共同年份的 WIS 與第一版完全相同。分層：下限群聚（非流行區的第一個群聚，170 個）模型 p* = 0.3 偵測 0.37、前置 3 天，CUSUM 0.28、0 天，EWMA 0；流行中加速（143 個）模型 0.64、9 天。詳見 PLAN §12.1。
- 校準結論：原始機率已校準（ECE 0.013、Brier 0.049），留一年交叉驗證的等張 / Platt 重校準反而變差（各季事件率 0–19% 差太多），**部署用原始機率，只做校準監測**。
- 兩級門檻建議：注意 0.3 / 警示 0.4–0.5；20 站轄區每週約 1.6 個注意標示、0.85–1.14 張查證單（0.30–0.46 張假警報），前置 4 天。站台規則「假警報單 ≤ 0.5 張中敏感度最高」在第二版選到 0.4（0.46 張，剛好在界內）。
- 第三輪（進行中，`dengue` 分支）：全台 274 鄉鎮、12 年（加 2013、2018、2020）、模式 final + asof_adj，tag `dengue_all`；報告與站台分層「台南高雄 / 其他縣市 × 下限群聚 / 流行中加速」。見 PLAN §13。
- 未做（PLAN §10 末尾）：多變量聯合 + 鄰區共變數、Farrington flexible 基準、每日起點、週層級腹瀉 / 腸病毒 / 類流感症候群套用同流程、「群聚起始」事件定義。只在使用者要求時再做。

---

## 7. 腸病毒子專案 `ev_forecast/`

- 流程：`scripts/refresh_data.sh` → `build_panel.py`（需要主專案 `data_processed/national_weekly.parquet` 提供 `rods_total`）→ `run_backtest.py --suite ...` → `forecast_now.py --joint ev_oe ev_out ev_rods --covariates school holiday` 與 `--targets ev_rods_pct` → `build_site.py`；一鍵 `scripts/weekly_update.sh [--push]`（會自己下載開放資料）。
- 回測（COVID 後，門急診合計）：最佳為三變量聯合（合計 + 門診 + RODS）+ 學校行事曆 + 假日，WIS 1,105、相對 naive 0.81（h1–4：0.79 / 0.80 / 0.82 / 0.83）、MAPE 15.3%、方向命中 0.71、11,000 閾值命中 0.84 / 誤報 0.08；80% 涵蓋 0.75（偏窄）。零樣本無共變數 0.85；AutoETS 0.95；MA3 1.18；季節性 naive 3.96。
- 最新預測（起點 202638，2026-10-01 更新）：門急診合計 9,761 → 9,699 → 10,087 → 10,647，達 2026 閾值 12,000 的機率約 0.05 → 0.05 → 0.2 → 0.3。前一次（起點 202636）預測上升到 12,866，實際 202637–202638 為 10,502、9,945，2 週前預測高估 17%（見 `ev_forecast/outputs/forecast_log/evaluation.csv`）。
- 預測紀錄與評估同主線：`scripts/log_forecast.py --project ev_forecast`、`scripts/evaluate_forecasts.py --project ev_forecast`，檔案在 `ev_forecast/outputs/forecast_log/`（已回填 2026-09-19 的起點 202636 快照）。
- 未做（PLAN §12.4）：取得健保急診檔後替換 RODS；加入重症、型別、停課共變數；區間校正（涵蓋率拉回 0.80）；年齡層與縣市聯合回測；每週保存開放資料快照估回補係數。

---

## 8. 使用者決定事項與工作規則（必讀）

### 8.1 決定事項（依日期）

| 日期 | 決定 |
|---|---|
| 2026-09-17 | 回測只用 2016–2025 十年資料 |
| 2026-09-17 | 評估加入命中率（方向、三分類、±10%、閾值命中/誤報）與 MAPE；基準加 MA3（前 3 週平均，h ≥ 2 遞迴） |
| 2026-09-17 | 回測頁放 1–4 週前預測 vs 實際四張圖；加全國類流感急診人次（健保）為預測目標 |
| 2026-09-17 | 圖表風格依 epi-dataviz-styleguide；兩個 dashboard + 一頁報告，GitHub Pages 發布 |
| 2026-09-17 | 每週預測頁開頭要有自動生成的趨勢判讀 |
| 2026-09-17 | 實驗室資料來源名稱為「實驗室自動通報系統（LARS）」；合約實驗室資料日後再加 |
| 2026-09-17 | 報告加「使用的資料」；RESP_LAB 納入近兩年評估（結論：不當共變數） |
| 2026-09-17 | RODS 流行閾值 11%（非 10%），網頁標注 |
| 2026-09-17 | 登革熱：確定病例、鄉鎮、EWARN + 3 例下限、不比 2.5 |
| 2026-09-17 | 誤收的登革熱檔案移出版控（已完成；歷史 commit 未清除） |
| 2026-09-18 | 每週頁加「全國門急診類流感人次」合計圖（門診 + 急診預測相加） |
| 2026-09-19 | 腸病毒：閾值依年記錄於 thresholds.csv；流行期規則暫定；急診以 RODS 代替；獨立子專案資料夾 |
| 2026-09-19 | 新聞稿數字需向使用者確認 |
| 2026-10-01 | RESP_LAB 改由 `fetch_resp_lab.py` 從 NIDSS 網站抓；每次抓到的值留存於 `data/resp_lab_history.csv`（網站只顯示近兩年，舊時點的值日後找不到） |
| 2026-10-01 | 每週預測值累積到 `outputs/forecast_log/forecast_log.csv`，供日後評估實際預測表現（`evaluate_forecasts.py`） |
| 2026-10-01 | 登革熱第二版在 `dengue` 分支做（主線每週更新在 `main`），完成後再合併 |
| 2026-10-01 | 登革熱：事件定義統一為最終資料閾值；面板起點拉到 2008；驗證只用固定的歷史快照，不處理資料來源斷線 |

### 8.2 工作規則

1. **git 只暫存明確路徑，禁止 `git add -A` / `git add .`**。曾經把平行進行中的登革熱檔案掃進 commit（aa5d214、04a6a1b），事後才用 `git rm --cached` 移除。commit 前先 `git status` 確認沒有別的工作線的檔案。
2. 原始資料不進 git（`.gitignore` 已設）。新增資料檔先確認大小與授權；新子專案的 `data/` 也要加到 `.gitignore`。
3. 所有工作直接在 `main` 上進行並 push（使用者的習慣）；使用者曾被建議改用 feature branch + PR，尚未採用。
4. 回覆與文件用繁體中文；程式註解可中英混用。
5. 站台改版後必須在瀏覽器實際驗證（本機 `http.server` 或線上加 `?v=時間戳`），再回報。
6. 流感原始資料是使用者手動放進 `data/` 的內部檔，不要用 `scripts/refresh_data.sh` 的開放資料檔直接覆蓋（欄位格式不同）。
7. 投影片、報告、網頁的數字一律從 `outputs/` 與 `docs/data/*.json` 讀取，不手打。
8. 記憶檔（`~/.claude/projects/-Users-drhao-Documents-GitHub-forecast-teller/memory/`）與本文件若有衝突，以本文件與 repo 內 PLAN 為準。

---

## 9. 每週更新 SOP

**流感（內部資料）**

1. 使用者把更新後的內部 CSV 放進 `data/`（NHI、RODS、NIDDS、LARS，檔名不變），必要時延伸 `data/tw_holiday.csv`。RESP_LAB 不用手動放，下一步會從 NIDSS 網站抓。
2. `scripts/weekly_update.sh --push`：`fetch_resp_lab.py`（抓取失敗不中斷，沿用現有檔）→ 面板 → 聯合預測（門診 + 急診 + RODS，春節 + 假日）→ 單變量預測（門診、RODS%、急診、重症）→ `log_forecast.py`（預測存檔）→ `evaluate_forecasts.py`（與實際值比對）→ `build_site.py` → commit 明確路徑 → push。
3. 檢查：`data_processed/coverage.json` 的 `last_complete`（NIDDS 通常落後 3 週）、`qa_report.md`、`fetch_resp_lab.py` 印出的回補修正、每週頁的趨勢判讀文字是否合理、線上頁面（約 1–2 分鐘後）。健保門診總就診人次若比前週掉一成以上（如 202638：5.15M 對 5.84M），多半是申報未齊而非真實變化，就診率會偏高，下週會上修。
4. 預測評估：`python scripts/evaluate_forecasts.py`（或 `--since 202636`）看歷次即時預測的 MAPE、涵蓋率、WIS。注意 forecast_log 裡起點 202603–202607 的列來自 2026-09-17 初始 commit 用舊資料擷取跑的預測（不是當週即時做的），而且 202607 是春節週，解讀時要分開看；真正的即時預測從起點 202636 起。
5. 若要同步更新投影片：`python scripts/make_slide_charts.py && python scripts/build_deck.py`。

**腸病毒（開放資料）**：先跑完流感的更新（面板提供 `rods_total`），再 `ev_forecast/scripts/weekly_update.sh --push`（自行下載、面板、預測、`log_forecast.py --project ev_forecast`、`evaluate_forecasts.py --project ev_forecast`、站台、commit `docs/ev` 與 `ev_forecast/{data_processed/coverage.json, data_processed/national_weekly.csv, outputs/latest, outputs/forecast_log}`）。每年初確認 `thresholds.csv` 是否要加新年度的閾值。

**登革熱**：無每週流程。

---

## 10. 待辦與候選下一步（都尚未開始，依使用者指示）

- 流感：集成（TimesFM + ETS）、微調（3.0 無官方腳本且授權受限）、健保申報回補 nowcast、合約實驗室資料加入、把「門診 + 急診合計」直接當一個序列預測以取代分位數相加、2027 假日檔延伸、流感版 `refresh_data.sh` 套用 TWCA 憑證修法。
- 腸病毒：§7 的未做清單。
- 登革熱：§6 的未做清單。
- 版控：改用 feature branch + PR 流程；是否清除歷史 commit 中的登革熱暫存檔（需 rewrite history，未做）。
- 授權：正式部署前評估 TimesFM 2.5。

---

## 11. 踩過的坑（避免重複）

- `tw_holiday.csv` 的春節名稱在不同年份放在不同欄位 → 同時比對 `name` 與 `holidayCategory`（有測試）。
- `holiday_weekly` 在歷史週與預測週重疊時出現重複索引 → 先去重。
- 年齡層面板缺格造成 NaN → 所有聚合格子補 0。
- `read_resp_lab` 的 `pd.NA` 轉 float 失敗 → 用 `np.nan`，並丟掉預填的空白未來列。
- 多目標 summary 樞紐表重複索引 → 只取 `target == primary`。
- RESP suite 的多變量設定因 NIDDS 尾端缺 3 週失敗 → resp suite 的聯合設定不含重症；`_past_only_matrix` 加 `.ffill()`。
- 基準模型的中位數被經驗誤差分位數位移 → `_from_errors(center=True)`。
- zsh 把 `$F='grep ...'` 當指令 → 不要用變數包指令。
- 正規表示式批次改 HTML 時掉了 `historyWeeks: weeks` → 改完要在瀏覽器驗證按鈕。
- PowerPoint AppleScript：`tell application "Keynote"` 會打到別的 app；PowerPoint 需用 bundle id、HFS 路徑、先關閉所有簡報，否則匯出舊內容；表格框線在 `a:tcPr` 要先放 `a:lnL/R/T/B` 再放填色。
- 瀏覽器快取舊 CSS/HTML → 資產連結加 `?v=`。
- 一次 push 因 DNS 失敗 → 重試即可。

---

## 12. 交接時的狀態快照（2026-10-01）

- `git status` 乾淨；HEAD `ddf9cda`（2026-10-01）。當天的 commit 依序：`ac77d74` NIDSS 抓取腳本、`a8df283` 流感每週更新、`a7932df` 預測存檔與評估 + RESP_LAB 留存 + 文件、`ddf9cda` 腸病毒每週更新。remote `origin` = https://github.com/drhao/forecast-teller.git ；GitHub Pages 已部署（`/` 與 `/ev/` 的 latest.json 起點皆 202638）。
- 線上七個頁面：`/`、`/backtest.html`、`/report.html`、`/dengue/`、`/ev/`、`/ev/backtest.html`、`/ev/report.html`；`docs/data/latest.json` 起點 202638、含 `combined`；`docs/ev/data/latest.json` 起點 202638。
- 資料：流感各來源完整至 202638（NIDDS 202635、RESP 202636）；腸病毒至 202638（2026-10-01 更新）；登革熱至 2025-07-23。
- `dengue` 分支（2026-10-01 開）：登革熱第二版，見 `dengue_ewarn/PLAN.md` §12；第一版的 outputs 備份在該 session 的 scratchpad，不在 repo。
- `outputs/forecast_log/forecast_log.csv` 有 7 個起點（202603–202638）共 96 列，`ev_forecast/outputs/forecast_log/forecast_log.csv` 有 2 個起點（202636、202638）共 44 列；`data/resp_lab_history.csv` 有 2026-09-17 與 2026-10-01 兩次抓取。
- 未追蹤 / 未納入流程：根目錄 `nidss_cdcwnh.py`（已進 git，只是參考程式）；流感版 `scripts/refresh_data.sh` 仍未套 TWCA 憑證修法（內部檔由使用者手動更新，所以暫時不需要）。
- 虛擬環境可用，模型快取在本機。
- 本機還有 Claude session 紀錄（§附錄 B），換帳號後舊 session 看不到，但檔案仍在。

---

## 附錄 A：Claude 記憶檔（同機器會自動沿用；換機器或記憶遺失時依此重建）

位置：`~/.claude/projects/-Users-drhao-Documents-GitHub-forecast-teller/memory/`，目錄依專案路徑而定，與帳號無關。若新 session 的記憶目錄是空的，請依下面三個檔案的要點重建（`MEMORY.md` 為索引）。

- `project-timesfm3-flu-forecasting.md`（type: project）：本文件 §1–§7 的濃縮版，含三條工作線的狀態、最佳設定、資料事實、授權注意、發布資訊。
- `feedback-verify-news-claims.md`（type: feedback）：§4.4 的規則與緣由（2026-09-19 使用者的提醒）。
- `user-profile.md`（type: user，本次新增）：使用者為公衛 / 流病研究者，GitHub 帳號 drhao，用繁體中文，Apple M1 16 GB，熟悉監測資料定義，偏好直接在 main 上工作並即時上線。

## 附錄 B：session 紀錄位置（僅同一台機器）

`~/.claude/projects/-Users-drhao-Documents-GitHub-forecast-teller/*.jsonl`：
`72197ff3-…`（流感主線，本文件的來源 session）、`c61b7a15-…`（登革熱子專案）、`0ab1b5ee-…`（腸病毒子專案）、`520b13b1-…`（2026-09-16 的早期短 session）。新帳號不會載入這些檔案，只作查證用。
