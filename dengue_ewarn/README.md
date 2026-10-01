# dengue_ewarn — 登革熱鄉鎮層級預測式預警（CHG 情境一驗證）

獨立於流感預測模組的子專案：以 Google TimesFM 3.0 對台南、高雄 74 個鄉鎮的登革熱每日確定病例做 7 日累計的分布預測，換算「未來 14 天內突破 EWARN 閾值的機率」，並以前置時間、假警報率與 EWMA、CUSUM 等偵測器比較。計畫與結果見 [PLAN.md](PLAN.md)，視覺化報告發布在主站 `docs/dengue/`。

## 結構

| 路徑 | 內容 |
|---|---|
| `data/Dengue_Daily.csv` | 疾管署登革熱每日確定病例（Internet Archive 2025-07-29 快照的官方檔案；不入版控） |
| `data_processed/` | 鄉鎮 × 日面板、通報延遲直方圖（`build_panel.py` 產生） |
| `src/dengue_ewarn/` | `data.py` 面板與 as-of 重建、`alerts.py` 機率/偵測器/前置時間、`baselines.py`、`model.py`（TimesFM 3.0 MLX 包裝，與流感模組相同）、`calibration.py` 機率校準（等張回歸、Platt、留一年交叉驗證）與兩級門檻工作量、`site.py` 報告頁 |
| `scripts/` | `build_panel.py` → `run_backtest.py` → `make_report.py` → `calibrate.py` → `make_practical.py`（白話解讀用：每季警示數、簡單規則對照、真實案例）→ `build_site.py` |
| `outputs/backtest/` | 逐起點預測（parquet）、`dengue_REPORT.md`、p* 掃描、前置時間、配對比較 CSV；`dengue_CALIBRATION_REPORT.md` 與校準指標、細掃描、兩級門檻（`dengue_two_tier.csv`）、成本損失表 |
| `outputs/figures/` | 案例圖與示意圖 |

## 執行

```bash
source ../.venv/bin/activate          # 與主專案共用虛擬環境（timesfm[mlx]）
python scripts/build_panel.py        # 面板自 2008-01-01 起（2026-10-01 由 2012 拉回）
python scripts/run_backtest.py --years 2010 2011 2012 2014 2015 2016 2019 2023 2024 --step 2 --modes final asof_adj asof --batch 32 --chunk 1024 --context 730
python scripts/make_report.py --tag dengue
python scripts/calibrate.py          # 機率校準（留一年交叉驗證）＋兩級門檻工作量表
python scripts/build_site.py         # → ../docs/dengue/index.html
```

回測約 30–40 分鐘（M1、16 GB）；每個模式 × 年份會存 checkpoint（`outputs/backtest/dengue_checkpoints/`），可中斷續跑。
**改了面板起點、閾值或事件定義後要先刪掉 checkpoints 再跑**，否則會載入舊結果。

事件定義（2026-10-01 起）：警示機率用的閾值是各 context 模式在起點當日可得資料算出的 EWARN 閾值（`thr`）；
事件（真值）一律是最終資料的 7 日累計 ≥ 以最終資料計算的閾值（`thr_final`），三種模式共用同一組事件，AUC 與 p* 掃描才可比。
流行年 / 平靜年由資料判定：該年台南 + 高雄本土病例 ≥ 1,000（`data.epidemic_years`）。

第三輪（全台 274 個鄉鎮，tag `dengue_all`；2026-10-01 使用者決定）：

```bash
python scripts/build_panel.py --scope all --tag dengue_all
python scripts/run_backtest.py --tag dengue_all --panel dengue_all --years 2010 2011 2012 2013 2014 2015 2016 2018 2019 2020 2023 2024 --step 2 --modes final asof_adj --batch 32 --chunk 1024 --context 730
python scripts/make_report.py --tag dengue_all --panel dengue_all
python scripts/calibrate.py --tag dengue_all --panel dengue_all
python scripts/make_practical.py --tag dengue_all --panel dengue_all   # {tag}_practical.csv、{tag}_rule_vs_model.csv、{tag}_examples.json
python scripts/build_site.py --tag dengue_all --panel dengue_all     # 站台改以全台結果為主；asof 消融引用 tag dengue 的數字
```

報告與站台的分層：區域（台南高雄 / 其他縣市）× 事件型態（下限群聚 = 閾值在 3 例下限，即非流行區的第一個群聚；流行中加速 = 閾值高於下限），
輸出 `{tag}_strata_leads.csv`（依事件的前置時間）與 `{tag}_strata_sweep.csv`（依鄉鎮週的 AUC / 敏感度 / 假警報）。
加入 2013（屏東）、2018（台中、新北）、2020（新北三峽）三個其他縣市有局部爆發的年份；asof 模式在第二輪已證明不校正不可用，第三輪不跑。
