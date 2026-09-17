# dengue_ewarn — 登革熱鄉鎮層級預測式預警（CHG 情境一驗證）

獨立於流感預測模組的子專案：以 Google TimesFM 3.0 對台南、高雄 74 個鄉鎮的登革熱每日確定病例做 7 日累計的分布預測，換算「未來 14 天內突破 EWARN 閾值的機率」，並以前置時間、假警報率與 EWMA、CUSUM 等偵測器比較。計畫與結果見 [PLAN.md](PLAN.md)，視覺化報告發布在主站 `docs/dengue/`。

## 結構

| 路徑 | 內容 |
|---|---|
| `data/Dengue_Daily.csv` | 疾管署登革熱每日確定病例（Internet Archive 2025-07-29 快照的官方檔案；不入版控） |
| `data_processed/` | 鄉鎮 × 日面板、通報延遲直方圖（`build_panel.py` 產生） |
| `src/dengue_ewarn/` | `data.py` 面板與 as-of 重建、`alerts.py` 機率/偵測器/前置時間、`baselines.py`、`model.py`（TimesFM 3.0 MLX 包裝，與流感模組相同）、`site.py` 報告頁 |
| `scripts/` | `build_panel.py` → `run_backtest.py` → `make_report.py` → `build_site.py` |
| `outputs/backtest/` | 逐起點預測（parquet）、`dengue_REPORT.md`、p* 掃描、前置時間、配對比較 CSV |
| `outputs/figures/` | 案例圖與示意圖 |

## 執行

```bash
source ../.venv/bin/activate          # 與主專案共用虛擬環境（timesfm[mlx]）
python scripts/build_panel.py
python scripts/run_backtest.py --years 2014 2015 2016 2019 2023 2024 --step 2 --modes final asof_adj asof --batch 32 --chunk 1024 --context 730
python scripts/make_report.py --tag dengue
python scripts/build_site.py         # → ../docs/dengue/index.html
```

回測約 20 分鐘（M1、16 GB）；每個模式 × 年份會存 checkpoint，可中斷續跑。
