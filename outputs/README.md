# outputs/

| 檔案 | 內容 |
|---|---|
| `backtest/nhi_out_ili_REPORT.md` | 全國類流感門診人次：Layer 1–3 回測總報告（排行榜、WIS by horizon、各年、診斷、圖） |
| `backtest/rods_ili_pct_REPORT.md` | RODS 急診類流感就診百分比：Layer 1 回測報告 |
| `backtest/county_nhi_out_ili_REPORT.md` | 22 縣市：逐縣市單變量 vs 22 縣市聯合多變量 |
| `backtest/age_nhi_out_ili_REPORT.md` | 18 年齡層：逐層單變量 vs 聯合多變量 |
| `backtest/*_forecasts.parquet` | 逐起點、逐 horizon 的中位數、q10–q90、實際值、WIS（可自行再分析） |
| `backtest/*_summary*.csv` | 各設定 × 分段 × horizon 的指標 |
| `figures/` | WIS by horizon、預測 vs 實際的時間序列圖 |
| `latest/latest_forecast.csv`、`latest/forecast_*.png` | 從最新完整週往後 4 週的即時預測（含分位數） |

重跑方式見專案根目錄 `README.md`。
