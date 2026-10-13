#!/usr/bin/env bash
# 新冠每週更新：下載開放資料 → 抓 NIDSS 門急診與重症 → 面板 → 即時預測 → 記錄預測 → 評估歷次預測 → 重建 docs/covid/。
# 用法：covid_forecast/scripts/weekly_update.sh [--push]（--push 只 add 明確路徑）
# 前提：主專案面板 data_processed/national_weekly.parquet 已更新（提供 rods_total 與 RESP_LAB），先跑流感的 weekly_update.sh。
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
export PYTHONWARNINGS=ignore
scripts/refresh_data.sh
$PY scripts/fetch_nidss_visits.py | tail -2
$PY scripts/fetch_severe.py | tail -1 || echo "重症抓取失敗，沿用現有 data/severe_weekly.csv"
$PY scripts/build_panel.py | head -2
$PY scripts/forecast_now.py --targets covid_oe covid_rods_pct covid_inp covid_oe_alt --covariates cny holiday | grep -v HF_TOKEN
$PY scripts/forecast_now.py --joint covid_out covid_rods --covariates cny holiday | grep -v HF_TOKEN
$PY ../scripts/log_forecast.py --project covid_forecast
$PY ../scripts/evaluate_forecasts.py --project covid_forecast --quiet
$PY scripts/build_site.py
if [[ "${1:-}" == "--push" ]]; then
  (cd .. && git add docs/covid covid_forecast/data_processed/coverage.json covid_forecast/data_processed/national_weekly.csv \
                    covid_forecast/outputs/latest covid_forecast/outputs/forecast_log \
                    covid_forecast/data/nidss_visits_history.csv covid_forecast/data/severe_history.csv \
    && git commit -m "新冠每週更新 $(date +%F)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || echo "nothing to commit"; git push)
fi
echo "done. preview: python3 -m http.server 8765 --directory ../docs  →  http://localhost:8765/covid/"
