#!/usr/bin/env bash
# 腸病毒每週更新：下載開放資料 → 面板 → 即時預測 → 記錄預測 → 評估歷次預測 → 重建 docs/ev/。
# 用法：ev_forecast/scripts/weekly_update.sh [--push]（--push 只 add 明確路徑）
# 前提：主專案面板 data_processed/national_weekly.parquet 已更新（提供 rods_total），先跑流感的 weekly_update.sh。
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
export PYTHONWARNINGS=ignore
scripts/refresh_data.sh
$PY scripts/build_panel.py | head -3
$PY scripts/forecast_now.py --joint ev_oe ev_out ev_rods --covariates school holiday | grep -v HF_TOKEN
$PY scripts/forecast_now.py --targets ev_rods_pct --covariates school holiday | grep -v HF_TOKEN
$PY ../scripts/log_forecast.py --project ev_forecast
$PY ../scripts/evaluate_forecasts.py --project ev_forecast --quiet
$PY scripts/build_site.py
if [[ "${1:-}" == "--push" ]]; then
  (cd .. && git add docs/ev ev_forecast/data_processed/coverage.json ev_forecast/data_processed/national_weekly.csv \
                    ev_forecast/outputs/latest ev_forecast/outputs/forecast_log \
    && git commit -m "腸病毒每週更新 $(date +%F)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || echo "nothing to commit"; git push)
fi
echo "done. preview: python3 -m http.server 8765 --directory ../docs  →  http://localhost:8765/ev/"
