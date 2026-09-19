#!/usr/bin/env bash
# 腸病毒每週更新：下載開放資料 → 面板 → 即時預測 → 重建 docs/ev/。
# 用法：ev_forecast/scripts/weekly_update.sh [--push]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
export PYTHONWARNINGS=ignore
scripts/refresh_data.sh
$PY scripts/build_panel.py | head -3
$PY scripts/forecast_now.py --joint ev_oe ev_out ev_rods --covariates school holiday | grep -v HF_TOKEN
$PY scripts/forecast_now.py --targets ev_rods_pct --covariates school holiday | grep -v HF_TOKEN
$PY scripts/build_site.py
if [[ "${1:-}" == "--push" ]]; then
  (cd .. && git add docs/ev ev_forecast/data_processed/coverage.json ev_forecast/data_processed/national_weekly.csv ev_forecast/outputs/latest && git commit -m "ev weekly forecast update $(date +%F)" || echo "nothing to commit"; git push)
fi
echo "done. preview: python3 -m http.server 8765 --directory ../docs  →  http://localhost:8765/ev/"
