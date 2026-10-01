#!/usr/bin/env bash
# 每週更新：抓 RESP_LAB → 建面板 → 即時預測（最佳設定）→ 記錄預測 → 評估歷次預測 → 重建 GitHub Pages 站台（docs/）。
# 用法：scripts/weekly_update.sh            只重建
#       scripts/weekly_update.sh --push     重建後 git commit + push（只 add 明確路徑）
# 前提：使用者已把更新後的內部 CSV 放進 data/（NHI、RODS、NIDDS、LARS）；RESP_LAB.csv 由本腳本從 NIDSS 網站抓。
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS=ignore
python scripts/fetch_resp_lab.py || echo "RESP_LAB 抓取失敗，沿用現有 data/RESP_LAB.csv"
python scripts/build_panel.py | head -3
python scripts/forecast_now.py --joint nhi_out_ili nhi_er_ili rods_ili --covariates cny holiday | grep -v HF_TOKEN
python scripts/forecast_now.py --targets nhi_out_ili rods_ili_pct nhi_er_ili nidds_severe --covariates cny holiday | grep -v HF_TOKEN
python scripts/log_forecast.py
python scripts/evaluate_forecasts.py --quiet
python scripts/build_site.py
if [[ "${1:-}" == "--push" ]]; then
  git add docs data_processed/coverage.json data_processed/qa_report.md data_processed/national_weekly.csv \
          outputs/latest outputs/forecast_log data/resp_lab_history.csv
  git commit -m "每週更新 $(date +%F)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || echo "nothing to commit"
  git push
fi
echo "done. preview: python3 -m http.server 8765 --directory docs  →  http://localhost:8765/"
