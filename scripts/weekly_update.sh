#!/usr/bin/env bash
# 每週更新：建面板 → 即時預測（最佳設定）→ 重建 GitHub Pages 站台（docs/）。
# 用法：scripts/weekly_update.sh            只重建
#       scripts/weekly_update.sh --push     重建後 git commit + push（需先 git init 並設定 remote）
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONWARNINGS=ignore
python scripts/build_panel.py | head -3
python scripts/forecast_now.py --joint nhi_out_ili nhi_er_ili rods_ili --covariates cny holiday | grep -v HF_TOKEN
python scripts/forecast_now.py --targets nhi_out_ili rods_ili_pct nhi_er_ili nidds_severe --covariates cny holiday | grep -v HF_TOKEN
python scripts/build_site.py
if [[ "${1:-}" == "--push" ]]; then
  git add docs data_processed/coverage.json data_processed/qa_report.md outputs/latest
  git commit -m "weekly forecast update $(date +%F)" || echo "nothing to commit"
  git push
fi
echo "done. preview: python3 -m http.server 8765 --directory docs  →  http://localhost:8765/"
