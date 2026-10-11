#!/usr/bin/env bash
# 登革熱每日預警：抓 NIDSS 快照 → 即時預測（TimesFM 3.0，asof_adj）→ 重建 docs/dengue/ → [--push]
# 用法：dengue_ewarn/scripts/daily_update.sh [--push]
# 台南 + 高雄 + 屏東共 108 鄉鎮，抓取約 13 分鐘。可在任一 clone 跑（資料來自網站）；重建站台需要本機有 outputs/backtest/*.parquet，沒有就只更新 latest.json。
set -euo pipefail
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
export PYTHONWARNINGS=ignore
$PY scripts/fetch_nidss.py
$PY scripts/forecast_now.py | grep -v HF_TOKEN
if ls outputs/backtest/dengue_all_*_forecasts.parquet >/dev/null 2>&1; then
  $PY scripts/build_site.py --tag dengue_all --panel dengue_all | tail -1
else
  echo "缺 outputs/backtest parquet，跳過 build_site（docs/dengue/latest.json 已更新，index.html 讀取時會自行載入）"
fi
if [[ "${1:-}" == "--push" ]]; then
  (cd .. && git add docs/dengue/latest.json docs/dengue/index.html \
                    dengue_ewarn/data/nidss_latest.csv dengue_ewarn/data/nidss_latest_meta.json dengue_ewarn/data/nidss_history.csv \
                    dengue_ewarn/outputs/latest/dengue_alerts.csv dengue_ewarn/outputs/forecast_log/dengue_forecast_log.csv \
    && git commit -m "登革熱每日預警 $(date +%F)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" || echo "nothing to commit"; git push)
fi
echo "done. preview: python3 -m http.server 8765 --directory ../docs  →  http://localhost:8765/dengue/"
