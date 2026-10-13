#!/usr/bin/env python
"""
fetch_severe.py — 從 NIDSS 法定傳染病統計頁抓「新冠併發重症」每週確定病例數（發病週），寫入
data/severe_weekly.csv（yw, severe），並把近 12 週的值附加到 data/severe_history.csv（進 git）。

原理同 scripts/fetch_resp_lab.py：頁面的趨勢圖資料以 hcJson.push({...}) 內嵌在 HTML 裡。
頁面預設顯示 2024 年第 1 週至本週；2024/9/1 起才有新定義的個案（舊定義在 id=19CVS，未納入）。
網站窗口外的舊週保留在既有檔案裡。

用法：python scripts/fetch_severe.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_forecast import DATA_DIR  # noqa: E402

URL = "https://nidss.cdc.gov.tw/nndss/disease?id=19SC"
OUT = DATA_DIR / "severe_weekly.csv"
HISTORY = DATA_DIR / "severe_history.csv"
TITLE_KEY = "新冠併發重症"
_HC_RE = re.compile(r"hcJson\.push\((\{.*?\})\);", re.S)
HISTORY_WEEKS = 12


def fetch_chart(url: str = URL) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    for raw in _HC_RE.findall(html):
        if TITLE_KEY in raw and "xAxis_categories" in raw:
            return json.loads(raw.replace("undefined", "null"))
    raise RuntimeError(f"頁面裡找不到「{TITLE_KEY}」趨勢圖的 hcJson 區塊（頁面改版？）")


def chart_to_series(chart: dict) -> pd.Series:
    weeks = [str(w) for w in chart["xAxis_categories"]]
    series = {s["name"]: s["data"] for s in chart["series"]}
    name = next((n for n in series if "確定病例" in n), next(iter(series)))
    vals = [(v.get("y") if isinstance(v, dict) else v) for v in series[name]]
    s = pd.Series([float(v) if v is not None else float("nan") for v in vals], index=weeks[: len(vals)], name="severe")
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    chart = fetch_chart()
    s = chart_to_series(chart).dropna()
    print(f"{chart.get('title_text', '')}：{s.index[0]}–{s.index[-1]}，共 {len(s)} 週；最近 6 週 {s.tail(6).astype(int).to_dict()}")
    old = pd.read_csv(OUT, dtype={"yw": str}, encoding="utf-8-sig").set_index("yw")["severe"] if OUT.exists() else pd.Series(dtype=float)
    merged = pd.concat([old[~old.index.isin(s.index)], s]).sort_index()
    if args.dry_run:
        return 0
    merged.rename("severe").astype(int).to_csv(OUT, index_label="yw", encoding="utf-8-sig")
    today = date.today().isoformat()
    new = s.tail(HISTORY_WEEKS).astype(int).rename("severe").reset_index().rename(columns={"index": "yw"})
    new.insert(0, "fetched_on", today)
    hist = pd.read_csv(HISTORY, dtype={"yw": str}, encoding="utf-8-sig") if HISTORY.exists() else pd.DataFrame(columns=new.columns)
    hist = hist[hist.fetched_on != today]
    hist = pd.concat([f for f in (hist, new) if len(f)], ignore_index=True).sort_values(["fetched_on", "yw"])
    hist.to_csv(HISTORY, index=False, encoding="utf-8-sig")
    print(f"已寫入 {OUT}（{len(merged)} 週）；{len(new)} 列（fetched_on={today}）附加到 {HISTORY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
