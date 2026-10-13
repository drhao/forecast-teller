#!/usr/bin/env python
"""
fetch_nidss_visits.py — 從 NIDSS「傳染病統計資料查詢系統」抓新冠（disease=11）健保就診人次，寫入
data/nidss_visits.csv（長格式：yw, area, age, case_type, visits, rate_pct），並把全國全年齡的近 8 週
首報值附加到 data/nidss_visits_history.csv（進 git），供日後估計回補幅度。

- 地區：全國 + 6 區（一次查詢可帶全部地區）；年齡：9 個既定組合（全部、0-2 … 65+）；
  就診類別：門診 + 急診（oe，疾管署新聞稿用的「門急診就診人次」）與 門診（out）。
- 網站對尚未申報的週回 0，io.read_nidss_visits 會把尾端的 0 視為缺值。
- 查詢起點 2021 年第 1 週（新冠門診 ICD 自 2021w27 起有值）。

用法：python scripts/fetch_nidss_visits.py [--start 2021] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_forecast import DATA_DIR  # noqa: E402
from forecast_teller.nidss import AGE, AREA, NIDSS  # noqa: E402
from forecast_teller.weeks import yw_of_date  # noqa: E402

OUT = DATA_DIR / "nidss_visits.csv"
HISTORY = DATA_DIR / "nidss_visits_history.csv"
DISEASE = 11
CASE_TYPES = {"oe": "門診 + 急診", "out": "門診"}
HISTORY_WEEKS = 8


def fetch(start_year: int) -> pd.DataFrame:
    api = NIDSS(pause=0.5)
    cur = yw_of_date(date.today())
    end = (int(cur[:4]), int(cur[4:]))
    frames = []
    for key, ct in CASE_TYPES.items():
        for age in AGE:
            df = api.area_trend(DISEASE, (start_year, 1), end, areas=list(AREA), age=age, case_type=ct)
            df = df.rename(columns={"period": "yw"})[["yw", "area", "age", "visits", "rate_pct"]]
            df["case_type"] = key
            frames.append(df)
            print(f"  {ct} {age:5s}: {len(df)} 列", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["yw"] = out["yw"].astype(str)
    return out.sort_values(["case_type", "area", "age", "yw"]).reset_index(drop=True)


def append_history(df: pd.DataFrame, fetched_on: str) -> int:
    nat = df[(df.area == "全國") & (df.age == "全部") & (df.visits > 0)]
    last = nat[nat.case_type == "oe"].yw.max()
    weeks = sorted(nat.yw.unique())[-HISTORY_WEEKS:]
    new = nat[nat.yw.isin(weeks)][["yw", "case_type", "visits", "rate_pct"]].copy()
    new.insert(0, "fetched_on", fetched_on)
    old = pd.read_csv(HISTORY, dtype={"yw": str}, encoding="utf-8-sig") if HISTORY.exists() else pd.DataFrame(columns=new.columns)
    old = old[old.fetched_on != fetched_on]
    allr = pd.concat([f for f in (old, new) if len(f)], ignore_index=True).sort_values(["fetched_on", "case_type", "yw"])
    allr.to_csv(HISTORY, index=False, encoding="utf-8-sig")
    return len(new)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    df = fetch(args.start)
    nat = df[(df.area == "全國") & (df.age == "全部") & (df.case_type == "oe") & (df.visits > 0)]
    print(f"全國門急診：{nat.yw.min()}–{nat.yw.max()}（最後有值週 {nat.yw.max()}，{int(nat.visits.iloc[-1]):,} 人次）；共 {len(df):,} 列")
    if args.dry_run:
        return 0
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    n = append_history(df, date.today().isoformat())
    print(f"已寫入 {OUT}；{n} 列（fetched_on={date.today()}）附加到 {HISTORY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
