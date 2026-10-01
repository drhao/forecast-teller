#!/usr/bin/env python
"""Build data_processed/{tag}_township_daily.parquet (+ delay histogram {tag}_delay_hist.npz) from data/Dengue_Daily.csv.

  python scripts/build_panel.py                             # 台南 + 高雄（tag dengue）
  python scripts/build_panel.py --scope all --tag dengue_all  # 全台有本土病例的鄉鎮（第三輪）
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn.data import CITIES, SCOPES, asof_counts, build, completeness, rolling7  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scope", default="tn_kh", choices=sorted(SCOPES))
ap.add_argument("--tag", default="dengue", help="panel file prefix under data_processed/")
args = ap.parse_args()

t0 = time.time(); res = build(scope=args.scope, tag=args.tag); c = res["counts"]; hist = res["hist"]
counties = sorted({s.split("|")[0] for s in c.columns})
print(f"built in {time.time()-t0:.0f}s | scope {args.scope} → tag {args.tag} | {c.shape[1]} series in {len(counties)} counties × {c.shape[0]} days ({c.index[0].date()} → {c.index[-1].date()})")
print("local cases per year:", res["yearly"].astype(int).to_dict())
county_daily = c.T.groupby(c.columns.str.split("|").str[0]).sum().T          # day × county
peak = {}
for col in county_daily.columns:
    if county_daily[col].sum() >= 100:
        s7 = pd.Series(rolling7(county_daily[col].to_numpy()), index=county_daily.index)
        peak[col] = (s7.idxmax().date().isoformat(), int(s7.max()))
print("peak 7-day sum by county (counties with >= 100 cases):", peak)
ll = res["line_list"]; loc = ll[(~ll.imported) & (ll.onset >= c.index[0])]
if SCOPES[args.scope]:
    loc = loc[loc.county.isin(SCOPES[args.scope])]
print("report delay (days) quantiles 50/75/90/95:", loc.delay.quantile([.5, .75, .9, .95]).astype(int).tolist())
d = c.index.get_loc(pd.Timestamp("2015-09-15"))
asof = asof_counts(hist, d); final = c.to_numpy().T[:, : d + 1]
print(f"as-of check 2015-09-15: last 7 days known {int(asof[:, -7:].sum())} vs final {int(final[:, -7:].sum())} "
      f"({100 * asof[:, -7:].sum() / max(final[:, -7:].sum(), 1):.0f}% complete); completeness k=0..7:", np.round(completeness(hist, d)[:8], 2).tolist())
