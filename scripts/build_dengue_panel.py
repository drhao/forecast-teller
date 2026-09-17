#!/usr/bin/env python
"""Build data_processed/dengue_township_daily.parquet (+ delay histogram) from data/Dengue_Daily.csv."""
import sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller.dengue import build, asof_counts, completeness, rolling7  # noqa: E402

t0 = time.time(); res = build(); c = res["counts"]; hist = res["hist"]
print(f"built in {time.time()-t0:.0f}s | {c.shape[1]} series × {c.shape[0]} days ({c.index[0].date()} → {c.index[-1].date()})")
print("local cases per year (台南+高雄):", res["yearly"].astype(int).to_dict())
city = c.T.groupby(c.columns.str.split("|").str[0]).sum().T
peak = {k: (v.idxmax().date().isoformat(), int(v.max())) for k, v in rolling7(city.T.to_numpy()).T.__array__().__class__ and {col: __import__("pandas").Series(rolling7(city[col].to_numpy()), index=city.index) for col in city.columns}.items()}
print("peak 7-day sum by city:", peak)
# report delay
ll = res["line_list"]; loc = ll[(~ll.imported) & ll.county.isin(("台南市", "高雄市")) & (ll.onset >= "2012-01-01")]
print("report delay (days) quantiles 50/75/90/95:", loc.delay.quantile([.5, .75, .9, .95]).astype(int).tolist())
d = c.index.get_loc(__import__("pandas").Timestamp("2015-09-15"))
asof = asof_counts(hist, d); final = c.to_numpy().T[:, : d + 1]
print(f"as-of check 2015-09-15: last 7 days known {int(asof[:, -7:].sum())} vs final {int(final[:, -7:].sum())} ({100*asof[:, -7:].sum()/max(final[:, -7:].sum(),1):.0f}% complete); completeness k=0..7:", np.round(completeness(hist, d)[:8], 2).tolist())
