#!/usr/bin/env python
"""Add hit-rate columns to saved backtest forecasts (no model re-run needed).

National files get y_origin from the national panel; county/age files from the long panels.
  python scripts/add_hit_rates.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller import OUTPUT_DIR  # noqa: E402
from forecast_teller.backtest import add_hit_columns  # noqa: E402
from forecast_teller.panel import load_long, load_national  # noqa: E402

bt = OUTPUT_DIR / "backtest"
nat = load_national()
for f in sorted(bt.glob("*_forecasts.parquet")):
    df = pd.read_parquet(f)
    name = f.name
    if name.startswith(("county_", "age_")):
        by = "county" if name.startswith("county_") else "age"
        indicator = name[len(by) + 1: -len("_forecasts.parquet")]
        long = load_long(by)
        wide = long[long.indicator == indicator].pivot(index="yw", columns=by, values="value")
        y0 = [wide.at[o, g] for o, g in zip(df["origin_yw"], df[by])]
        df["y_origin"] = np.asarray(y0, dtype=float)
        df = add_hit_columns(df)
    else:
        parts = []
        for tgt, g in df.groupby("target", sort=False):
            g = g.copy()
            g["y_origin"] = g["origin_yw"].map(nat[tgt]).astype(float)
            parts.append(add_hit_columns(g))
        df = pd.concat(parts).sort_index()
    assert df[["dir_hit", "dir_hit3", "hit_tol10"]].notna().all().all(), name
    df.to_parquet(f)
    print(f"{name:45s} rows={len(df):7d}  dir_hit={df.dir_hit.mean():.3f}  dir_hit3={df.dir_hit3.mean():.3f}  tol10={df.hit_tol10.mean():.3f}")
