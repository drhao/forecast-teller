#!/usr/bin/env python
"""Township geometry for the real-time KDE (PLAN_erakde §10.6 / §12): from the historical line list, per panel township the
case-weighted mean of the 最小統計區 centroids (representative point) and the village evaluation points with their historical
case share. Written to data/kde_geometry.csv (small, in git) so forecast_now.py can run on any clone without the line list.

  python scripts/build_kde_geometry.py            # panel dengue_all (274 townships)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dengue_ewarn.data import load_line_list, load_panel  # noqa: E402
from dengue_ewarn.kde import line_list_with_coords, load_points  # noqa: E402

counts, _ = load_panel("dengue_all"); series = list(counts.columns); dates = counts.index
raw = pd.read_csv(ROOT / "data" / "Dengue_Daily.csv", encoding="utf-8-sig", dtype=str, low_memory=False)
pts = load_points(line_list_with_coords(raw, load_line_list()), series, dates)
n_cases = np.bincount(pts["s_idx"], weights=pts["n"], minlength=len(series))
rows = [{"kind": "town", "series": s, "x_km": round(float(pts["cx"][i]), 4), "y_km": round(float(pts["cy"][i]), 4), "w": int(n_cases[i])}
        for i, s in enumerate(series) if n_cases[i] > 0]
rows += [{"kind": "village", "series": series[int(si)], "x_km": round(float(x), 4), "y_km": round(float(y), 4), "w": round(float(sh), 5)}
         for x, y, si, sh in zip(pts["ex"], pts["ey"], pts["es"], pts["eshare"])]
out = ROOT / "data" / "kde_geometry.csv"; pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
print(f"{sum(r['kind']=='town' for r in rows)} townships with cases, {sum(r['kind']=='village' for r in rows)} village points → {out} "
      f"(frame: lon0 120.5, lat0 23.0, km; cases 2008-01-01 → {dates[-1].date()})")
