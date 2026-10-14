#!/usr/bin/env python
"""Village-level hotspot backtest (PLAN_erakde §11): scores every named village at every origin.

  python scripts/run_village_backtest.py --tag village --years 2010 ... 2024 --modes asof_adj final --models own town kde_05 ...
Writes outputs/backtest/{tag}_{mode}_village.parquet (origin × village rows; float32 score columns per model).
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import OUTPUT_DIR  # noqa: E402
from dengue_ewarn.data import load_panel  # noqa: E402
from dengue_ewarn.village import H, MODELS, VillageData, VillageForecaster  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="village")
ap.add_argument("--years", nargs="+", type=int, default=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2018, 2019, 2020, 2023, 2024])
ap.add_argument("--modes", nargs="+", default=["asof_adj"])
ap.add_argument("--models", nargs="+", default=list(MODELS))
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--season", nargs=2, default=["06-01", "12-31"])
args = ap.parse_args()

root = Path(__file__).resolve().parents[1]
counts, hist = load_panel("dengue_all"); dates = counts.index
vil = pd.read_parquet(root / "data_processed" / "villages.parquet"); cases = pd.read_parquet(root / "data_processed" / "village_cases.parquet")
D = VillageData(vil, cases, dates, hist)
print(f"{D.V} villages ({(D.region == '台南高雄').sum()} 台南高雄), {len(D.v):,} cases, {len(D.town_codes)} townships", flush=True)
fcs = {m: VillageForecaster(D, MODELS[m]) for m in args.models}
out = OUTPUT_DIR / "backtest"; out.mkdir(parents=True, exist_ok=True)
T = len(dates)
for mode in args.modes:
    frames = []
    for year in args.years:
        t0 = time.time()
        lo = dates.get_indexer([pd.Timestamp(f"{year}-{args.season[0]}")])[0]; hi = dates.get_indexer([pd.Timestamp(f"{year}-{args.season[1]}")])[0]
        if lo < 0 or hi < 0:
            print(f"  skip {year}"); continue
        origins = [d for d in range(lo, hi + 1, args.step) if d + H < T]
        for d in origins:
            known = D.known(d, mode); m, w, age = known
            raw14 = np.bincount(D.v[m][age < 14], weights=D.n[m][age < 14], minlength=D.V)   # reported, unadjusted
            y = D.truth(d)
            df = {"origin": np.full(D.V, dates[d]), "v": np.arange(D.V, dtype=np.int32), "y": y.astype(np.float32),
                  "own_raw": raw14.astype(np.float32)}
            for name, fc in fcs.items():
                df[name] = fc.score(fc.mu(d, mode, known)).astype(np.float32)
            frames.append(pd.DataFrame(df))
        n_ev = sum(int((f.y >= 2).sum()) for f in frames[-len(origins):])
        print(f"  [{mode}] {year}: {len(origins)} origins × {D.V} villages in {time.time()-t0:.0f}s | village-origins with ≥2 new cases: {n_ev:,}", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res["year"] = res.origin.dt.year.astype(np.int16)
    res.to_parquet(out / f"{args.tag}_{mode}_village.parquet")
    print(f"saved {args.tag}_{mode}_village.parquet ({len(res):,} rows)", flush=True)
