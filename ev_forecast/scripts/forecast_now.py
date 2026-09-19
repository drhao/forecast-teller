#!/usr/bin/env python
"""Live enterovirus forecast from the latest complete week.

  python scripts/forecast_now.py --targets ev_oe ev_out ev_rods ev_rods_pct --covariates school holiday
  python scripts/forecast_now.py --joint ev_oe ev_out ev_rods --covariates school holiday
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ev_forecast import OUTPUT_DIR  # noqa: E402
from ev_forecast.covariates import future_covariates, school_calendar  # noqa: E402
from ev_forecast.panel import LABELS, load_national  # noqa: E402
from forecast_teller.covariates import holiday_weekly  # noqa: E402
from forecast_teller.metrics import Q_LEVELS  # noqa: E402
from forecast_teller.model_timesfm3 import TimesFM3Model  # noqa: E402
from forecast_teller.report import plot_latest_forecast  # noqa: E402
from forecast_teller.weeks import next_weeks, week_start  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--targets", nargs="+", default=["ev_oe", "ev_out", "ev_rods", "ev_rods_pct"])
ap.add_argument("--joint", nargs="*", default=None)
ap.add_argument("--horizon", type=int, default=4)
ap.add_argument("--covariates", nargs="*", default=["school", "holiday"])
ap.add_argument("--window", default="expanding")
ap.add_argument("--start", default="201601")
ap.add_argument("--log1p", action="store_true")
ap.add_argument("--symmetric", action="store_true")
args = ap.parse_args()

nat = load_national()
model = TimesFM3Model(backend="mlx", batch_size=8)
out_dir = OUTPUT_DIR / "latest"; out_dir.mkdir(parents=True, exist_ok=True)
H = args.horizon
kinds = tuple(args.covariates) if args.covariates else ()
rows = []


def emit(tgt, hist, med, q, tw, mode):
    sc = school_calendar(tw); hol = holiday_weekly(list(hist.index[-4:]) + tw).loc[tw]
    for h in range(H):
        rows.append({"target": tgt, "label": LABELS.get(tgt, tgt), "mode": mode, "origin_yw": hist.index[-1], "target_yw": tw[h],
                     "target_week_start": week_start(tw[h]).date(), "h": h + 1, "median": med[h],
                     **{f"q{int(l*100)}": q[h, i] for i, l in enumerate(Q_LEVELS)},
                     "summer_break": int(sc.iloc[h]["summer_break"]), "winter_break": int(sc.iloc[h]["winter_break"]),
                     "reopen": int(sc.iloc[h]["reopen"]), "holiday_days": int(hol.iloc[h]["holiday_days"]), "cny": int(hol.iloc[h]["cny_week"])})
    plot_latest_forecast(hist, med, q, tw, out_dir / f"forecast_{mode}_{tgt}.png",
                         title=f"{LABELS.get(tgt, tgt)}：自 {hist.index[-1]} 起 {H} 週預測（TimesFM 3.0, {mode}）")
    print(f"{tgt:12s} [{mode}] origin {hist.index[-1]} → {tw[0]}..{tw[-1]}  median {np.round(med, 1).tolist()}  "
          f"q10 {np.round(q[:, 0], 1).tolist()}  q90 {np.round(q[:, 8], 1).tolist()}")


if args.joint:
    tg = args.joint
    series = {t: nat[t].loc[args.start:].dropna() for t in tg}
    common_end = min(s.index[-1] for s in series.values())
    yws = [w for w in series[tg[0]].index if w <= common_end]
    if args.window != "expanding":
        yws = yws[-int(args.window):]
    Y = np.stack([series[t].loc[yws].to_numpy(np.float64) for t in tg])
    pf = future_covariates(yws, H, kinds=kinds) if kinds else None
    (med, q), = model.forecast([Y], H, past_future=[pf] if pf is not None else None, log1p=args.log1p, symmetric=args.symmetric)
    tw = next_weeks(yws[-1], H)
    print(f"joint forecast of {tg}: common last complete week {common_end}, context {len(yws)} weeks")
    for v, t in enumerate(tg):
        emit(t, series[t].loc[yws], med[v], q[v], tw, "joint")
    csv = out_dir / "latest_forecast_joint.csv"
else:
    for tgt in args.targets:
        s = nat[tgt].loc[args.start:].dropna()
        if args.window != "expanding":
            s = s.iloc[-int(args.window):]
        yws, y = list(s.index), s.to_numpy(np.float64)
        pf = future_covariates(yws, H, kinds=kinds) if kinds else None
        (med, q), = model.forecast([y], H, past_future=[pf] if pf is not None else None, log1p=args.log1p, symmetric=args.symmetric)
        emit(tgt, s, med, q, next_weeks(yws[-1], H), "uni")
    csv = out_dir / "latest_forecast.csv"

pd.DataFrame(rows).to_csv(csv, index=False, encoding="utf-8-sig")
print(f"saved {csv} and figures")
