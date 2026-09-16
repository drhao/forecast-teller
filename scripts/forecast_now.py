#!/usr/bin/env python
"""Live forecast from the latest complete week for national indicators.

Univariate (one model call per indicator):
  python scripts/forecast_now.py --targets nhi_out_ili rods_ili_pct nhi_er_ili nidds_severe --covariates cny holiday
Joint multivariate (best backtest config; contexts truncated to the common last complete week):
  python scripts/forecast_now.py --joint nhi_out_ili nhi_er_ili rods_ili nidds_severe --covariates cny holiday
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller import OUTPUT_DIR  # noqa: E402
from forecast_teller.covariates import future_covariates  # noqa: E402
from forecast_teller.metrics import Q_LEVELS  # noqa: E402
from forecast_teller.model_timesfm3 import TimesFM3Model  # noqa: E402
from forecast_teller.panel import load_national  # noqa: E402
from forecast_teller.report import plot_latest_forecast  # noqa: E402
from forecast_teller.weeks import next_weeks, week_start  # noqa: E402

LABELS = {"nhi_out_ili": "全國類流感門診就診人次", "rods_ili_pct": "RODS 急診類流感就診百分比 (%)",
          "rods_ili": "RODS 急診類流感就診人次", "nhi_er_ili": "全國類流感急診就診人次",
          "nidds_severe": "流感併發重症週病例數（發病週）", "nhi_out_ili_rate": "類流感門診就診率 (%)"}

ap = argparse.ArgumentParser()
ap.add_argument("--targets", nargs="+", default=["nhi_out_ili", "rods_ili_pct", "nhi_er_ili", "nidds_severe"])
ap.add_argument("--joint", nargs="*", default=None, help="forecast these indicators jointly (first = primary)")
ap.add_argument("--horizon", type=int, default=4)
ap.add_argument("--covariates", nargs="*", default=["cny", "holiday"])
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
    for h in range(H):
        rows.append({"target": tgt, "label": LABELS.get(tgt, tgt), "mode": mode, "origin_yw": hist.index[-1], "target_yw": tw[h],
                     "target_week_start": week_start(tw[h]).date(), "h": h + 1, "median": med[h],
                     **{f"q{int(l*100)}": q[h, i] for i, l in enumerate(Q_LEVELS)}})
    plot_latest_forecast(hist, med, q, tw, out_dir / f"forecast_{mode}_{tgt}.png",
                         title=f"{LABELS.get(tgt, tgt)}：自 {hist.index[-1]} 起 {H} 週預測（TimesFM 3.0, {mode}）")
    print(f"{tgt:14s} [{mode}] origin {hist.index[-1]} → {tw[0]}..{tw[-1]}  median {np.round(med, 1).tolist()}  "
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
