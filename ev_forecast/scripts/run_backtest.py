#!/usr/bin/env python
"""Rolling-origin backtests for enterovirus targets (2016–2025 window, origins 2018w01–2025w53, h=1–4).

Reuses forecast_teller.backtest (Config / run_config / summarize) with the EV national panel and
EV covariates (school calendar, two-harmonic season, holidays).

  python scripts/run_backtest.py --suite baselines --target ev_oe ev_out
  python scripts/run_backtest.py --suite layer1 layer2 layer3 --target ev_out
  python scripts/run_backtest.py --suite core --target ev_oe
"""
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from ev_forecast import OUTPUT_DIR  # noqa: E402
from ev_forecast.covariates import future_covariates  # noqa: E402
from ev_forecast.panel import load_national  # noqa: E402
from ev_forecast.thresholds import load_thresholds  # noqa: E402
from forecast_teller.backtest import Config, run_config, save_results, summarize  # noqa: E402

BT = OUTPUT_DIR / "backtest"


def suite(name: str, target: str, H: int) -> list[Config]:
    base = dict(target=target, horizon=H, start="201601", end="202553", warmup=104)
    sh = dict(covariates=("school", "holiday"))
    if name == "baselines":
        return [Config("snaive", model="snaive", **base), Config("naive", model="naive", **base), Config("ma3", model="ma3", **base)]
    if name == "stats":
        return [Config("ets", model="ets", **base), Config("theta", model="theta", **base)]
    if name == "layer1":
        return [Config("tfm_expanding", **base), Config("tfm_expanding_log1p", log1p=True, **base),
                Config("tfm_expanding_sym", symmetric=True, **base), Config("tfm_slide156", window=156, **base),
                Config("tfm_slide260", window=260, **base)]
    if name == "layer2":
        return [Config("tfm_cov_school", covariates=("school",), **base),
                Config("tfm_cov_hol", covariates=("holiday",), **base),
                Config("tfm_cov_cny_hol", covariates=("cny", "holiday"), **base),
                Config("tfm_cov_school_hol", **sh, **base),
                Config("tfm_cov_school_hol_log1p", log1p=True, **sh, **base),
                Config("tfm_cov_season2", covariates=("season2",), **base),
                Config("tfm_cov_season2_school_hol", covariates=("season2", "school", "holiday"), **base),
                Config("tfm_cov_season_school_hol", covariates=("season", "school", "holiday"), **base)]
    if name == "layer3":
        others = [c for c in ("ev_out", "ev_rods", "ev_inp", "ev_oe") if c != target]
        return [Config("tfm_mv_rods", targets=(target, "ev_rods"), **sh, **base),
                Config("tfm_mv_rods_inp", targets=(target, "ev_rods", "ev_inp"), **sh, **base),
                Config("tfm_mv_rods_inp_log1p", targets=(target, "ev_rods", "ev_inp"), log1p=True, **sh, **base),
                Config("tfm_po_rods", past_covariates=("ev_rods",), past_lag=0, **sh, **base),
                Config("tfm_po_rods_inp", past_covariates=("ev_rods", "ev_inp"), past_lag=0, **sh, **base),
                Config("tfm_mv_all", targets=(target, *[o for o in others if o != "ev_oe"][:3]), **sh, **base)]
    if name == "core":  # for the operational indicator ev_oe
        return [Config("snaive", model="snaive", **base), Config("naive", model="naive", **base), Config("ma3", model="ma3", **base),
                Config("tfm_expanding", **base), Config("tfm_expanding_log1p", log1p=True, **base),
                Config("tfm_cov_hol", covariates=("holiday",), **base), Config("tfm_cov_school_hol", **sh, **base),
                Config("tfm_cov_school_hol_log1p", log1p=True, **sh, **base),
                Config("tfm_cov_season2_school_hol", covariates=("season2", "school", "holiday"), **base),
                Config("tfm_mv_out_rods", targets=(target, "ev_out", "ev_rods"), **sh, **base),
                Config("tfm_mv_out_rods_inp", targets=(target, "ev_out", "ev_rods", "ev_inp"), **sh, **base),
                Config("tfm_mv_out_rods_inp_log1p", targets=(target, "ev_out", "ev_rods", "ev_inp"), log1p=True, **sh, **base)]
    raise SystemExit(f"unknown suite {name}")


def thresholds_for(target: str, start: str, end: str) -> dict[str, float]:
    if target != "ev_oe":
        return {}
    t = load_thresholds()
    vals = {int(y): float(v) for y, v in t["threshold"].items() if int(start[:4]) <= int(y) <= int(end[:4])}
    if len(set(vals.values())) > 1:
        print(f"WARNING: threshold differs within the backtest window {vals}; using the most common value")
    return {"ev_oe": max(set(vals.values()), key=list(vals.values()).count)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", required=True)
    ap.add_argument("--target", nargs="+", default=["ev_oe"])
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    nat = load_national()
    model = None
    for target in args.target:
        for s in args.suite:
            cfgs = suite(s, target, args.horizon)
            if any(c.model == "timesfm3" for c in cfgs) and model is None:
                from forecast_teller.model_timesfm3 import TimesFM3Model
                model = TimesFM3Model(backend="mlx", batch_size=args.batch)
            res = []
            for cfg in cfgs:
                t0 = time.time()
                res.append(run_config(cfg, model=model if cfg.model == "timesfm3" else None, panel=nat, covariate_fn=future_covariates))
            df = pd.concat(res, ignore_index=True)
            summ = summarize(df, thresholds=thresholds_for(target, cfgs[0].start, cfgs[0].end))
            tag = f"{s}_{target}"
            save_results(df, summ, tag, out_dir=BT)
            prim = summ[(summ.target == target) & (summ.segment == "post_covid")]
            piv = prim.pivot_table(index="config", columns="h", values="wis").round(0)
            print(f"\n== {tag}: post-COVID WIS by horizon\n{piv.to_string()}\n", flush=True)


if __name__ == "__main__":
    main()
