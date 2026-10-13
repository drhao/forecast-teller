#!/usr/bin/env python
"""Rolling-origin backtests for COVID-19 targets.

Window: context from 2021w27 (includes the 2022 Omicron wave; user decision 2026-10-13), origins from 2024w01
to the latest complete week, h = 1–4. Segments are calendar years (y2024, y2025, y2026) plus "all".
Reuses forecast_teller.backtest (Config / run_config / summarize) with the COVID national panel and the
core holiday / CNY / season covariates.

  python scripts/run_backtest.py --suite baselines stats core --target covid_oe
  python scripts/run_backtest.py --suite layer1 layer2 layer3 --target covid_oe
"""
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from covid_forecast import OUTPUT_DIR  # noqa: E402
from covid_forecast.panel import load_national  # noqa: E402
from forecast_teller.backtest import SEGMENTS, Config, run_config, save_results, summarize  # noqa: E402
from forecast_teller.covariates import future_covariates  # noqa: E402

BT = OUTPUT_DIR / "backtest"
START, ORIGIN_START = "202127", "202401"
RODS_START = "202235"          # RODS COVID-19 open data is sparse until 2022w34 (a few rows a week); complete from 2022w35
SEGMENTS.clear(); SEGMENTS.update({"y2024": (2024, 2024), "y2025": (2025, 2025), "y2026": (2026, 2026)})
COMPANIONS = ["covid_out", "covid_rods", "covid_inp"]


def suite(name: str, target: str, H: int, end: str, nat: pd.DataFrame) -> list[Config]:
    first = lambda cols: max([START] + [str(nat[c].first_valid_index()) for c in cols])   # no NaN inside start..end
    others = [c for c in COMPANIONS if c != target]
    # warmup=0: the origin set is fixed by ORIGIN_START for every config, so leaderboards compare the same 141 origins
    base = dict(target=target, horizon=H, start=first([target]), end=end, origin_start=ORIGIN_START, warmup=0)
    hol = dict(covariates=("cny", "holiday"))
    mv = {**base, "start": max(RODS_START, first([target, *others]))}   # multivariate / past-covariate configs include covid_rods
    if name == "baselines":
        return [Config("snaive", model="snaive", **base), Config("naive", model="naive", **base), Config("ma3", model="ma3", **base)]
    if name == "stats":
        return [Config("ets", model="ets", **base), Config("theta", model="theta", **base)]
    if name == "layer1":
        return [Config("tfm_expanding", **base), Config("tfm_expanding_log1p", log1p=True, **base),
                Config("tfm_expanding_sym", symmetric=True, **base), Config("tfm_slide104", window=104, **base),
                Config("tfm_ctx2023", **{**base, "start": "202312"}), Config("tfm_ctx2022w20", **{**base, "start": RODS_START})]
    if name == "layer2":
        return [Config("tfm_cov_hol", covariates=("holiday",), **base), Config("tfm_cov_cny_hol", **hol, **base),
                Config("tfm_cov_season_cny_hol", covariates=("season", "cny", "holiday"), **base),
                Config("tfm_cov_cny_hol_log1p", log1p=True, **hol, **base)]
    if name == "layer3":
        return [Config("tfm_mv_out_rods", targets=(target, *others[:2]), **hol, **mv),
                Config("tfm_mv_out_rods_inp", targets=(target, *others[:3]), **hol, **mv),
                Config("tfm_mv_out_rods_nocov", targets=(target, *others[:2]), **mv),
                Config("tfm_po_rods", past_covariates=("covid_rods",), past_lag=0, **hol, **mv)]
    if name == "core":  # operational set for the primary indicator
        return [Config("snaive", model="snaive", **base), Config("naive", model="naive", **base), Config("ma3", model="ma3", **base),
                Config("ets", model="ets", **base),
                Config("tfm_expanding", **base), Config("tfm_cov_cny_hol", **hol, **base),
                Config("tfm_mv_out_rods", targets=(target, *others[:2]), **hol, **mv),
                Config("tfm_mv_out_rods_inp", targets=(target, *others[:3]), **hol, **mv)]
    raise SystemExit(f"unknown suite {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", required=True)
    ap.add_argument("--target", nargs="+", default=["covid_oe"])
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    nat = load_national()
    model = None
    for target in args.target:
        end = str(nat[[target, *[c for c in COMPANIONS if c != target]]].dropna().index[-1])   # common last complete week
        for s in args.suite:
            cfgs = suite(s, target, args.horizon, end, nat)
            if any(c.model == "timesfm3" for c in cfgs) and model is None:
                from forecast_teller.model_timesfm3 import TimesFM3Model
                model = TimesFM3Model(backend="mlx", batch_size=args.batch)
            res = []
            for cfg in cfgs:
                t0 = time.time()
                res.append(run_config(cfg, model=model if cfg.model == "timesfm3" else None, panel=nat, covariate_fn=future_covariates, verbose=False))
                print(f"  {cfg.name:28s} {len(res[-1]):5d} rows {time.time() - t0:5.1f}s", flush=True)
            df = pd.concat(res, ignore_index=True)
            summ = summarize(df)
            tag = f"{s}_{target}"
            save_results(df, summ, tag, out_dir=BT)
            prim = summ[(summ.target == target) & (summ.segment == "all")]
            piv = prim.pivot_table(index="config", columns="h", values="wis").round(0)
            print(f"\n== {tag} (origins {ORIGIN_START}–, end {end}): WIS by horizon, all origins\n{piv.to_string()}\n", flush=True)


if __name__ == "__main__":
    main()
