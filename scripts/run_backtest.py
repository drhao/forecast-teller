#!/usr/bin/env python
"""Run rolling-origin backtests for a target series.

Examples:
  python scripts/run_backtest.py --target nhi_out_ili --suite layer1
  python scripts/run_backtest.py --target rods_ili_pct --suite layer1 --tag rods
"""
import argparse, sys, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller.backtest import Config, run_config, save_results, summarize  # noqa: E402
from forecast_teller.model_timesfm3 import TimesFM3Model  # noqa: E402
from forecast_teller.report import leaderboard, wis_by_horizon_table  # noqa: E402


def suite(name: str, target: str, H: int) -> list[Config]:
    base = dict(target=target, horizon=H)
    if name == "baselines":
        return [Config("snaive", model="snaive", **base), Config("naive", model="naive", **base), Config("ma3", model="ma3", **base)]
    if name == "ma3":
        return [Config("ma3", model="ma3", **base)]
    if name == "stats":
        return [Config("ets", model="ets", **base), Config("theta", model="theta", **base)]
    if name == "layer1":
        return [
            Config("snaive", model="snaive", **base),
            Config("naive", model="naive", **base),
            Config("tfm_expanding", **base),
            Config("tfm_expanding_log1p", log1p=True, **base),
            Config("tfm_expanding_sym", symmetric=True, **base),
            Config("tfm_slide260", window=260, **base),
            Config("tfm_slide156", window=156, **base),
        ]
    if name == "layer2":
        return [
            Config("tfm_cov_season", covariates=("season",), **base),
            Config("tfm_cov_cny", covariates=("cny",), **base),
            Config("tfm_cov_season_cny", covariates=("season", "cny"), **base),
            Config("tfm_cov_season_cny_hol", covariates=("season", "cny", "holiday"), **base),
            Config("tfm_cov_season_cny_log1p", covariates=("season", "cny"), log1p=True, **base),
        ]
    if name == "core":
        # compact suite for additional national targets: baselines, best zero-shot variants,
        # holiday covariates and the multivariate combinations (target first, then the others)
        others = [c for c in ["nhi_out_ili", "nhi_er_ili", "rods_ili", "nidds_severe"] if c != target]
        hol = dict(covariates=("season", "cny", "holiday")); ch = dict(covariates=("cny", "holiday"))
        return [
            Config("snaive", model="snaive", **base), Config("naive", model="naive", **base), Config("ma3", model="ma3", **base),
            Config("tfm_expanding", **base), Config("tfm_expanding_sym", symmetric=True, **base), Config("tfm_slide260", window=260, **base),
            Config("tfm_cov_hol", covariates=("holiday",), **base), Config("tfm_cov_cny_hol", **ch, **base), Config("tfm_cov_season_cny_hol", **hol, **base),
            Config("tfm_best_mv4_cny_hol", targets=(target, *others), **ch, **base), Config("tfm_best_mv4_hol", targets=(target, *others), **hol, **base),
            Config("tfm_best_mv3_hol", targets=(target, *others[:2]), **hol, **base),
            Config("tfm_best_po_rods_hol", past_covariates=("rods_ili_pct",), past_lag=1, **hol, **base),
        ]
    if name == "layer2b":
        return [
            Config("tfm_cov_hol", covariates=("holiday",), **base),
            Config("tfm_cov_cny_hol", covariates=("cny", "holiday"), **base),
            Config("tfm_cov_season_hol", covariates=("season", "holiday"), **base),
            Config("tfm_cov_season_cny_hol_sym", covariates=("season", "cny", "holiday"), symmetric=True, **base),
            Config("tfm_cov_season_cny_hol_slide260", covariates=("season", "cny", "holiday"), window=260, **base),
        ]
    if name == "resp":
        # recent-period experiment (origins 2025w09–2026w32, targets to 2026w36): does the community
        # contract-lab respiratory PCR panel (RESP_LAB, from 2025w01) help as a past-only covariate?
        # NIDDS (onset week) lags ~3 weeks behind the other sources, so the joint configs here use the
        # three timely series only (target + outpatient/ER + RODS)
        others = [c for c in ["nhi_out_ili", "nhi_er_ili", "rods_ili"] if c != target]
        rb = dict(target=target, horizon=H, end="202636", origin_start="202509")
        ch = dict(covariates=("cny", "holiday"))
        return [
            Config("snaive", model="snaive", **rb), Config("naive", model="naive", **rb), Config("ma3", model="ma3", **rb),
            Config("tfm_cov_cny_hol", **ch, **rb),
            Config("tfm_mv3_cny_hol", targets=(target, *others), **ch, **rb),
            Config("tfm_po_resp_flu", past_covariates=("resp_flu", "resp_flu_pos_rate"), past_lag=1, **ch, **rb),
            Config("tfm_po_resp_flu_share", past_covariates=("resp_flu", "resp_flu_share", "resp_pos_pct"), past_lag=1, **ch, **rb),
            Config("tfm_po_resp_all", past_covariates=("resp_flu", "resp_rsv", "resp_covid", "resp_hmpv", "resp_pos_pct"), past_lag=1, **ch, **rb),
            Config("tfm_po_lars", past_covariates=("lab_flu_a", "lab_pos_rate"), past_lag=1, **ch, **rb),
            Config("tfm_po_resp_lars", past_covariates=("resp_flu", "resp_flu_pos_rate", "lab_flu_a", "lab_pos_rate"), past_lag=1, **ch, **rb),
            Config("tfm_mv3_po_resp", targets=(target, *others), past_covariates=("resp_flu", "resp_flu_pos_rate"), past_lag=1, **ch, **rb),
            Config("tfm_po_resp_flu_slide104", past_covariates=("resp_flu", "resp_flu_pos_rate"), past_lag=1, window=104, **ch, **rb),
        ]
    if name == "resp_smooth":
        rb = dict(target=target, horizon=H, end="202636", origin_start="202509"); ch = dict(covariates=("cny", "holiday"))
        return [Config("tfm_po_resp_smooth", past_covariates=("resp_flu_ma3", "resp_flu_pos_rate_ma3"), past_lag=1, **ch, **rb),
                Config("tfm_po_resp_smooth_lag2", past_covariates=("resp_flu_ma3", "resp_flu_pos_rate_ma3"), past_lag=2, **ch, **rb)]
    if name == "layer3":
        cov = dict(covariates=("season", "cny"))
        return [
            Config("tfm_po_lab", past_covariates=("lab_flu_a", "lab_flu_b"), past_lag=1, **cov, **base),
            Config("tfm_po_labpos", past_covariates=("lab_pos_rate", "lab_a_share"), past_lag=1, **cov, **base),
            Config("tfm_po_rods", past_covariates=("rods_ili_pct",), past_lag=1, **cov, **base),
            Config("tfm_po_all_lag1", past_covariates=("lab_flu_a", "lab_flu_b", "rods_ili_pct", "nidds_severe"), past_lag=1, **cov, **base),
            Config("tfm_po_all_lag2", past_covariates=("lab_flu_a", "lab_flu_b", "rods_ili_pct", "nidds_severe"), past_lag=2, **cov, **base),
            Config("tfm_mv_out_er_rods", targets=(target, "nhi_er_ili", "rods_ili"), **cov, **base),
            Config("tfm_mv_out_er_rods_sev", targets=(target, "nhi_er_ili", "rods_ili", "nidds_severe"), **cov, **base),
            Config("tfm_mv_out_er_rods_nocov", targets=(target, "nhi_er_ili", "rods_ili"), **base),
        ]
    if name == "layer4":
        hol = dict(covariates=("season", "cny", "holiday"))
        return [
            Config("tfm_best_mv4_hol", targets=(target, "nhi_er_ili", "rods_ili", "nidds_severe"), **hol, **base),
            Config("tfm_best_mv3_hol", targets=(target, "nhi_er_ili", "rods_ili"), **hol, **base),
            Config("tfm_best_mv4_cny_hol", targets=(target, "nhi_er_ili", "rods_ili", "nidds_severe"), covariates=("cny", "holiday"), **base),
            Config("tfm_best_po_rods_hol", past_covariates=("rods_ili_pct",), past_lag=1, **hol, **base),
            Config("tfm_best_mv4_hol_slide260", targets=(target, "nhi_er_ili", "rods_ili", "nidds_severe"), window=260, **hol, **base),
        ]
    raise ValueError(name)


ap = argparse.ArgumentParser()
ap.add_argument("--target", default="nhi_out_ili")
ap.add_argument("--suite", default="layer1")
ap.add_argument("--horizon", type=int, default=4)
ap.add_argument("--tag", default=None)
ap.add_argument("--batch-size", type=int, default=8)
args = ap.parse_args()

cfgs = suite(args.suite, args.target, args.horizon)
model = TimesFM3Model(backend="mlx", batch_size=args.batch_size) if any(c.model == "timesfm3" for c in cfgs) else None
if model:
    print(f"model loaded in {model.load_seconds:.1f}s")
t0 = time.time()
results = pd.concat([run_config(c, model=model) for c in cfgs], ignore_index=True)
summary = summarize(results)
tag = args.tag or f"{args.target}_{args.suite}"
out = save_results(results, summary, tag)
print(f"\nsaved to {out}/{tag}_* in {time.time()-t0:.0f}s\n")
for seg in ["post_covid", "pre_covid", "covid", "all"]:
    print(f"== {seg}: mean over h=1..4 ==")
    print(leaderboard(summary, seg, target=args.target).round(3).to_string())
    print()
print("== WIS by horizon (post_covid) =="); print(wis_by_horizon_table(summary, "post_covid", target=args.target).round(1).to_string())
