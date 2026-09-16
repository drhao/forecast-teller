"""Rolling-origin backtest engine (expanding or sliding window) for TimesFM 3.0 and baselines.

Supports univariate targets, past+future covariates (season / cny / holiday), past-only
covariates taken from other national indicators (lagged by the reporting delay), and
multivariate joint forecasting of several national indicators (cfg.targets).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import OUTPUT_DIR
from .baselines import moving_average, naive_last, seasonal_naive, statsforecast_models
from .covariates import future_covariates
from .metrics import (Q_LEVELS, abs_err, covered, directional_hit, directional_hit3, mase_scale,
                      threshold_flags, threshold_rates, tolerance_hit, wis)
from .panel import load_national
from .weeks import yw_range

SEGMENTS = {"pre_covid": (2018, 2019), "covid": (2020, 2022), "post_covid": (2023, 2025)}

# Epidemic thresholds used for the threshold hit rate. Targets not listed use the
# THRESHOLD_PCT quantile of actuals within the backtest window.
DEFAULT_THRESHOLDS = {"rods_ili_pct": 10.0}  # 急診類流感就診百分比：使用者指定 10%
THRESHOLD_PCT = 0.75


def threshold_note(target: str, user_value: float | None = None) -> str:
    if user_value is not None:
        return f"使用者指定 {user_value:g}"
    if target in DEFAULT_THRESHOLDS:
        return f"流行閾值 {DEFAULT_THRESHOLDS[target]:g}%（使用者指定）"
    return "資料窗內實際值第 75 百分位"


@dataclass
class Config:
    name: str
    model: str = "timesfm3"                 # timesfm3 | snaive | naive | ma3 | ets | theta
    target: str = "nhi_out_ili"             # primary target (national panel column)
    targets: tuple[str, ...] = ()           # if set: joint multivariate forecast of these columns (target must be first)
    window: str | int = "expanding"         # 'expanding' or number of context weeks (sliding)
    log1p: bool = False
    symmetric: bool = False
    covariates: tuple[str, ...] = ()        # past+future kinds: season, cny, holiday
    past_covariates: tuple[str, ...] = ()   # national panel columns used as past-only covariates
    past_lag: int = 1                       # reporting delay applied to past-only covariates (weeks)
    start: str = "201601"
    end: str = "202553"
    warmup: int = 104
    horizon: int = 4


def load_series(targets: tuple[str, ...], start: str, end: str) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    """Returns (yws, Y (V, N), panel slice). Raises if any target has missing weeks."""
    nat = load_national()
    yws = yw_range(start, end)
    sub = nat.loc[yws]
    Y = sub[list(targets)].to_numpy(np.float64).T
    if np.isnan(Y).any():
        bad = [(t, int(np.isnan(Y[i]).sum())) for i, t in enumerate(targets) if np.isnan(Y[i]).any()]
        raise ValueError(f"missing weeks in {start}..{end}: {bad}")
    return yws, Y, sub


def _past_only_matrix(sub: pd.DataFrame, cols: tuple[str, ...], lag: int) -> np.ndarray:
    return sub[list(cols)].shift(lag).bfill().to_numpy(np.float32).T  # (K, N)


def run_config(cfg: Config, model=None, verbose: bool = True) -> pd.DataFrame:
    """Long DataFrame: one row per (target variate, origin, h) with truth, median, q10..q90 and scores."""
    targets = tuple(cfg.targets) if cfg.targets else (cfg.target,)
    if cfg.targets and cfg.targets[0] != cfg.target:
        raise ValueError("cfg.target must be the first entry of cfg.targets")
    yws, Y, sub = load_series(targets, cfg.start, cfg.end)
    V, N, H = Y.shape[0], Y.shape[1], cfg.horizon
    origins = list(range(cfg.warmup, N - H + 1))
    t0 = time.time()

    def ctx_slice(t):
        lo = 0 if cfg.window == "expanding" else max(0, t - int(cfg.window))
        return lo, t

    preds: list[tuple[np.ndarray, np.ndarray]] = []
    if cfg.model == "timesfm3":
        assert model is not None, "pass a TimesFM3Model"
        contexts, pfs, pos = [], [], []
        pf_full = future_covariates(yws, H, kinds=cfg.covariates) if cfg.covariates else None
        po_full = _past_only_matrix(sub, cfg.past_covariates, cfg.past_lag) if cfg.past_covariates else None
        for t in origins:
            lo, hi = ctx_slice(t)
            contexts.append(Y[:, lo:hi] if V > 1 else Y[0, lo:hi])
            pfs.append(None if pf_full is None else pf_full[:, lo:hi + H])
            pos.append(None if po_full is None else po_full[:, lo:hi])
        preds = model.forecast(contexts, H,
                               past_future=pfs if pf_full is not None else None,
                               past_only=pos if po_full is not None else None,
                               log1p=cfg.log1p, symmetric=cfg.symmetric)
    else:
        if V > 1:
            raise ValueError("baselines are univariate")
        for t in origins:
            lo, hi = ctx_slice(t)
            c = Y[0, lo:hi]
            if cfg.model == "snaive":
                preds.append(seasonal_naive(c, H))
            elif cfg.model == "naive":
                preds.append(naive_last(c, H))
            elif cfg.model == "ma3":
                preds.append(moving_average(c, H, window=3))
            elif cfg.model in ("ets", "theta"):
                preds.append(statsforecast_models(c, H, models=(cfg.model,))[cfg.model])
            else:
                raise ValueError(cfg.model)

    rows = []
    for t, (med, q) in zip(origins, preds):
        lo, hi = ctx_slice(t)
        med2, q2 = (med[None, :], q[None, :, :]) if med.ndim == 1 else (med, q)
        for v, tgt in enumerate(targets):
            truth = Y[v, t:t + H]
            scale = mase_scale(Y[v, :t])
            w = wis(truth, q2[v]); c80, c60 = covered(truth, q2[v], 80), covered(truth, q2[v], 60)
            ae = abs_err(truth, med2[v])
            y0 = Y[v, t - 1]
            dh, dh3, th = directional_hit(y0, truth, med2[v]), directional_hit3(y0, truth, med2[v]), tolerance_hit(truth, med2[v])
            for h in range(H):
                rows.append({
                    "config": cfg.name, "model": cfg.model, "target": tgt,
                    "origin_yw": yws[t - 1], "target_yw": yws[t + h], "h": h + 1,
                    "target_year": int(yws[t + h][:4]), "context_len": hi - lo,
                    "y": truth[h], "median": med2[v][h],
                    **{f"q{int(l*100)}": q2[v][h, i] for i, l in enumerate(Q_LEVELS)},
                    "abs_err": ae[h], "wis": w[h], "cov80": bool(c80[h]), "cov60": bool(c60[h]),
                    "mase_scale": scale, "y_origin": y0,
                    "dir_hit": bool(dh[h]), "dir_hit3": bool(dh3[h]), "hit_tol10": bool(th[h]),
                })
    df = pd.DataFrame(rows)
    if verbose:
        prim = df[df.target == cfg.target]
        print(f"[{cfg.name}] {len(origins)} origins x h={H} in {time.time()-t0:.1f}s | "
              f"WIS h1-{H} ({cfg.target}) = {prim.groupby('h')['wis'].mean().round(1).tolist()}", flush=True)
    return df


def segment_of(year: int) -> str:
    for name, (a, b) in SEGMENTS.items():
        if a <= year <= b:
            return name
    return "other"


def add_hit_columns(results: pd.DataFrame, y_origin: pd.Series | None = None) -> pd.DataFrame:
    """Add dir_hit / dir_hit3 / hit_tol10 to results (needs y_origin column or a Series to join on origin_yw)."""
    df = results.copy()
    if "y_origin" not in df.columns:
        if y_origin is None:
            raise ValueError("y_origin needed")
        df["y_origin"] = df["origin_yw"].map(y_origin)
    y0, y, m = df["y_origin"].to_numpy(float), df["y"].to_numpy(float), df["median"].to_numpy(float)
    df["dir_hit"] = (y > y0) == (m > y0)
    band = 0.05
    cy = np.where((y - y0) / np.maximum(np.abs(y0), 1e-9) > band, 1, np.where((y - y0) / np.maximum(np.abs(y0), 1e-9) < -band, -1, 0))
    cm = np.where((m - y0) / np.maximum(np.abs(y0), 1e-9) > band, 1, np.where((m - y0) / np.maximum(np.abs(y0), 1e-9) < -band, -1, 0))
    df["dir_hit3"] = cy == cm
    df["hit_tol10"] = np.abs(m - y) <= 0.10 * np.abs(y)
    return df


def summarize(results: pd.DataFrame, thresholds: dict[str, float] | None = None, threshold_pct: float = 0.75) -> pd.DataFrame:
    """Per (config, target, segment, h): WIS, MAE, WAPE, MASE, coverage, hit rates, threshold metrics, n,
    and WIS relative to snaive. Threshold per target defaults to the `threshold_pct` quantile of actuals."""
    df = results.copy()
    df["segment"] = df["target_year"].map(segment_of)
    has_hits = {"dir_hit", "dir_hit3", "hit_tol10"}.issubset(df.columns)
    thr_map = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    for tgt, g in df.groupby("target"):
        thr_map.setdefault(tgt, float(g.drop_duplicates(["origin_yw", "h"])["y"].quantile(threshold_pct)))
    df["ape"] = np.where(df["y"] != 0, 100 * df["abs_err"] / df["y"].abs(), np.nan)  # percent
    df["thr"] = df["target"].map(thr_map)
    df["event"] = df["y"] >= df["thr"]; df["alarm"] = df["median"] >= df["thr"]
    df["tp"] = df.event & df.alarm; df["fp"] = ~df.event & df.alarm; df["fn"] = df.event & ~df.alarm; df["tn"] = ~df.event & ~df.alarm
    parts = []
    for seg_name, seg in [("all", df)] + [(s, df[df.segment == s]) for s in SEGMENTS]:
        if seg.empty:
            continue
        g = seg.groupby(["config", "target", "h"])
        out = pd.DataFrame({
            "wis": g["wis"].mean(),
            "mae": g["abs_err"].mean(),
            "mape": g["ape"].mean(),
            "wape": g["abs_err"].sum() / g["y"].apply(lambda s: s.abs().sum()),
            "mase": g.apply(lambda d: float((d["abs_err"] / d["mase_scale"]).mean()), include_groups=False),
            "cov80": g["cov80"].mean(),
            "cov60": g["cov60"].mean(),
            **({"dir_hit": g["dir_hit"].mean(), "dir_hit3": g["dir_hit3"].mean(), "hit_tol10": g["hit_tol10"].mean()} if has_hits else {}),
            "thr": g["thr"].first(),
            "thr_hit_rate": g["tp"].sum() / (g["tp"].sum() + g["fn"].sum()).replace(0, np.nan),
            "thr_false_alarm": g["fp"].sum() / (g["fp"].sum() + g["tn"].sum()).replace(0, np.nan),
            "thr_accuracy": (g["tp"].sum() + g["tn"].sum()) / g.size(),
            "n": g.size(),
        }).reset_index()
        out.insert(2, "segment", seg_name)
        parts.append(out)
    summ = pd.concat(parts, ignore_index=True)
    ref = summ[summ.config == "snaive"].set_index(["target", "segment", "h"])["wis"]
    summ["rel_wis_vs_snaive"] = [
        r.wis / ref[(r.target, r.segment, r.h)] if (r.target, r.segment, r.h) in ref.index else np.nan
        for r in summ.itertuples()
    ]
    return summ


def save_results(results: pd.DataFrame, summary: pd.DataFrame, tag: str, out_dir: Path | None = None) -> Path:
    out = (out_dir or OUTPUT_DIR / "backtest")
    out.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out / f"{tag}_forecasts.parquet")
    summary.to_csv(out / f"{tag}_summary.csv", index=False)
    return out
