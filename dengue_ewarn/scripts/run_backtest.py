#!/usr/bin/env python
"""Daily rolling-origin backtest for dengue 7-day township case sums (CHG scenario 1, phase-1 validation).

Modes: final   = context built from final onset-date counts (oracle, no reporting delay)
       asof    = context rebuilt from cases reported by the origin day (honest)
       asof_adj= as-of counts divided by historical reporting completeness (simple nowcast correction)
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from dengue_ewarn.alerts import prob_ge  # noqa: E402
from dengue_ewarn.baselines import naive_last, seasonal_naive  # noqa: E402
from dengue_ewarn.data import completeness, ewarn_threshold, rolling7  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--years", nargs="+", type=int, default=list(range(2013, 2025)))
ap.add_argument("--modes", nargs="+", default=["final", "asof", "asof_adj"])
ap.add_argument("--models", nargs="+", default=["tfm", "naive", "snaive"])
ap.add_argument("--step", type=int, default=1, help="origin step in days within the season")
ap.add_argument("--season", nargs=2, default=["06-01", "12-31"])
ap.add_argument("--context", type=int, default=1095)
ap.add_argument("--horizon", type=int, default=14)
ap.add_argument("--hs", nargs="+", type=int, default=[1, 3, 5, 7, 10, 14])
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--tag", default="dengue")
ap.add_argument("--floor", type=float, default=3.0, help="minimum-case floor of the EWARN threshold")
ap.add_argument("--chunk", type=int, default=1024, help="contexts per TimesFM call (bounds GPU memory)")
args = ap.parse_args()

counts = pd.read_parquet(PROCESSED_DIR / "dengue_township_daily.parquet")
z = np.load(PROCESSED_DIR / "dengue_delay_hist.npz", allow_pickle=True)
hist = z["hist"]; dates = counts.index; series = list(counts.columns)
C = counts.to_numpy().T.astype(np.int64)                      # (S, T) final daily counts
S7 = rolling7(C).astype(np.float64)                            # final 7-day sums
CUM = np.cumsum(hist, axis=2)                                  # (S, T, K) cases reported within k days
S, T, K = hist.shape; H = args.horizon; L = args.context
print(f"{S} series × {T} days ({dates[0].date()}→{dates[-1].date()}), modes={args.modes}, models={args.models}", flush=True)

QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def wis_vec(y, q):  # y (N,), q (N, 9)
    tot = 0.5 * np.abs(y - q[:, 4])
    for lo, hi, a in [(0, 8, .2), (1, 7, .4), (2, 6, .6), (3, 5, .8)]:
        l, u = q[:, lo], q[:, hi]; tot += (a / 2) * ((u - l) + (2 / a) * (l - y) * (y < l) + (2 / a) * (y - u) * (y > u))
    return tot / 4.5

def known_daily(d, mode):
    if mode == "final":
        return C[:, : d + 1].astype(np.float64)
    k = np.minimum(d - np.arange(d + 1), K - 1)
    known = CUM[:, np.arange(d + 1), k].astype(np.float64)      # (S, d+1)
    if mode == "asof_adj":
        c = np.maximum(completeness(hist, d), 0.2)             # (K,) floor 0.2 → at most ×5 inflation of the newest days
        adj = 1.0 / c[k[-K:]] if d + 1 >= K else 1.0 / c[k]
        known[:, -len(adj):] = known[:, -len(adj):] * adj
    return known

model = None; mx = None
if "tfm" in args.models:
    from dengue_ewarn.model import TimesFM3Model
    import mlx.core as mx
    try:  # keep MLX from growing its buffer cache without bound (unified memory → swapping)
        mx.set_memory_limit(6 * 1024 ** 3); mx.set_cache_limit(1 * 1024 ** 3)
    except Exception as e:
        print("mlx memory limit not set:", e)
    model = TimesFM3Model(backend="mlx", batch_size=args.batch); print(f"model loaded {model.load_seconds:.0f}s", flush=True)


def forecast_chunked(ctxs, H):
    out = []
    for i in range(0, len(ctxs), args.chunk):
        out.extend(model.forecast(ctxs[i:i + args.chunk], H))
        try:
            mx.clear_cache()
        except Exception:
            pass
    return out

out_dir = OUTPUT_DIR / "backtest"; out_dir.mkdir(parents=True, exist_ok=True)
ck_dir = out_dir / f"{args.tag}_checkpoints"; ck_dir.mkdir(exist_ok=True)
for mode in args.modes:
    frames = []
    for year in args.years:
        t0 = time.time()
        ck = ck_dir / f"{mode}_{year}.parquet"
        if ck.exists():
            frames.append(pd.read_parquet(ck)); print(f"  [{mode}] {year}: loaded checkpoint", flush=True); continue
        lo = dates.get_indexer([pd.Timestamp(f"{year}-{args.season[0]}")])[0]; hi = dates.get_indexer([pd.Timestamp(f"{year}-{args.season[1]}")])[0]
        if lo < 0 or hi < 0:
            print(f"  skip {year}: season outside data"); continue
        origins = [d for d in range(lo, hi + 1, args.step) if d >= 365 and d + H < T]
        if not origins:
            continue
        ctxs, meta, thrs = [], [], []
        for d in origins:
            daily = known_daily(d, mode)
            s7 = rolling7(daily)[:, max(0, d + 1 - L): d + 1]
            thr = ewarn_threshold(daily, d, floor=args.floor)
            for s_i in range(S):
                ctxs.append(s7[s_i].astype(np.float32)); meta.append((s_i, d)); thrs.append(thr[s_i])
        n = len(ctxs); thrs = np.array(thrs)
        truth = np.stack([S7[s_i, d + 1: d + 1 + H] for s_i, d in meta])   # (n, H)
        preds = {}
        if model is not None:
            preds["tfm"] = forecast_chunked(ctxs, H)
        if "naive" in args.models:
            preds["naive"] = [naive_last(c, H) for c in ctxs]
        if "snaive" in args.models:
            preds["snaive"] = [seasonal_naive(c, H, season=364) if len(c) >= 372 else naive_last(c, H) for c in ctxs]
        for name, pr in preds.items():
            Q = np.stack([q for _, q in pr])                       # (n, H, 9)
            for h in args.hs:
                qh = Q[:, h - 1, :]; y = truth[:, h - 1]
                df = pd.DataFrame({"model": name, "mode": mode, "series": [series[s] for s, _ in meta], "origin": [dates[d] for _, d in meta],
                                   "year": year, "h": h, "y": y, "median": qh[:, 4], "q10": qh[:, 0], "q20": qh[:, 1], "q80": qh[:, 7], "q90": qh[:, 8],
                                   "wis": wis_vec(y, qh), "cov80": (y >= qh[:, 0]) & (y <= qh[:, 8]), "thr": thrs,
                                   "prob": prob_ge(qh, thrs), "event": y >= thrs})
                frames.append(df)
            # 'any crossing within the horizon' summary row (h = 0 marker)
            pany = np.max(prob_ge(np.transpose(Q, (1, 0, 2)), np.broadcast_to(thrs, (H, n))), axis=0)
            eany = (truth >= thrs[:, None]).any(axis=1)
            frames.append(pd.DataFrame({"model": name, "mode": mode, "series": [series[s] for s, _ in meta], "origin": [dates[d] for _, d in meta],
                                        "year": year, "h": 0, "y": truth.max(axis=1), "median": np.nan, "q10": np.nan, "q20": np.nan, "q80": np.nan, "q90": np.nan,
                                        "wis": np.nan, "cov80": False, "thr": thrs, "prob": pany, "event": eany}))
        year_df = pd.concat(frames[-len(preds) * (len(args.hs) + 1):], ignore_index=True)
        year_df.to_parquet(ck)
        h7 = year_df[year_df.h == 7].groupby("model")["wis"].mean()
        print(f"  [{mode}] {year}: {len(origins)} origins × {S} series = {n} contexts in {time.time()-t0:.0f}s | "
              + " ".join(f"{k}: WIS h7 {v:.1f}" for k, v in h7.items()), flush=True)
    res = pd.concat(frames, ignore_index=True)
    res["city"] = res["series"].str.split("|").str[0]
    res.to_parquet(out_dir / f"{args.tag}_{mode}_forecasts.parquet")
    print(f"saved {args.tag}_{mode}_forecasts.parquet ({len(res):,} rows)", flush=True)
