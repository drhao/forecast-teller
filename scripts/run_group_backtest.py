#!/usr/bin/env python
"""Rolling-origin backtest at county or age-group level: univariate per series vs joint multivariate.

  python scripts/run_group_backtest.py --by county --indicator nhi_out_ili
  python scripts/run_group_backtest.py --by age --indicator nhi_out_ili
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller import OUTPUT_DIR  # noqa: E402
from forecast_teller.backtest import SEGMENTS, segment_of  # noqa: E402
from forecast_teller.baselines import moving_average, naive_last  # noqa: E402
from forecast_teller.backtest import add_hit_columns  # noqa: E402
from forecast_teller.covariates import future_covariates  # noqa: E402
from forecast_teller.metrics import Q_LEVELS, abs_err, covered, directional_hit, directional_hit3, mase_scale, tolerance_hit, wis  # noqa: E402
from forecast_teller.panel import load_long  # noqa: E402
from forecast_teller.weeks import yw_range  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--by", choices=["county", "age"], default="county")
ap.add_argument("--indicator", default="nhi_out_ili")
ap.add_argument("--start", default="201601"); ap.add_argument("--end", default="202553")
ap.add_argument("--warmup", type=int, default=104); ap.add_argument("--horizon", type=int, default=4)
ap.add_argument("--configs", nargs="*", default=["naive", "uni_nocov", "uni_cov", "mv_nocov", "mv_cov"])
ap.add_argument("--dry-run", action="store_true", help="load data and baselines only")
ap.add_argument("--merge", action="store_true", help="re-score only --configs and keep the other configs from the existing parquet")
ap.add_argument("--window", type=int, default=3, help="MA window for the ma baseline")
args = ap.parse_args()

long = load_long(args.by)
yws = yw_range(args.start, args.end)
wide = (long[long.indicator == args.indicator].pivot(index="yw", columns=args.by, values="value").reindex(yws))
wide = wide.dropna(axis=1, how="all")
assert not wide.isna().any().any(), f"missing values: {wide.isna().sum()[wide.isna().sum() > 0].to_dict()}"
groups = list(wide.columns)
Y = wide.to_numpy(np.float64).T  # (G, N)
G, N, H = Y.shape[0], Y.shape[1], args.horizon
origins = list(range(args.warmup, N - H + 1))
print(f"{args.by}={G} series, {N} weeks {yws[0]}..{yws[-1]}, {len(origins)} origins, h={H}")

pf_full = future_covariates(yws, H, kinds=("season", "cny"))


def score_rows(cfg, preds_by_origin):
    """preds_by_origin: list over origins of (med (G,H), q (G,H,9))."""
    rows = []
    for t, (med, q) in zip(origins, preds_by_origin):
        for g_i, g in enumerate(groups):
            truth = Y[g_i, t:t + H]
            w = wis(truth, q[g_i]); c80 = covered(truth, q[g_i], 80); ae = abs_err(truth, med[g_i])
            scale = mase_scale(Y[g_i, :t])
            y0 = Y[g_i, t - 1]
            dh, dh3, th = directional_hit(y0, truth, med[g_i]), directional_hit3(y0, truth, med[g_i]), tolerance_hit(truth, med[g_i])
            for h in range(H):
                rows.append({"config": cfg, args.by: g, "origin_yw": yws[t - 1], "target_yw": yws[t + h], "h": h + 1,
                             "target_year": int(yws[t + h][:4]), "y": truth[h], "median": med[g_i, h],
                             **{f"q{int(l*100)}": q[g_i, h, i] for i, l in enumerate(Q_LEVELS)},
                             "abs_err": ae[h], "wis": w[h], "cov80": bool(c80[h]), "mase_scale": scale, "y_origin": y0,
                             "dir_hit": bool(dh[h]), "dir_hit3": bool(dh3[h]), "hit_tol10": bool(th[h])})
    return pd.DataFrame(rows)


results = []
t0 = time.time()
if "naive" in args.configs:
    preds = []
    for t in origins:
        ms, qs = zip(*[naive_last(Y[g_i, :t], H) for g_i in range(G)])
        preds.append((np.stack(ms), np.stack(qs)))
    results.append(score_rows("naive", preds)); print(f"[naive] {time.time()-t0:.1f}s", flush=True)
if "ma3" in args.configs:
    preds = []
    for t in origins:
        ms, qs = zip(*[moving_average(Y[g_i, :t], H, window=args.window) for g_i in range(G)])
        preds.append((np.stack(ms), np.stack(qs)))
    results.append(score_rows("ma3", preds)); print(f"[ma3] {time.time()-t0:.1f}s", flush=True)

if not args.dry_run and any(c not in ("naive", "ma3") for c in args.configs):
    from forecast_teller.model_timesfm3 import TimesFM3Model
    model = TimesFM3Model(backend="mlx", batch_size=16)
    print(f"model loaded {model.load_seconds:.1f}s", flush=True)
    for cfg in [c for c in args.configs if c not in ("naive", "ma3")]:
        t1 = time.time()
        mode, cov = cfg.split("_")
        use_cov = cov == "cov"
        if mode == "uni":
            ctxs = [Y[g_i, :t] for t in origins for g_i in range(G)]
            pfs = [pf_full[:, :t + H] for t in origins for _ in range(G)] if use_cov else None
            outs = model.forecast(ctxs, H, past_future=pfs)
            preds = []
            for k, t in enumerate(origins):
                chunk = outs[k * G:(k + 1) * G]
                preds.append((np.stack([m for m, _ in chunk]), np.stack([q for _, q in chunk])))
        else:  # joint multivariate: (G, L) per origin
            ctxs = [Y[:, :t] for t in origins]
            pfs = [pf_full[:, :t + H] for t in origins] if use_cov else None
            preds = model.forecast(ctxs, H, past_future=pfs)
        results.append(score_rows(cfg, preds)); print(f"[{cfg}] {time.time()-t1:.1f}s", flush=True)

res = pd.concat(results, ignore_index=True)
res = add_hit_columns(res)
out = OUTPUT_DIR / "backtest"; out.mkdir(parents=True, exist_ok=True)
tag = f"{args.by}_{args.indicator}"
if args.merge and (out / f"{tag}_forecasts.parquet").exists():
    old = pd.read_parquet(out / f"{tag}_forecasts.parquet")
    old = old[~old.config.isin(set(res.config))]
    res = pd.concat([old, res], ignore_index=True)
    print(f"merged with existing file: configs now {sorted(res.config.unique())}", flush=True)
res["segment"] = res.target_year.map(segment_of)
res.to_parquet(out / f"{tag}_forecasts.parquet")

# summary: per config x segment x h aggregated over groups (sum of WIS = national-scale comparable), plus rel to naive
agg = res.groupby(["config", "segment", "h"]).agg(wis_sum=("wis", "sum"), wis=("wis", "mean"), mae=("abs_err", "mean"),
                                                   cov80=("cov80", "mean"), n=("wis", "size")).reset_index()
ref = agg[agg.config == "naive"].set_index(["segment", "h"])["wis_sum"]
agg["rel_wis_vs_naive"] = [r.wis_sum / ref[(r.segment, r.h)] for r in agg.itertuples()]
agg.to_csv(out / f"{tag}_summary.csv", index=False)
# per-group post-covid relative WIS (h=1..4 mean) for each config
post = res[res.segment == "post_covid"]
per_group = post.groupby(["config", args.by])["wis"].mean().unstack("config")
per_group_rel = per_group.div(per_group["naive"], axis=0)
lines = [f"# {args.by} 層級回測：{args.indicator}", "",
         f"- {G} 條序列，資料窗 {yws[0]}–{yws[-1]}，{len(origins)} 個起點，h = 1–{H}；rel = 相對 per-series last-value naive 的 WIS", "",
         "## 各分段 h = 1–4 平均（加總所有序列的 WIS，rel 為相對 naive）", ""]
for seg in ["post_covid", "pre_covid", "covid", "all"]:
    sub = agg[agg.segment == seg] if seg != "all" else agg.groupby(["config", "h"]).agg(wis_sum=("wis_sum", "sum"), cov80=("cov80", "mean")).reset_index().assign(segment="all")
    if seg == "all":
        ref_all = sub[sub.config == "naive"].set_index("h")["wis_sum"]
        sub["rel_wis_vs_naive"] = [r.wis_sum / ref_all[r.h] for r in sub.itertuples()]
    tab = sub.groupby("config").agg(wis_sum=("wis_sum", "mean"), rel_wis=("rel_wis_vs_naive", "mean"), cov80=("cov80", "mean")).sort_values("wis_sum")
    lines += [f"### {seg}", "", tab.round(3).to_markdown(), ""]
lines += ["## COVID 後：各序列相對 naive 的 WIS（h = 1–4 平均；< 1 表示優於 naive）", "", per_group_rel.round(2).to_markdown(), ""]
hits = post.groupby(["config", "h"])[["dir_hit", "dir_hit3", "hit_tol10", "cov80"]].mean().rename(columns={"cov80": "hit80"})
lines += ["## COVID 後命中率（所有序列合併；dir_hit = 方向命中率，dir_hit3 = 漲/持平/跌三分類，hit_tol10 = ±10% 容忍帶，hit80 = 80% 區間）", "",
          hits.groupby("config").mean().round(3).to_markdown(), "", "方向命中率依 horizon：", "",
          hits["dir_hit"].unstack("h").round(3).to_markdown(), ""]
(out / f"{tag}_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:40]))
print(f"saved {out / tag}_* in {time.time()-t0:.0f}s")
