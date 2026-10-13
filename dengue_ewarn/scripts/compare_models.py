#!/usr/bin/env python
"""Equal-false-alarm comparison of forecast models on the same events (PLAN_erakde.md §5; same form as PLAN §14.1).

For each mode × region × event type stratum: at a false-alarm budget F (per 100 township-weeks) find, per model, the
largest p* whose false-alarm rate <= F and report sensitivity (township-week) and PPV; plus the episode-level detection
share and median lead at that p*. Also WIS (h = 7) and AUC per stratum.

  python scripts/compare_models.py --tag dengue_kde --panel dengue
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import OUTPUT_DIR  # noqa: E402
from dengue_ewarn.alerts import alert_metrics, auc, episodes  # noqa: E402
from dengue_ewarn.data import ewarn_threshold, forecast_files, load_panel, region_of, rolling7  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="dengue_kde")
ap.add_argument("--panel", default=None)
ap.add_argument("--floor", type=float, default=3.0)
ap.add_argument("--budgets", nargs="+", type=float, default=[0.5, 1, 2, 5, 10, 20])
ap.add_argument("--lead-window", type=int, default=14)
args = ap.parse_args()

bt = OUTPUT_DIR / "backtest"
res = pd.concat([pd.read_parquet(f) for f in forecast_files(bt, args.tag)], ignore_index=True).drop_duplicates(subset=["model", "mode", "series", "origin", "h"])
res["origin"] = pd.to_datetime(res["origin"]); res["region"] = res.series.map(region_of)
tf_col = "thr_final" if "thr_final" in res.columns else "thr"
ETYPE = lambda thr: np.where(np.asarray(thr) <= args.floor, "下限群聚", "流行中加速")
counts, _ = load_panel(args.panel or args.tag); dates = counts.index; series = list(counts.columns)
C = counts.to_numpy().T.astype(float); S7 = rolling7(C)
thr_final = np.stack([ewarn_threshold(C, t, floor=args.floor) for t in range(C.shape[1])], axis=1)
years = sorted(res.year.unique()); modes = sorted(res["mode"].unique()); models = sorted(res.model.unique())
P_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

# --- township-week table per mode × model
anyr = res[res.h == 0].copy(); anyr["week"] = anyr.origin.dt.to_period("W")
wk = anyr.groupby(["mode", "model", "series", "week"]).agg(prob=("prob", "max"), event=("event", "max"), thr_max=(tf_col, "max"), region=("region", "first")).reset_index()
wk["etype"] = ETYPE(wk.thr_max)

# --- episodes and per-episode lead per (mode, model, p*)
ep = [(s_i, t) for s_i in range(C.shape[0]) for t in episodes(S7[s_i], thr_final[s_i]) if dates[t].year in years and 6 <= dates[t].month <= 12]
ep_df = pd.DataFrame([{"series": series[s_i], "region": region_of(series[s_i]), "etype": str(ETYPE(thr_final[s_i, t])), "t": t} for s_i, t in ep])
prob_map = {k: dict(zip(zip(g.series, g.origin), g.prob)) for k, g in anyr.groupby(["mode", "model"])}

def leads(mode, model, p_star):
    key = prob_map[(mode, model)]; out = []
    for s_i, t in ep:
        cand = [k for k in range(1, args.lead_window + 1) if key.get((series[s_i], dates[t - k]), -1) >= p_star]
        out.append(max(cand) if cand else np.nan)
    return np.array(out, dtype=float)

rows = []
for (mode, model), g_all in wk.groupby(["mode", "model"]):
    for (region, etype), g in g_all.groupby(["region", "etype"]):
        ev, pr = g.event.to_numpy(bool), g.prob.to_numpy(float)
        if ev.sum() == 0:
            continue
        A = auc(ev, pr)
        sweep = pd.DataFrame([alert_metrics(ev, pr, p) for p in P_GRID])
        eidx = ep_df[(ep_df.region == region) & (ep_df.etype == etype)].index.to_numpy()
        for F in args.budgets:
            ok = sweep[sweep.false_alarm_per100 <= F].sort_values("sensitivity", ascending=False)
            if ok.empty:
                rows.append({"mode": mode, "model": model, "區域": region, "事件型態": etype, "鄉鎮週": len(g), "事件週": int(ev.sum()), "AUC": round(A, 3), "假警報預算/100": F,
                             "p*": np.nan, "敏感度": 0.0, "假警報/100": 0.0, "PPV": np.nan, "事件數": len(eidx), "偵測事件比例": 0.0, "前置中位數": np.nan})
                continue
            b = ok.iloc[0]; lv = leads(mode, model, float(b.p_star))[eidx] if len(eidx) else np.array([])
            det = ~np.isnan(lv)
            rows.append({"mode": mode, "model": model, "區域": region, "事件型態": etype, "鄉鎮週": len(g), "事件週": int(ev.sum()), "AUC": round(A, 3), "假警報預算/100": F,
                         "p*": float(b.p_star), "敏感度": round(float(b.sensitivity), 3), "假警報/100": round(float(b.false_alarm_per100), 2), "PPV": round(float(b.ppv), 3) if not np.isnan(b.ppv) else np.nan,
                         "事件數": len(eidx), "偵測事件比例": round(float(det.mean()), 2) if len(lv) else np.nan, "前置中位數": float(np.nanmedian(lv)) if det.any() else np.nan})
cmp = pd.DataFrame(rows); cmp.to_csv(bt / f"{args.tag}_model_compare.csv", index=False)

# --- WIS by stratum (h = 7)
w7 = res[res.h == 7].copy(); w7["etype"] = ETYPE(w7[tf_col]); w7["year_type"] = w7.year.astype(str)
wis = w7.groupby(["mode", "model", "region", "etype"]).agg(wis=("wis", "mean"), cov80=("cov80", "mean"), n=("wis", "size")).reset_index().round(3)
wis.to_csv(bt / f"{args.tag}_wis_strata.csv", index=False)

print(f"tag {args.tag}: {len(res):,} rows | modes {modes} | models {models} | years {years} | episodes {len(ep)}")
for mode in modes:
    print(f"\n=== {mode}: WIS h=7 (cov80) by region × event type ===")
    t = wis[wis["mode"] == mode].pivot_table(index="model", columns=["region", "etype"], values="wis")
    print(t.round(2).to_string())
    print(f"\n=== {mode}: AUC by stratum ===")
    print(cmp[cmp["mode"] == mode].drop_duplicates(["model", "區域", "事件型態"]).pivot_table(index="model", columns=["區域", "事件型態"], values="AUC").round(3).to_string())
    for F in args.budgets:
        sub = cmp[(cmp["mode"] == mode) & (cmp["假警報預算/100"] == F)]
        print(f"\n=== {mode}: false-alarm budget {F}/100 township-weeks: sensitivity / detected episodes / median lead ===")
        piv = sub.assign(cell=sub.apply(lambda r: f"{r['敏感度']:.2f} / {r['偵測事件比例']:.2f} / {r['前置中位數']:.0f}d" if not np.isnan(r['前置中位數']) else f"{r['敏感度']:.2f} / {r['偵測事件比例']:.2f} / —", axis=1)).pivot_table(index="model", columns=["區域", "事件型態"], values="cell", aggfunc="first")
        print(piv.to_string())
print("\nsaved", bt / f"{args.tag}_model_compare.csv", "and", bt / f"{args.tag}_wis_strata.csv")
