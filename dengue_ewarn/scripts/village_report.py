#!/usr/bin/env python
"""Metrics for the village-level hotspot backtest (PLAN_erakde §11).

  python scripts/village_report.py --tag village --modes asof_adj
Per mode × region × own-state stratum (0 / 1 / ≥2 reported cases in the village over the last 14 days at the origin):
  AUC of P(≥2 new cases in 14 d) pooled over origins; event rate; and per-origin-day recall@k / precision@k (k villages
  alerted per day within the region), averaged over origin days with ≥1 event. Secondary event ≥3.
Writes outputs/backtest/{tag}_village_metrics.csv and prints markdown tables.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import OUTPUT_DIR  # noqa: E402
from scipy.stats import rankdata  # noqa: E402


def auc(ev, s):
    """Mann–Whitney AUC with average ranks for ties (many villages share a score of exactly 0)."""
    ev = np.asarray(ev, bool); s = np.asarray(s, float); n1 = ev.sum(); n0 = len(ev) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s); return float((r[ev].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="village")
ap.add_argument("--modes", nargs="+", default=["asof_adj"])
ap.add_argument("--ks", nargs="+", type=int, default=[10, 30, 100])
ap.add_argument("--years", nargs="*", type=int, default=None, help="restrict (e.g. epidemic years only)")
args = ap.parse_args()
root = Path(__file__).resolve().parents[1]; bt = OUTPUT_DIR / "backtest"
vil = pd.read_parquet(root / "data_processed" / "villages.parquet"); vil = vil[vil.village != ""].reset_index(drop=True)
region = np.where(vil.county.isin(["台南市", "高雄市"]), "台南高雄", "其他縣市")

rows = []
for mode in args.modes:
    res = pd.read_parquet(bt / f"{args.tag}_{mode}_village.parquet")
    if args.years:
        res = res[res.year.isin(args.years)]
    models = [c for c in res.columns if c not in ("origin", "v", "y", "own_raw", "year")]
    res["region"] = region[res.v.to_numpy()]
    res["stratum"] = pd.cut(res.own_raw, [-1, 0.5, 1.5, np.inf], labels=["0 例", "1 例", "≥2 例"]).astype(str)
    for k_ev in (2, 3):
        ev_all = res.y >= k_ev
        for reg, g in res.groupby("region"):
            ev = (g.y >= k_ev).to_numpy()
            # top-k per origin day within the region (all strata together: that is what the daily list looks like)
            for mdl in models + ["own_raw"]:
                rec = {k: [] for k in args.ks}; prec = {k: [] for k in args.ks}
                for _, go in g.groupby("origin"):
                    e = (go.y >= k_ev).to_numpy()
                    if e.sum() == 0:
                        continue
                    order = np.argsort(-go[mdl].to_numpy(), kind="stable")
                    for k in args.ks:
                        hits = e[order[:k]].sum(); rec[k].append(hits / e.sum()); prec[k].append(hits / k)
                row = {"mode": mode, "event": f"≥{k_ev}", "region": reg, "stratum": "全部", "model": mdl, "village-origins": len(g), "events": int(ev.sum()),
                       "event_rate": ev.mean(), "AUC": auc(ev, g[mdl].to_numpy()) if ev.sum() else np.nan, "days_with_events": len(rec[args.ks[0]])}
                for k in args.ks:
                    row[f"recall@{k}"] = float(np.mean(rec[k])) if rec[k] else np.nan; row[f"precision@{k}"] = float(np.mean(prec[k])) if prec[k] else np.nan
                rows.append(row)
            for st, gs in g.groupby("stratum"):
                evs = (gs.y >= k_ev).to_numpy()
                for mdl in models + ["own_raw"]:
                    rows.append({"mode": mode, "event": f"≥{k_ev}", "region": reg, "stratum": st, "model": mdl, "village-origins": len(gs), "events": int(evs.sum()),
                                 "event_rate": evs.mean(), "AUC": auc(evs, gs[mdl].to_numpy()) if 0 < evs.sum() < len(evs) else np.nan})
met = pd.DataFrame(rows); met.to_csv(bt / f"{args.tag}_village_metrics.csv", index=False)
for mode in args.modes:
    for k_ev in (2, 3):
        sub = met[(met["mode"] == mode) & (met.event == f"≥{k_ev}")]
        print(f"\n## {mode} — 事件：未來 14 天新增 ≥{k_ev} 例\n")
        t = sub[sub.stratum == "全部"].pivot_table(index="model", columns="region", values=["AUC"] + [f"recall@{k}" for k in args.ks] + [f"precision@{k}" for k in [args.ks[1]]])
        print("全部村里（每日前 k 名）：事件數 " + ", ".join(f"{r}: {int(sub[(sub.stratum=='全部')&(sub.region==r)].events.iloc[0]):,} / {int(sub[(sub.stratum=='全部')&(sub.region==r)]['village-origins'].iloc[0]):,} 村里起點" for r in sorted(sub.region.unique())))
        print(t.round(3).to_string())
        print("\n分層 AUC（起點當日村里內近 14 天已通報例數）：")
        print(sub[sub.stratum != "全部"].pivot_table(index="model", columns=["region", "stratum"], values="AUC").round(3).to_string())
        print("分層事件數：", sub[sub.stratum != "全部"].drop_duplicates(["region", "stratum"]).set_index(["region", "stratum"]).events.to_dict())
print("\nsaved", bt / f"{args.tag}_village_metrics.csv")
