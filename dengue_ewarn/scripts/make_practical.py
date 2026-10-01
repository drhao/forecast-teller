#!/usr/bin/env python
"""make_practical.py — 白話解讀用的實務數字與真實案例（asof_adj 模式，TimesFM）。

輸出（outputs/backtest/）：
  {tag}_practical.csv       每個流行季（6–12 月）的警示週數、其中真 / 假、事件週數、漏掉的事件週，依 p* × 區域 × 事件型態
  {tag}_rule_vs_model.csv   「最近 7 天已通報 ≥ k 例」的簡單規則 vs 模型，在下限群聚層的敏感度 / 假警報 / PPV 與事件前置時間
  {tag}_examples.json       偵測到、漏掉、假警報的真實案例，含每日病例時間線（實際發病數、警示當天已知數、完整度校正後）

用法：python scripts/make_practical.py --tag dengue_all --panel dengue_all
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import OUTPUT_DIR  # noqa: E402
from dengue_ewarn.alerts import alert_metrics, auc, episodes  # noqa: E402
from dengue_ewarn.data import completeness, epidemic_years, ewarn_threshold, load_panel, region_of, rolling7  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="dengue_all"); ap.add_argument("--panel", default=None); ap.add_argument("--mode", default="asof_adj")
ap.add_argument("--floor", type=float, default=3.0); ap.add_argument("--window", type=int, default=14)
args = ap.parse_args()
bt = OUTPUT_DIR / "backtest"

counts, hist = load_panel(args.panel or args.tag); dates = counts.index; series = list(counts.columns); sidx = {s: i for i, s in enumerate(series)}
C = counts.to_numpy().T.astype(float); S7 = rolling7(C); S, T = C.shape; K = hist.shape[2]
CUM = np.cumsum(hist, axis=2, dtype=np.int32)
thr_final = np.stack([ewarn_threshold(C, t, floor=args.floor) for t in range(T)], axis=1)
big = epidemic_years(counts)
res = pd.read_parquet(bt / f"{args.tag}_{args.mode}_forecasts.parquet", columns=["model", "h", "series", "origin", "year", "prob", "event", "thr", "thr_final", "y"])
a = res[(res.h == 0) & (res.model == "tfm")].copy(); a["origin"] = pd.to_datetime(a.origin)
a["region"] = a.series.map(region_of); a["etype"] = np.where(a.thr_final <= args.floor, "下限群聚", "流行中加速"); a["week"] = a.origin.dt.to_period("W")
years = sorted(a.year.unique()); origins_d = sorted({dates.get_loc(o) for o in a.origin.unique()})
ETYPE = lambda v: "下限群聚" if v <= args.floor else "流行中加速"


def known_daily(d: int, adj: bool) -> np.ndarray:
    """Daily counts by onset day known at day d, (S, d+1); adj = divided by historical completeness (as in the backtest)."""
    k = np.minimum(d - np.arange(d + 1), K - 1); kn = CUM[:, np.arange(d + 1), k].astype(float)
    if adj:
        c = np.maximum(completeness(hist, d), 0.2); f = 1.0 / c[k[-K:]] if d + 1 >= K else 1.0 / c[k]
        kn[:, -len(f):] *= f
    return kn


# ---- 1) simple rule "reported >= k cases with onset in the last 7 days" at each origin (what the officer sees that day)
raw7 = {d: known_daily(d, False)[:, -7:].sum(axis=1) for d in origins_d}
adj7 = {d: known_daily(d, True)[:, -7:].sum(axis=1) for d in origins_d}
a["raw7"] = [raw7[dates.get_loc(o)][sidx[s]] for s, o in zip(a.series, a.origin)]
a["adj7"] = [adj7[dates.get_loc(o)][sidx[s]] for s, o in zip(a.series, a.origin)]
a["rule1"] = a.raw7 >= 1; a["rule2"] = a.raw7 >= 2

# ---- 2) per-season practical counts (weekly aggregation, like the report)
wk = a.groupby(["series", "week"]).agg(prob=("prob", "max"), event=("event", "max"), thr_max=("thr_final", "max"), rule1=("rule1", "max"), rule2=("rule2", "max"),
                                       raw7=("raw7", "max"), adj7=("adj7", "max"), region=("region", "first"), year=("year", "first")).reset_index()
wk["etype"] = wk.thr_max.map(ETYPE)
prac = []
for p in (0.3, 0.5):
    for (region, etype), g in wk.groupby(["region", "etype"]):
        for yr, gy in g.groupby("year"):
            al = gy.prob >= p; ev = gy.event.astype(bool)
            prac.append({"p_star": p, "區域": region, "事件型態": etype, "year": int(yr), "鄉鎮數": int(gy.series.nunique()), "鄉鎮週": len(gy), "警示週": int(al.sum()),
                         "真警示週": int((al & ev).sum()), "假警示週": int((al & ~ev).sum()), "事件週": int(ev.sum()), "漏掉事件週": int((~al & ev).sum())})
prac = pd.DataFrame(prac); prac.to_csv(bt / f"{args.tag}_practical.csv", index=False)
season = prac.groupby(["p_star", "區域", "事件型態"]).agg(季數=("year", "size"), 鄉鎮數=("鄉鎮數", "max"), 每季警示週=("警示週", "mean"), 每季真警示=("真警示週", "mean"),
                                                      每季假警示=("假警示週", "mean"), 每季事件週=("事件週", "mean"), 每季漏掉=("漏掉事件週", "mean")).reset_index()

# ---- 3) rule vs model in the 下限群聚 stratum (weekly metrics + episode lead)
ep = [(s_i, t) for s_i in range(S) for t in episodes(S7[s_i], thr_final[s_i]) if dates[t].year in years and 6 <= dates[t].month <= 12]
ep_df = pd.DataFrame([{"s_i": s_i, "t": t, "series": series[s_i], "region": region_of(series[s_i]), "year": dates[t].year, "etype": ETYPE(thr_final[s_i, t])} for s_i, t in ep])
flag_lookup = {name: dict(zip(zip(a.series, a.origin), a[col])) for name, col in [("rule1", "rule1"), ("rule2", "rule2")]}
prob_lookup = dict(zip(zip(a.series, a.origin), a.prob))


def leads(flag_fn):
    out = []
    for s_i, t in ep:
        ks = [k for k in range(1, args.window + 1) if flag_fn(series[s_i], dates[t - k])]
        out.append(max(ks) if ks else np.nan)
    return np.array(out, float)


methods = {"規則：最近 7 天已通報 ≥ 1 例": lambda s, o: flag_lookup["rule1"].get((s, o), False), "規則：最近 7 天已通報 ≥ 2 例": lambda s, o: flag_lookup["rule2"].get((s, o), False),
           "TimesFM asof_adj p*=0.3": lambda s, o: prob_lookup.get((s, o), -1) >= 0.3, "TimesFM asof_adj p*=0.5": lambda s, o: prob_lookup.get((s, o), -1) >= 0.5}
lead_by = {name: leads(fn) for name, fn in methods.items()}
rv = []
for region in sorted(wk.region.unique()):
    g = wk[(wk.region == region) & (wk.etype == "下限群聚")]; ev = g.event.to_numpy(bool)
    idx = ep_df[(ep_df.region == region) & (ep_df.etype == "下限群聚")].index.to_numpy()
    for name in methods:
        flag = (g.rule1 if "≥ 1" in name else g.rule2 if "≥ 2" in name else g.prob >= (0.3 if "0.3" in name else 0.5)).to_numpy(bool)
        tp, fp, fn, tn = int((flag & ev).sum()), int((flag & ~ev).sum()), int((~flag & ev).sum()), int((~flag & ~ev).sum())
        lv = lead_by[name][idx]; det = ~np.isnan(lv)
        rv.append({"區域": region, "方法": name, "鄉鎮週": len(g), "事件週": int(ev.sum()), "敏感度": tp / max(tp + fn, 1), "假警報/100": 100 * fp / max(fp + tn, 1), "PPV": tp / max(tp + fp, 1),
                   "事件數": len(idx), "偵測到的事件比例": float(det.mean()), "前置中位數（天）": float(np.nanmedian(lv)) if det.any() else np.nan})
rule_vs_model = pd.DataFrame(rv); rule_vs_model.to_csv(bt / f"{args.tag}_rule_vs_model.csv", index=False)

# ---- 3b) same false-alarm axis: count-rule family (已通報 ≥ k) vs model p* grid, 下限群聚 weekly; AUC of the reported count as a score
fam, auc_quiet = [], {}
for region in sorted(wk.region.unique()):
    g = wk[(wk.region == region) & (wk.etype == "下限群聚")]; ev = g.event.to_numpy(bool); nseas = max(g.year.nunique(), 1)
    auc_quiet[region] = {"model": float(auc(ev, g.prob.to_numpy())), "raw7": float(auc(ev, g.raw7.to_numpy())), "adj7": float(auc(ev, g.adj7.to_numpy())), "event_weeks": int(ev.sum()), "weeks": len(g), "townships": int(g.series.nunique())}
    for name, flag in [(f"規則：最近 7 天已通報 ≥ {k} 例", (g.raw7 >= k).to_numpy(bool)) for k in (1, 2, 3)] + [(f"TimesFM p* = {p}", (g.prob >= p).to_numpy(bool)) for p in (0.1, 0.2, 0.3, 0.5, 0.7)]:
        tp, fp, fn, tn = int((flag & ev).sum()), int((flag & ~ev).sum()), int((~flag & ev).sum()), int((~flag & ~ev).sum())
        fam.append({"區域": region, "方法": name, "敏感度": tp / max(tp + fn, 1), "假警報/100": 100 * fp / max(fp + tn, 1), "PPV": tp / max(tp + fp, 1), "每季真警示": tp / nseas, "每季假警報": fp / nseas})
rule_family = pd.DataFrame(fam); rule_family.to_csv(bt / f"{args.tag}_rule_family.csv", index=False)

# ---- 4) examples with timelines
def timeline(s, d0, pre, post, mark_days=()):
    s_i = sidx[s]; kr = known_daily(d0, False)[s_i]; ka = known_daily(d0, True)[s_i]
    lo, hi = max(0, d0 - pre), min(T - 1, d0 + post)
    return {"dates": [dates[d].strftime("%m-%d") for d in range(lo, hi + 1)], "final": [int(C[s_i, d]) for d in range(lo, hi + 1)],
            "known": [int(kr[d]) if d <= d0 else None for d in range(lo, hi + 1)], "adjusted": [round(float(ka[d]), 1) if d <= d0 else None for d in range(lo, hi + 1)],
            "origin": dates[d0].strftime("%Y-%m-%d"), "marks": [dates[d].strftime("%m-%d") for d in mark_days if lo <= d <= hi]}


model_lead = lead_by["TimesFM asof_adj p*=0.3"]
ep_df["lead3"] = model_lead; ep_df["thr"] = [thr_final[s_i, t] for s_i, t in ep]; ep_df["s7_max14"] = [S7[s_i, t:t + 15].max() for s_i, t in ep]
ep_df["cases_next14"] = [C[s_i, t + 1:t + 15].sum() for s_i, t in ep]; ep_df["cases_prev14"] = [C[s_i, max(0, t - 14):t].sum() for s_i, t in ep]
ep_df["cross"] = [dates[t].strftime("%Y-%m-%d") for _, t in ep]
examples = {"hits": [], "misses": [], "false_alarms": [], "epidemic_hits": []}
def alert_origin(s_i, t, p=0.3):
    ks = [k for k in range(1, args.window + 1) if prob_lookup.get((series[s_i], dates[t - k]), -1) >= p]
    return (t - max(ks)) if ks else None
oth_hits = ep_df[(ep_df.region == "其他縣市") & (ep_df.etype == "下限群聚") & (ep_df.lead3 >= 3)].sort_values("s7_max14", ascending=False)
seen = set()
for _, r in oth_hits.iterrows():
    county = r.series.split("|")[0]
    if county in seen or len(examples["hits"]) >= 4:
        continue
    seen.add(county); d0 = alert_origin(r.s_i, r.t)
    examples["hits"].append({"series": r.series, "cross": r.cross, "thr": float(r.thr), "s7_max14": float(r.s7_max14), "cases_next14": float(r.cases_next14), "lead": int(r.lead3),
                             "alert_date": dates[d0].strftime("%Y-%m-%d"), "prob": round(float(prob_lookup[(r.series, dates[d0])]), 2), "known7": int(raw7[d0][r.s_i]) if d0 in raw7 else None,
                             "timeline": timeline(r.series, d0, 10, 18, mark_days=(d0, r.t))})
oth_miss = ep_df[(ep_df.region == "其他縣市") & (ep_df.etype == "下限群聚") & ep_df.lead3.isna()].sort_values("s7_max14", ascending=False)
for _, r in oth_miss.head(3).iterrows():
    d0 = max(o for o in origins_d if o < r.t)   # last origin before the crossing
    examples["misses"].append({"series": r.series, "cross": r.cross, "thr": float(r.thr), "s7_max14": float(r.s7_max14), "cases_next14": float(r.cases_next14), "cases_prev14": float(r.cases_prev14),
                               "last_origin": dates[d0].strftime("%Y-%m-%d"), "prob": round(float(prob_lookup.get((r.series, dates[d0]), np.nan)), 2), "known14": int(known_daily(d0, False)[r.s_i, -14:].sum()),
                               "timeline": timeline(r.series, d0, 10, 12, mark_days=(d0, r.t))})
fa = a[(a.prob >= 0.5) & (~a.event) & (a.region == "其他縣市")].sort_values("prob", ascending=False).drop_duplicates("series")
for _, r in pd.concat([fa[fa.raw7 == 1].head(2), fa[fa.raw7 >= 3].head(1)]).iterrows():
    d0 = dates.get_loc(r.origin); s_i = sidx[r.series]
    examples["false_alarms"].append({"series": r.series, "origin": r.origin.strftime("%Y-%m-%d"), "prob": round(float(r.prob), 2), "known7": int(r.raw7), "adj7": round(float(known_daily(d0, True)[s_i, -7:].sum()), 1),
                                     "final7": float(S7[s_i, d0]), "max_s7_next14": float(r.y), "cases_next14": float(C[s_i, d0 + 1:d0 + 15].sum()), "timeline": timeline(r.series, d0, 10, 14, mark_days=(d0,))})
tk_hits = ep_df[(ep_df.region == "台南高雄") & (ep_df.etype == "流行中加速") & (ep_df.lead3 >= 5)].sort_values("thr", ascending=False)
for _, r in tk_hits.head(2).iterrows():
    d0 = alert_origin(r.s_i, r.t)
    examples["epidemic_hits"].append({"series": r.series, "cross": r.cross, "thr": float(r.thr), "s7_cross": float(S7[r.s_i, r.t]), "s7_max14": float(r.s7_max14), "lead": int(r.lead3),
                                      "alert_date": dates[d0].strftime("%Y-%m-%d"), "prob": round(float(prob_lookup[(r.series, dates[d0])]), 2), "known7": int(raw7[d0][r.s_i]), "final7_at_alert": float(S7[r.s_i, d0]),
                                      "timeline": timeline(r.series, d0, 10, 18, mark_days=(d0, r.t))})
# share of first-cluster episodes (other counties) that were "batch" (0 reported cases with onset in the last 14 days at the last origin before crossing)
batch = []
for _, r in ep_df[(ep_df.region == "其他縣市") & (ep_df.etype == "下限群聚")].iterrows():
    d0 = max(o for o in origins_d if o < r.t); batch.append(int(known_daily(d0, False)[r.s_i, -14:].sum()) == 0)
summary = {"n_first_cluster_other": int(len(batch)), "share_batch_other": float(np.mean(batch)), "share_small_other": float((ep_df[(ep_df.region == "其他縣市") & (ep_df.etype == "下限群聚")].s7_max14 <= 5).mean()),
           "season_table": season.round(2).to_dict(orient="records"), "rule_vs_model": rule_vs_model.round(3).to_dict(orient="records"),
           "rule_family": rule_family.round(3).to_dict(orient="records"), "auc_quiet": auc_quiet}
(bt / f"{args.tag}_examples.json").write_text(json.dumps({"examples": examples, "summary": summary}, ensure_ascii=False, indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o)), encoding="utf-8")

pd.set_option("display.width", 220)
print("每季平均（p* × 區域 × 事件型態）：\n", season.round(1).to_string(index=False))
print("\n下限群聚層：簡單規則 vs 模型\n", rule_vs_model.round(3).to_string(index=False))
print("\n下限群聚層（鄉鎮週）：同一假警報軸上的規則家族 vs 模型；AUC", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in auc_quiet.items()}, "\n", rule_family.round(3).to_string(index=False))
print(f"\n其他縣市的第一個群聚 {summary['n_first_cluster_other']} 個：突破前最後一個起點已通報 0 例（整批出現）的比例 {summary['share_batch_other']:.2f}；兩週內最高 7 日累計 ≤ 5 例的比例 {summary['share_small_other']:.2f}")
for k, v in examples.items():
    print(f"\n[{k}]")
    for e in v:
        print(" ", {kk: vv for kk, vv in e.items() if kk != "timeline"})
