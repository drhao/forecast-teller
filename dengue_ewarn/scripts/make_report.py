#!/usr/bin/env python
"""Alert evaluation for the dengue backtest: WIS/coverage, threshold-crossing probability skill (AUC),
p* sweep (sensitivity / false alarms per 100 township-weeks), lead time vs EWARN / EWMA / CUSUM, figures, markdown.

  python scripts/dengue_alert_report.py --tag dengue
"""
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from dengue_ewarn.alerts import alert_metrics, auc, cusum_alarm, episodes, ewma_alarm  # noqa: E402
from dengue_ewarn.data import ewarn_threshold, rolling7  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="dengue")
ap.add_argument("--floor", type=float, default=3.0)
ap.add_argument("--p-grid", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ap.add_argument("--lead-window", type=int, default=14)
args = ap.parse_args()

P = {"300": "#B4C9B1", "500": "#739A6D", "600": "#5D7F58", "800": "#374C34"}; N = {"200": "#E4E7E4", "400": "#A2ABA0", "600": "#5D675B", "700": "#444C43", "900": "#181B18"}
LINE = {"primary": "#5D7F58", "blue": "#587A9D", "yellow": "#A8821F", "alert": "#BE373C"}
avail = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({"font.family": [f for f in ["Noto Sans TC", "PingFang TC", "Heiti TC", "Arial Unicode MS"] if f in avail] + ["DejaVu Sans"],
                     "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": N["200"], "grid.linewidth": 0.6})

bt = OUTPUT_DIR / "backtest"; figs = OUTPUT_DIR / "figures"; figs.mkdir(exist_ok=True)
files = [bt / f"{args.tag}_{m}_forecasts.parquet" for m in ("final", "asof", "asof_adj") if (bt / f"{args.tag}_{m}_forecasts.parquet").exists()]
res = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates(subset=["model", "mode", "series", "origin", "h"])
res["origin"] = pd.to_datetime(res["origin"])
modes = [m for m in ["final", "asof", "asof_adj"] if m in set(res["mode"])]
models = [m for m in ["tfm", "naive", "snaive"] if m in set(res["model"])]
years = sorted(res.year.unique()); big = {2014, 2015, 2023}
print(f"loaded {len(res):,} rows | modes {modes} | models {models} | years {years}")

counts = pd.read_parquet(PROCESSED_DIR / "dengue_township_daily.parquet"); dates = counts.index; series = list(counts.columns)
C = counts.to_numpy().T.astype(float); S7 = rolling7(C)
thr_final = np.stack([ewarn_threshold(C, t, floor=args.floor) for t in range(C.shape[1])], axis=1)  # (S, T) EWARN threshold each day
lines = [f"# 登革熱鄉鎮每日回測：閾值突破預警（CHG 情境一，第一版）", "",
         f"資料：登革熱每日確定病例（本土，居住鄉鎮），台南 37 區 + 高雄 38 區中 2012 起有病例的 {C.shape[0]} 區；目標 = 7 日累計病例；閾值 = max(2 × 前 3 週週均值, {args.floor:g} 例)（EWARN 規則加下限）。",
         f"起點：{', '.join(map(str, years))} 年 6–12 月每 2 天；horizon 1–14 天；context 2 年（730 天）。模式：final = 以最終資料為 context（無通報延遲）、asof = 只用起點當日已通報者、asof_adj = as-of 除以歷史通報完整度。", ""]

# ---- 1) forecast skill
lines += ["## 1. 7 日累計的分布預測表現（WIS 越低越好；cov80 理想 0.80）", ""]
sk = res[res.h > 0].groupby(["mode", "model", "h"]).agg(wis=("wis", "mean"), cov80=("cov80", "mean")).reset_index()
tab = sk.pivot_table(index=["mode", "model"], columns="h", values="wis").round(2); tab.columns = [f"WIS h={h}" for h in tab.columns]
cov = sk.pivot_table(index=["mode", "model"], columns="h", values="cov80").round(2); cov.columns = [f"cov80 h={h}" for h in cov.columns]
lines += [pd.concat([tab, cov[[c for c in cov.columns if c.endswith("h=7") or c.endswith("h=14")]]], axis=1).to_markdown(), ""]
sub7 = res[res.h == 7]
sk2 = sub7.assign(season=np.where(sub7.year.isin(big), "大流行年", "非流行年")).groupby(["mode", "model", "season"])["wis"].mean().unstack("season").round(2)
lines += ["h = 7 的 WIS 依年份類型（大流行年 = 2014、2015、2023）：", "", sk2.to_markdown(), ""]

# ---- 2) probability skill and p* sweep (weekly aggregation of daily/2-daily origins)
lines += ["## 2. 閾值突破預警：機率技巧與 p* 取捨", "",
          "事件 = 未來 14 天內 7 日累計 ≥ 起點當日的閾值；模型機率 = 各 horizon 機率的最大值。假警報率以「鄉鎮 × 週」計：該週任一起點發警示即算一次警示週，該週任一起點有事件即算事件週。", ""]
any_rows = res[res.h == 0].copy(); any_rows["week"] = any_rows.origin.dt.to_period("W")
sweep_rows = []
for mode in modes:
    for model in models:
        a = any_rows[(any_rows["mode"] == mode) & (any_rows.model == model)]
        wk = a.groupby(["series", "week"]).agg(prob=("prob", "max"), event=("event", "max")).reset_index()
        ev, pr = wk.event.to_numpy(bool), wk.prob.to_numpy(float)
        A = auc(ev, pr)
        for p in args.p_grid:
            m = alert_metrics(ev, pr, p); m.update(mode=mode, model=model, auc=A, n_weeks=len(wk), event_weeks=int(ev.sum())); sweep_rows.append(m)
sweep = pd.DataFrame(sweep_rows)
lines += ["AUC（鄉鎮週層級，事件 vs 模型機率）：", "", sweep.groupby(["mode", "model"])["auc"].first().unstack("model").round(3).to_markdown(), ""]
for mode in modes:
    t = sweep[(sweep["mode"] == mode) & (sweep.model == "tfm")][["p_star", "sensitivity", "false_alarm_per100", "ppv", "alerts"]].round(3)
    lines += [f"TimesFM（{mode}）p* 掃描（{int(sweep[(sweep['mode']==mode)&(sweep.model=='tfm')].n_weeks.iloc[0]):,} 鄉鎮週，其中事件週 {int(sweep[(sweep['mode']==mode)&(sweep.model=='tfm')].event_weeks.iloc[0]):,}）：", "", t.to_markdown(index=False), ""]

# ---- 3) episodes, lead time, and statistical detectors on final daily counts
lines += ["## 3. 前置時間：模型預警 vs EWARN 靜態規則、EWMA、CUSUM", "",
          f"事件（episode）= 最終資料的 7 日累計首次達到當日 EWARN 閾值，且之前至少 14 天低於閾值。EWARN 靜態規則在事件當天才觸發（前置 0 天）。模型：事件前 {args.lead_window} 天內最早發出警示（機率 ≥ p*）的起點到事件日的天數；EWMA/CUSUM：事件前 {args.lead_window} 天內最早的警報日。", ""]
ewma = ewma_alarm(C); cusum = cusum_alarm(C)
ep = [(s_i, t) for s_i in range(C.shape[0]) for t in episodes(S7[s_i], thr_final[s_i]) if dates[t].year in years and 6 <= dates[t].month <= 12]
lines += [f"事件數（{', '.join(map(str, years))} 年 6–12 月）：{len(ep)}，其中大流行年 {sum(1 for _, t in ep if dates[t].year in big)}。", ""]

_prob_cache = {}
def model_leads(mode, model, p_star):
    if (mode, model) not in _prob_cache:
        a = res[(res.h == 0) & (res["mode"] == mode) & (res.model == model)]
        _prob_cache[(mode, model)] = dict(zip(zip(a.series, a.origin), a.prob))
    key = _prob_cache[(mode, model)]
    leads = []
    for s_i, t in ep:
        # lead = days between the earliest alerting origin inside the window and the crossing day
        cand = [k for k in range(1, args.lead_window + 1) if key.get((series[s_i], dates[t - k]), -1) >= p_star]
        leads.append(max(cand) if cand else np.nan)
    return np.array(leads, dtype=float)

def detector_leads(alarm):
    leads = []
    for s_i, t in ep:
        days = [t - k for k in range(0, args.lead_window + 1) if alarm[s_i, t - k]]
        leads.append((t - min(days)) if days else np.nan)
    return np.array(leads, dtype=float)

rows = []
for name, lead in [("EWARN 靜態規則", np.zeros(len(ep))), ("EWMA", detector_leads(ewma)), ("CUSUM", detector_leads(cusum))]:
    rows.append({"方法": name, "偵測到的事件比例": np.mean(~np.isnan(lead)), "前置時間中位數（天）": np.nanmedian(lead) if np.any(~np.isnan(lead)) else np.nan, "前置 ≥ 3 天的比例": np.nanmean(lead >= 3) if np.any(~np.isnan(lead)) else np.nan})
for mode in modes:
    for p in (0.3, 0.5, 0.7):
        lead = model_leads(mode, "tfm", p)
        rows.append({"方法": f"TimesFM {mode} p*={p}", "偵測到的事件比例": np.mean(~np.isnan(lead)), "前置時間中位數（天）": np.nanmedian(lead) if np.any(~np.isnan(lead)) else np.nan, "前置 ≥ 3 天的比例": np.nanmean(lead >= 3) if np.any(~np.isnan(lead)) else np.nan})
lead_tab = pd.DataFrame(rows).round(2)
lines += [lead_tab.to_markdown(index=False), ""]
# false alarms of the detectors (weekly, outside ±14 days of any episode)
ep_mask = np.zeros(C.shape, dtype=bool)
for s_i, t in ep: ep_mask[s_i, max(0, t - 14): t + 15] = True
in_years = np.array([d.year in years and 6 <= d.month <= 12 for d in dates])
det_rows = []
for name, alarm in [("EWMA", ewma), ("CUSUM", cusum)]:
    sel = in_years[None, :] & ~ep_mask
    wk_idx = pd.Series(dates).dt.to_period("W").astype(str).to_numpy()
    fa = pd.DataFrame({"s": np.repeat(np.arange(C.shape[0]), C.shape[1]), "w": np.tile(wk_idx, C.shape[0]), "alarm": alarm.ravel(), "sel": sel.ravel()})
    fa = fa[fa.sel].groupby(["s", "w"])["alarm"].max()
    det_rows.append({"方法": name, "非事件鄉鎮週數": len(fa), "假警報週": int(fa.sum()), "每 100 鄉鎮週假警報": round(100 * fa.mean(), 2)})
lines += ["統計偵測器在非事件期間（事件前後 14 天以外）的假警報：", "", pd.DataFrame(det_rows).to_markdown(index=False), ""]
# matched comparison: model sensitivity / lead at a false-alarm rate no higher than each detector's
match_rows = []
for det in det_rows:
    far = det["每 100 鄉鎮週假警報"]
    for mode in modes:
        sw = sweep[(sweep["mode"] == mode) & (sweep.model == "tfm") & (sweep.false_alarm_per100 <= far)].sort_values("sensitivity", ascending=False)
        if len(sw):
            p = float(sw.iloc[0].p_star); lead = model_leads(mode, "tfm", p)
            match_rows.append({"對照偵測器": det["方法"], "其假警報/100 鄉鎮週": far, "模式": mode, "模型 p*": p, "模型假警報/100": round(float(sw.iloc[0].false_alarm_per100), 2),
                               "模型敏感度（週）": round(float(sw.iloc[0].sensitivity), 3), "模型偵測事件比例": round(float(np.mean(~np.isnan(lead))), 2), "模型前置中位數（天）": float(np.nanmedian(lead)) if np.any(~np.isnan(lead)) else np.nan})
if match_rows:
    lines += ["配對比較：在不高於各偵測器假警報率的 p* 下，模型的敏感度與前置時間：", "", pd.DataFrame(match_rows).to_markdown(index=False), ""]
pd.DataFrame(det_rows).to_csv(bt / f"{args.tag}_detectors.csv", index=False)
pd.DataFrame(match_rows).to_csv(bt / f"{args.tag}_matched.csv", index=False)
# per-method lead-time arrays for the site (distribution plots)
lead_long = []
for name, lead in [("EWMA", detector_leads(ewma)), ("CUSUM", detector_leads(cusum))] + [(f"TimesFM {m} p*={pp}", model_leads(m, "tfm", pp)) for m in modes for pp in args.p_grid]:
    for (s_i, t), lv in zip(ep, lead):
        lead_long.append({"method": name, "series": series[s_i], "crossing": dates[t], "lead": lv})
pd.DataFrame(lead_long).to_csv(bt / f"{args.tag}_leads_long.csv", index=False)

# ---- 4) figures: example townships in 2015 / 2023 Tainan
def example(mode, series_name, year, path):
    """Two stacked panels (shared x): cases + EWARN threshold on top, model crossing probability below."""
    a = res[(res.h == 0) & (res["mode"] == mode) & (res.model == "tfm") & (res.series == series_name) & (res.year == year)].sort_values("origin")
    s_i = series.index(series_name); sel = (dates.year == year) & (dates.month >= 5)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), dpi=170, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax.plot(dates[sel], S7[s_i, sel], color=LINE["primary"], lw=2.2, label="7 日累計病例（最終資料）")
    ax.plot(dates[sel], thr_final[s_i, sel], color=N["600"], lw=1.2, ls="--", label="EWARN 閾值（2 × 前 3 週均值，≥ 3）")
    ax2.plot(a.origin, a.prob, color=LINE["blue"], lw=1.6, label="模型：14 天內突破閾值的機率")
    ax2.axhline(0.5, color=N["400"], lw=1, ls=":"); ax2.set_ylim(0, 1.02); ax2.set_ylabel("機率")
    for t in [t for si, t in ep if si == s_i and dates[t].year == year]:
        for axx in (ax, ax2): axx.axvline(dates[t], color=LINE["alert"], lw=1, ls=":")
    ax.set_ylabel("病例數"); ax.set_title(f"{series_name.replace('|', ' ')} {year}（context 模式：{mode}）　紅色虛線 = 實際突破閾值日", loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

peak_series = {}
for year in [y for y in (2015, 2023) if y in years]:
    sel = dates.year == year; peak_series[year] = series[int(np.argmax(S7[:, sel].max(axis=1)))]
mode_fig = "asof_adj" if "asof_adj" in modes else modes[0]
for year, sname in peak_series.items():
    p = figs / f"dengue_{year}_{sname.split('|')[1]}_{mode_fig}.png"; example(mode_fig, sname, year, p)
    lines += [f"## 案例：{sname.replace('|', ' ')} {year}", "", f"![]( ../figures/{p.name})", ""]
(bt / f"{args.tag}_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
sweep.to_csv(bt / f"{args.tag}_alert_sweep.csv", index=False); lead_tab.to_csv(bt / f"{args.tag}_lead_times.csv", index=False)
print("\n".join(lines[:60])); print("report:", bt / f"{args.tag}_REPORT.md")
