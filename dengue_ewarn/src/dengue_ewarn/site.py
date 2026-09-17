"""Build the dengue early-warning visual report (docs/dengue/index.html + data.json + figures)."""
from __future__ import annotations

import datetime as dt
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from . import OUTPUT_DIR, PROCESSED_DIR, SITE_DIR
from .alerts import episodes, prob_ge
from .data import completeness, ewarn_threshold, rolling7

P = {"300": "#B4C9B1", "500": "#739A6D", "600": "#5D7F58", "800": "#374C34"}
N = {"200": "#E4E7E4", "400": "#A2ABA0", "600": "#5D675B", "700": "#444C43", "900": "#181B18"}
LINE = {"primary": "#5D7F58", "blue": "#587A9D", "alert": "#BE373C"}
BT = OUTPUT_DIR / "backtest"; FIGS = OUTPUT_DIR / "figures"
MODE_LABEL = {"final": "final（最終資料，無通報延遲）", "asof": "asof（僅起點當日已通報）", "asof_adj": "asof_adj（已通報 ÷ 通報完整度）"}
HS = [1, 3, 5, 7, 10, 14]


def _fonts():
    avail = {f.name for f in font_manager.fontManager.ttflist}
    fam = [f for f in ["Noto Sans TC", "PingFang TC", "Heiti TC", "Arial Unicode MS"] if f in avail] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": N["200"], "grid.linewidth": 0.6, "font.size": 10.5})


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else round(float(x), nd)


def illustration(res: pd.DataFrame, tag: str, years=(2015, 2023), min_lead: int = 3, min_cases: float = 20) -> tuple[str, dict]:
    """One origin, one township: as-of-adjusted context, quantile fan, threshold, crossing probabilities.

    Picks, over all townships and the given years, the (episode, origin) pair where the model flagged
    most strongly at least `min_lead` days before an actual crossing of a threshold >= min_cases.
    """
    counts = pd.read_parquet(PROCESSED_DIR / "dengue_township_daily.parquet"); dates = counts.index
    z = np.load(PROCESSED_DIR / "dengue_delay_hist.npz", allow_pickle=True); hist = z["hist"]; series = list(z["series"])
    C = counts.to_numpy().T.astype(float); S7 = rolling7(C)
    thr_full = np.stack([ewarn_threshold(C, t) for t in range(C.shape[1])], axis=1)  # (S, T)
    a_all = res[(res.h == 0) & (res["mode"] == "asof_adj") & (res.model == "tfm") & (res.year.isin(years))]
    best = None
    for s_i, sname in enumerate(series):
        a = a_all[a_all.series == sname]
        if a.empty:
            continue
        for cc in episodes(S7[s_i], thr_full[s_i]):
            if dates[cc].year not in years or thr_full[s_i, cc] < min_cases:
                continue
            cand = a[(a.origin >= dates[cc - 14]) & (a.origin <= dates[cc - min_lead])]
            if len(cand):
                top = cand.sort_values("prob", ascending=False).iloc[0]
                if best is None or top.prob > best[1]:
                    best = (s_i, float(top.prob), top.origin, cc)
    s_i, _, origin, c = best
    series_name = series[s_i]; thr_all = thr_full[s_i]
    d = dates.get_loc(origin)
    rows = res[(res["mode"] == "asof_adj") & (res.model == "tfm") & (res.series == series_name) & (res.origin == origin) & (res.h > 0)].sort_values("h")
    # as-of adjusted context for this township
    K = hist.shape[2]; k = np.minimum(d - np.arange(d + 1), K - 1)
    known = np.cumsum(hist[s_i, : d + 1, :], axis=1)[np.arange(d + 1), k].astype(float)
    cc = np.maximum(completeness(hist, d), 0.2); adj = 1.0 / cc[k[-K:]] if d + 1 >= K else 1.0 / cc[k]
    known[-len(adj):] *= adj
    ctx = rolling7(known)[-60:]; ctx_dates = dates[d - 59: d + 1]
    fdates = [dates[d + h] for h in rows.h]; thr = float(rows.thr.iloc[0])
    _fonts(); fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=170)
    ax.plot(ctx_dates, ctx, color=LINE["primary"], lw=2.2, label="起點當日可得的 7 日累計（已通報 ÷ 完整度）")
    ax.plot(dates[d - 59: d + 15], S7[s_i, d - 59: d + 15], color=N["600"], lw=1.2, ls="--", label="最終資料的 7 日累計（事後才知道）")
    ax.fill_between(fdates, rows.q10, rows.q90, color=P["300"], alpha=0.25, linewidth=0, label="80% 預測區間（q10–q90）")
    ax.fill_between(fdates, rows.q20, rows.q80, color=P["300"], alpha=0.5, linewidth=0, label="60% 預測區間（q20–q80）")
    ax.plot(fdates, rows["median"], color=LINE["primary"], lw=2.2, ls=(0, (6, 3)), marker="o", ms=4, label="預測中位數")
    ax.axhline(thr, color=LINE["alert"], lw=1.4, ls="--"); ax.text(ctx_dates[0], thr, f" 閾值 {thr:.0f}（2 × 前 3 週均值，≥ 3）", color=LINE["alert"], fontsize=9, va="bottom")
    ymax = max(float(rows.q90.max()), float(S7[s_i, d - 59: d + 15].max()), thr) * 1.25
    ax.set_ylim(0, ymax)
    ax.axvline(origin, color=N["400"], lw=1, ls=":"); ax.text(origin, ymax * 0.58, f"預測起點 {origin.date()} ", color=N["600"], fontsize=9, va="center", ha="right")
    for i, (h, fd_, q) in enumerate(zip(rows.h, fdates, rows.prob)):  # probabilities listed along the top, staggered
        ax.annotate(f"h={h}\nP={q:.2f}", (fd_, ymax * (0.97 if i % 2 == 0 else 0.86)), ha="center", va="top", fontsize=8, color=P["800"], fontweight="bold")
        ax.plot([fd_, fd_], [float(rows[rows.h == h].q90.iloc[0]), ymax * (0.905 if i % 2 == 0 else 0.795)], color=N["200"], lw=0.8)
    ax.set_ylabel("7 日累計病例"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{series_name.replace('|', ' ')}：從 {origin.date()} 往後 14 天的預測與突破閾值機率（P = 分布中超過閾值的比例）", loc="left", fontsize=10.5)
    fig.tight_layout(); path = FIGS / f"{tag}_illustration.png"; fig.savefig(path); plt.close(fig)
    info = {"series": series_name, "origin": str(origin.date()), "threshold": thr, "crossing": str(dates[c].date()),
            "rows": [{"h": int(h), "date": str(fd_.date()), "median": r(m, 1), "q10": r(q1, 1), "q90": r(q9, 1), "prob": r(p, 2), "truth": r(S7[s_i, d + int(h)], 0)}
                     for h, fd_, m, q1, q9, p in zip(rows.h, fdates, rows["median"], rows.q10, rows.q90, rows.prob)]}
    return path.name, info


def build(tag: str = "dengue", n_sites: int = 20) -> dict:
    SITE_DIR.mkdir(parents=True, exist_ok=True); (SITE_DIR / "figures").mkdir(exist_ok=True)
    files = [BT / f"{tag}_{m}_forecasts.parquet" for m in ("final", "asof", "asof_adj") if (BT / f"{tag}_{m}_forecasts.parquet").exists()]
    res = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True); res["origin"] = pd.to_datetime(res["origin"])
    modes = [m for m in ("asof_adj", "final", "asof") if m in set(res["mode"])]
    sweep = pd.read_csv(BT / f"{tag}_alert_sweep.csv"); leads = pd.read_csv(BT / f"{tag}_lead_times.csv")
    det = pd.read_csv(BT / f"{tag}_detectors.csv"); matched = pd.read_csv(BT / f"{tag}_matched.csv"); leads_long = pd.read_csv(BT / f"{tag}_leads_long.csv")
    years = sorted(int(y) for y in res.year.unique()); big = {2014, 2015, 2023}
    # --- WIS / coverage by horizon
    g = res[res.h > 0].groupby(["mode", "model", "h"]).agg(wis=("wis", "mean"), cov80=("cov80", "mean")).reset_index()
    wis_by_h = {m: {mdl: [r(v, 2) for v in g[(g["mode"] == m) & (g.model == mdl)].sort_values("h").wis] for mdl in ("tfm", "naive", "snaive")} for m in modes}
    cov_by_h = {m: {mdl: [r(v, 3) for v in g[(g["mode"] == m) & (g.model == mdl)].sort_values("h").cov80] for mdl in ("tfm", "naive")} for m in modes}
    h7 = res[res.h == 7].assign(kind=lambda d: np.where(d.year.isin(big), "epidemic", "quiet")).groupby(["mode", "model", "kind"]).wis.mean()
    wis_h7 = {m: {mdl: {k: r(h7.get((m, mdl, k), np.nan), 2) for k in ("epidemic", "quiet")} for mdl in ("tfm", "naive", "snaive")} for m in modes}
    # --- alerts: AUC, sweep, calibration (weekly aggregation as in the report)
    auc = {m: {mdl: r(sweep[(sweep["mode"] == m) & (sweep.model == mdl)].auc.iloc[0], 3) for mdl in ("tfm", "naive", "snaive")} for m in modes}
    sw = {m: sweep[(sweep["mode"] == m) & (sweep.model == "tfm")].sort_values("p_star") for m in modes}
    sweep_json = {m: {"p": sw[m].p_star.round(2).tolist(), "sens": sw[m].sensitivity.round(3).tolist(), "far": sw[m].false_alarm_per100.round(2).tolist(),
                      "ppv": sw[m].ppv.round(3).tolist(), "alerts": sw[m].alerts.astype(int).tolist(), "n_weeks": int(sw[m].n_weeks.iloc[0]), "event_weeks": int(sw[m].event_weeks.iloc[0])} for m in modes}
    anyr = res[(res.h == 0) & (res.model == "tfm") & (res["mode"] == "asof_adj")].copy(); anyr["week"] = anyr.origin.dt.to_period("W")
    wk = anyr.groupby(["series", "week"]).agg(prob=("prob", "max"), event=("event", "max")).reset_index()
    bins = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]); idx = np.digitize(wk.prob, bins) - 1
    calib = [{"bin": f"{bins[i]:.1f}–{bins[i+1] if i < 9 else 1.0:.1f}", "n": int((idx == i).sum()), "observed": r(wk.event[idx == i].mean(), 3) if (idx == i).any() else None, "mid": r((bins[i] + min(bins[i + 1], 1)) / 2, 2)} for i in range(10)]
    # --- leads
    lead_tab = leads.to_dict(orient="records")
    lead_stats = {}
    for name, grp in leads_long.groupby("method"):
        lv = grp.lead.dropna(); lead_stats[name] = {"detected": r(grp.lead.notna().mean(), 3), "median": r(lv.median(), 1) if len(lv) else None, "ge3": r((lv >= 3).mean(), 3) if len(lv) else None, "n": int(len(grp))}
    # --- workload example (asof_adj): alerts per township-week × n_sites
    work = []
    for _, row in sw["asof_adj"].iterrows():
        p = round(float(row.p_star), 1); ls = lead_stats.get(f"TimesFM asof_adj p*={p}", {})
        work.append({"p": p, "sens": r(row.sensitivity, 2), "far": r(row.false_alarm_per100, 2), "ppv": r(row.ppv, 2),
                     "alerts_per_site_week": r(row.alerts / row.n_weeks, 4), "alerts_week_district": r(row.alerts / row.n_weeks * n_sites, 1),
                     "false_week_district": r(row.alerts / row.n_weeks * n_sites * (1 - (row.ppv if not np.isnan(row.ppv) else 0)), 1),
                     "detected": ls.get("detected"), "lead_median": ls.get("median")})
    fig_name, illus = illustration(res, tag)
    for f in list(FIGS.glob(f"{tag}_*.png")):
        shutil.copy(f, SITE_DIR / "figures" / f.name)
    cases = sorted(p.name for p in FIGS.glob(f"{tag}_20*_*.png"))
    data = {"generated_at": dt.datetime.now().astimezone().isoformat(timespec="minutes"), "years": years, "n_series": int(res.series.nunique()),
            "n_origins_per_year": int(res[(res.h == 1) & (res.model == "tfm") & (res["mode"] == modes[0])].groupby("year").origin.nunique().iloc[0]),
            "modes": modes, "mode_label": MODE_LABEL, "hs": HS, "wis_by_h": wis_by_h, "cov_by_h": cov_by_h, "wis_h7": wis_h7, "auc": auc, "sweep": sweep_json,
            "calibration": calib, "lead_table": lead_tab, "lead_stats": lead_stats, "detectors": det.to_dict(orient="records"), "matched": matched.to_dict(orient="records"),
            "workload": work, "n_sites_example": n_sites, "illustration": {"figure": fig_name, **illus}, "cases": cases,
            "n_events": int(leads_long[leads_long.method == "CUSUM"].shape[0])}
    (SITE_DIR / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (SITE_DIR / "index.html").write_text(render_html(data), encoding="utf-8")
    return data


def render_html(D: dict) -> str:
    fmt = lambda v, nd=2: "—" if v is None else f"{v:,.{nd}f}"
    aa = D["auc"]["asof_adj"]; sw = D["sweep"]["asof_adj"]; i5 = sw["p"].index(0.5) if 0.5 in sw["p"] else 4
    ls5 = D["lead_stats"].get("TimesFM asof_adj p*=0.5", {}); ls3 = D["lead_stats"].get("TimesFM asof_adj p*=0.3", {})
    w7 = D["wis_h7"]["asof_adj"]; impr = 100 * (1 - w7["tfm"]["epidemic"] / w7["naive"]["epidemic"])
    wf = D["wis_h7"]["final"]; impr_f = 100 * (1 - wf["tfm"]["epidemic"] / wf["naive"]["epidemic"])
    il = D["illustration"]; cus = next(x for x in D["detectors"] if x["方法"] == "CUSUM"); ewm = next(x for x in D["detectors"] if x["方法"] == "EWMA")
    mc = [m for m in D["matched"] if m["模式"] == "asof_adj"]
    def matched_row(det):
        m = next((x for x in mc if x["對照偵測器"] == det), None)
        return m
    m_cus, m_ewm = matched_row("CUSUM"), matched_row("EWMA")
    lead_rows = "".join(f"<tr class='{'best' if x['方法'].startswith('TimesFM asof_adj p*=0.5') else ('baseline' if not x['方法'].startswith('TimesFM') else '')}'><td>{x['方法']}</td><td class='num'>{fmt(x['偵測到的事件比例'])}</td><td class='num'>{fmt(x['前置時間中位數（天）'], 1)}</td><td class='num'>{fmt(x['前置 ≥ 3 天的比例'])}</td></tr>" for x in D["lead_table"])
    work_rows = "".join(f"<tr class='{'best' if w['p'] == 0.5 else ''}'><td class='num'>{w['p']}</td><td class='num'>{fmt(w['sens'])}</td><td class='num'>{fmt(w['far'])}</td><td class='num'>{fmt(w['ppv'])}</td><td class='num'>{fmt(w['alerts_week_district'], 1)}</td><td class='num'>{fmt(w['false_week_district'], 1)}</td><td class='num'>{fmt(w['detected'])}</td><td class='num'>{fmt(w['lead_median'], 1)}</td></tr>" for w in D["workload"])
    il_rows = "".join(f"<tr><td class='num'>{x['h']}</td><td>{x['date']}</td><td class='num'>{fmt(x['median'], 0)}</td><td class='num'>{fmt(x['q10'], 0)}–{fmt(x['q90'], 0)}</td><td class='num'><b>{fmt(x['prob'])}</b></td><td class='num'>{fmt(x['truth'], 0)}</td></tr>" for x in il["rows"])
    wis_rows = "".join(f"<tr><td>{D['mode_label'][m]}</td>" + "".join(f"<td class='num'>{fmt(D['wis_by_h'][m][mdl][D['hs'].index(7)])}</td>" for mdl in ("tfm", "naive", "snaive")) + f"<td class='num'>{fmt(D['wis_h7'][m]['tfm']['epidemic'])} / {fmt(D['wis_h7'][m]['naive']['epidemic'])}</td><td class='num'>{fmt(D['cov_by_h'][m]['tfm'][D['hs'].index(7)])}</td><td class='num'>{fmt(D['auc'][m]['tfm'])} / {fmt(D['auc'][m]['naive'])}</td></tr>" for m in D["modes"])
    cases_html = "".join(f"<div class='card'><h3>{'台南市 北區 2015：史上最大流行' if '2015' in c else '台南市 南區 2023：2015 後最大流行'}</h3><div class='sub'>上：最終資料的 7 日累計與 EWARN 閾值（虛線）；下：模型每個起點給的「14 天內突破閾值」機率；紅色虛線 = 實際突破日</div><img src='figures/{c}' alt='{c}' style='width:100%'></div>" for c in D["cases"])
    return f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>登革熱鄉鎮預測式預警 · TimesFM 3.0 驗證報告</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;600;700&family=Noto+Serif+TC:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/epi.css?v=dengue1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="../assets/charts.js?v=dengue1" defer></script>
<style>.readme li{{margin-bottom:6px}} .callout{{background:var(--p-50);border-left:3px solid var(--p-400);padding:12px 16px;margin:14px 0;font-size:14px}} .steps{{counter-reset:s;list-style:none;padding:0}} .steps li{{position:relative;padding-left:38px;margin-bottom:10px}} .steps li::before{{counter-increment:s;content:counter(s);position:absolute;left:0;top:1px;width:26px;height:26px;border-radius:50%;background:var(--p-500);color:#fff;font-weight:700;font-size:13px;display:grid;place-items:center}} img.fig{{width:100%;border:1px solid var(--n-200);border-radius:4px;background:#fff}}</style>
</head><body>
<header class="masthead"><div class="masthead-inner">
  <div class="masthead-left"><div class="crest">FT</div><div class="masthead-title">登革熱鄉鎮預測式預警 · TimesFM 3.0<small>DENGUE-EWARN · CHG SCENARIO 1 · VALIDATION REPORT</small></div></div>
  <nav class="masthead-right"><a href="../index.html">流感每週預測</a><a href="../backtest.html">流感回測</a><a href="../report.html">流感報告</a><a href="index.html" class="active">登革熱預警</a></nav>
</div></header>
<main>
  <div class="hero-kicker">Validation report · {D['generated_at'][:10]}</div>
  <h1 class="page-title">登革熱鄉鎮層級預測式預警：<em>TimesFM 3.0 在承平期資料上的驗證</em></h1>
  <p class="lead">對應 CHG 情境一（複合災難下收容所自報的離線早期警示）第一至二年的驗證工作：以疾管署登革熱每日確定病例，模擬「小區域、每日、低計數」的症候群序列，檢驗時序基礎模型能否在 WHO EWARN 靜態閾值被突破之前數天、帶著不確定區間發出預警。本頁說明資料、方法、模型輸出的讀法、驗證結果與應用情境。</p>
  <div class="meta-strip"><span>資料：疾管署登革熱每日確定病例（本土，居住鄉鎮），台南 + 高雄 <b>{D['n_series']} 個鄉鎮</b></span><span>回測年份：<b>{'、'.join(map(str, D['years']))}</b> 年 6–12 月，每 2 天一個起點（每年 {D['n_origins_per_year']} 個）</span><span>模型：TimesFM 3.0（零樣本，未微調）</span></div>

  <div class="kpi-strip">
    <div class="kpi"><div class="label">突破閾值的預警能力（AUC）</div><div class="value">{fmt(aa['tfm'])}</div><div class="sub">只用當日已通報資料 + 完整度校正；naive {fmt(aa['naive'])}、無校正 {fmt(D['auc']['asof']['tfm'])}</div></div>
    <div class="kpi"><div class="label">前置時間中位數（p* = 0.5）</div><div class="value">{fmt(ls5.get('median'), 0)}<small>天</small></div><div class="sub">EWARN 靜態規則 0 天、CUSUM 1 天、EWMA 2 天</div></div>
    <div class="kpi"><div class="label">假警報（p* = 0.5）</div><div class="value">{fmt(sw['far'][i5], 1)}<small>/100 鄉鎮週</small></div><div class="sub">敏感度 {fmt(sw['sens'][i5])}；EWMA {fmt(ewm['每 100 鄉鎮週假警報'])}、CUSUM {fmt(cus['每 100 鄉鎮週假警報'])}</div></div>
    <div class="kpi"><div class="label">7 日累計 WIS vs naive（大流行年，h = 7）</div><div class="value">−{impr:.0f}%</div><div class="sub">無通報延遲時 −{impr_f:.0f}%；季節性 naive 對登革熱無用</div></div>
  </div>

  <section>
    <div class="section-head"><span class="section-num">01</span><h2>要回答的問題</h2><span class="en">From scenario to testable question</span></div>
    <div class="grid-2">
      <div class="card"><h3>情境一的要求</h3><ul class="findings readme">
        <li>收容所層級、每日、低計數且零膨脹的症候群序列，離線可用。</li>
        <li>不只判定「現在是否超標」，而是前瞻 N 天、帶不確定區間地預測「是否即將突破流行閾值」。</li>
        <li>比 WHO EWARN 的靜態規則（達前 3 週均值兩倍）更早，且假警報可控；校準衰退時要能降級。</li></ul></div>
      <div class="card"><h3>承平期的替代驗證</h3><ul class="findings readme">
        <li>以「鄉鎮 × 日」的登革熱本土確定病例模擬「收容所 × 日」，涵蓋 2014 高雄、2015 台南高雄、2023 台南三次大流行與三個平靜年。</li>
        <li>目標 = 7 日累計病例；閾值 = max(2 × 前 3 週週均值, 3 例)。事件 = 未來 14 天內 7 日累計 ≥ 起點閾值。</li>
        <li>誠實回測：用通報日重建每個起點「當日可得」的資料，並比較有無通報完整度校正。</li></ul></div>
    </div>
  </section>

  <section>
    <div class="section-head"><span class="section-num">02</span><h2>資料與方法</h2><span class="en">Data &amp; method</span></div>
    <div class="tbl-wrap"><table class="tbl compact"><tbody>
      <tr><td><b>資料</b></td><td>疾管署「登革熱 1998 年起每日確定病例統計」個案檔（發病日、通報日、居住鄉鎮、本土/境外）。公開平台已於 2026 年 3 月下架，使用 Internet Archive 保存的官方檔案（發病日至 2025-07-23）。</td></tr>
      <tr><td><b>面板</b></td><td>台南 37 區、高雄 38 區中 2012 年起有本土病例的 {D['n_series']} 區 × 日；同時建立「發病日 × 通報延遲」直方圖，通報延遲中位數 2 天、第 95 百分位 7 天，起點當日近 7 天約僅 68% 完整。</td></tr>
      <tr><td><b>三種 context 模式</b></td><td><b>final</b>：用最終資料（無延遲，樂觀上限）；<b>asof</b>：只用起點當日已通報者（誠實但尾端偏低）；<b>asof_adj</b>：as-of 除以歷史通報完整度（下限 0.2），是實務可行的版本。</td></tr>
      <tr><td><b>模型</b></td><td>TimesFM 3.0 零樣本，單變量批次，context 2 年，horizon 1–14 天，輸出 9 個分位數；基準 last-value naive、季節性 naive（364 天）；偵測器 EWARN 靜態規則、EWMA、CUSUM。</td></tr>
      <tr><td><b>評估</b></td><td>分布：WIS（近似 CRPS）、80% 涵蓋率。警示：事件 vs 機率的 AUC；p* 掃描下的敏感度、每 100 鄉鎮週假警報、PPV（鄉鎮週層級）；前置時間 = 事件前 14 天內最早警示到突破日的天數；配對比較 = 在不高於偵測器假警報率的 p* 下比較。</td></tr>
    </tbody></table></div>
  </section>

  <section>
    <div class="section-head"><span class="section-num">03</span><h2>模型輸出是什麼、怎麼讀</h2><span class="en">How to read the output</span></div>
    <div class="card"><h3>一個起點的完整輸出：{il['series'].replace('|', ' ')}，{il['origin']}（實際突破日 {il['crossing']}）</h3>
      <div class="sub">綠實線是起點當天能拿到的資料（已通報並做完整度校正），灰虛線是事後才知道的最終值；扇形是模型對未來 14 天 7 日累計的分布，紅虛線是當天的 EWARN 閾值</div>
      <img class="fig" src="figures/{il['figure']}" alt="illustration">
      <div class="tbl-wrap" style="margin-top:12px"><table class="tbl compact"><thead><tr><th class="num">h（天）</th><th>目標日</th><th class="num">中位數</th><th class="num">80% 區間</th><th class="num">P(≥ 閾值 {il['threshold']:.0f})</th><th class="num">事後實際</th></tr></thead><tbody>{il_rows}</tbody></table></div></div>
    <div class="grid-2" style="margin-top:18px">
      <div class="card"><h3>讀圖四步</h3><ol class="steps">
        <li><b>先看區間，不看單點。</b>模型輸出的是分布：中位數是「一半機會比它高、一半比它低」，60%/80% 區間表示大部分可能的範圍。區間寬代表不確定，是資訊不是缺陷。</li>
        <li><b>機率 = 分布超過閾值的比例。</b>每個 horizon 的 P(≥ 閾值) 由分位數內插而得；頁面與儀表板用的是 14 天內的最大值，代表「這兩週內任一時點突破」的機率。</li>
        <li><b>p* 是行動門檻，不是模型參數。</b>機率 ≥ p* 才發警示；p* 越低越早、越敏感，但假警報越多。第 6 節把 p* 換成每週查證單數量，由防疫單位選。</li>
        <li><b>前置時間才是價值。</b>EWARN 規則在突破當天才知道；模型在事件前幾天就把機率拉高，這段時間就是病媒防治、隔離空間與 ORS 整備的時間。</li></ol></div>
      <div class="card"><h3>三種資料狀態的意義</h3><ul class="findings readme">
        <li><b>final</b> 是理論上限：假設所有病例當天就通報。實務上不存在。</li>
        <li><b>asof</b> 是災難現場真正看到的樣子：最近幾天永遠偏低。直接餵給模型，它會把「通報未齊」誤讀成「疫情下降」，前置時間幾乎歸零（見第 4 節）。</li>
        <li><b>asof_adj</b> 用歷史完整度把最近幾天放大回估計值，是一種簡單的 nowcast。這一步讓 AUC 從 {fmt(D['auc']['asof']['tfm'])} 回到 {fmt(aa['tfm'])}、前置時間回到 {fmt(ls5.get('median'), 0)} 天，是部署時的必要元件；文件裡的「分布位移偵測 → 降級」也應以完整度異常為第一個觸發條件。</li>
        <li>閾值本身是反應式的：疫情爆發後閾值隨之升高、遠在病例之上，因此事件集中在流行「起始」與「加速」時刻，這正是 EWARN 想抓的時點。</li></ul></div>
    </div>
  </section>

  <section>
    <div class="section-head"><span class="section-num">04</span><h2>驗證結果</h2><span class="en">Results</span></div>
    <div class="grid-2">
      <div class="card"><h3>7 日累計的 WIS 依 horizon</h3><div class="sub">越低越好；asof_adj 為實務可得資料，final 為無延遲上限</div><div class="chart-box"><canvas id="c-wis"></canvas></div></div>
      <div class="card"><h3>校準：預測機率 vs 實際發生比例（asof_adj）</h3><div class="sub">鄉鎮週層級；點落在對角線附近表示機率可直接當機率用</div><div class="chart-box"><canvas id="c-cal"></canvas></div></div>
      <div class="card"><h3>p* 取捨：敏感度</h3><div class="sub">事件週中被警示的比例</div><div class="chart-box short"><canvas id="c-sens"></canvas></div></div>
      <div class="card"><h3>p* 取捨：每 100 鄉鎮週假警報</h3><div class="sub">灰線為 EWMA、CUSUM 在非事件期的假警報率</div><div class="chart-box short"><canvas id="c-far"></canvas></div></div>
    </div>
    <div class="tbl-wrap" style="margin-top:18px"><table class="tbl compact"><thead><tr><th>context 模式</th><th class="num">WIS h=7 TimesFM</th><th class="num">naive</th><th class="num">季節性 naive</th><th class="num">大流行年 TimesFM / naive</th><th class="num">80% 涵蓋（h=7）</th><th class="num">AUC TimesFM / naive</th></tr></thead><tbody>{wis_rows}</tbody></table></div>
    <div class="grid-2" style="margin-top:18px">
      <div class="card"><h3>前置時間：偵測到的事件比例</h3><div class="sub">{D['n_events']} 個突破事件；14 天窗內有無警示</div><div class="chart-box"><canvas id="c-lead-det"></canvas></div></div>
      <div class="card"><h3>前置時間中位數（天）</h3><div class="sub">EWARN 靜態規則定義上為 0</div><div class="chart-box"><canvas id="c-lead-med"></canvas></div></div>
    </div>
    <div class="tbl-wrap" style="margin-top:18px"><table class="tbl compact"><thead><tr><th>方法</th><th class="num">偵測到的事件比例</th><th class="num">前置中位數（天）</th><th class="num">前置 ≥ 3 天</th></tr></thead><tbody>{lead_rows}</tbody></table></div>
    <p class="note"><b>配對比較。</b>在不高於 CUSUM 假警報率（{fmt(cus['每 100 鄉鎮週假警報'])}/100 鄉鎮週）的 p* 下，TimesFM（asof_adj，p* = {m_cus['模型 p*'] if m_cus else '—'}）偵測 {fmt(m_cus['模型偵測事件比例']) if m_cus else '—'} 的事件、前置中位數 {fmt(m_cus['模型前置中位數（天）'], 0) if m_cus else '—'} 天；CUSUM 偵測 {fmt(next(x for x in D['lead_table'] if x['方法']=='CUSUM')['偵測到的事件比例'])}、前置 {fmt(next(x for x in D['lead_table'] if x['方法']=='CUSUM')['前置時間中位數（天）'], 0)} 天。在不高於 EWMA 假警報率（{fmt(ewm['每 100 鄉鎮週假警報'])}）下，模型 p* = {m_ewm['模型 p*'] if m_ewm else '—'}：偵測 {fmt(m_ewm['模型偵測事件比例']) if m_ewm else '—'}、前置 {fmt(m_ewm['模型前置中位數（天）'], 0) if m_ewm else '—'} 天；EWMA 偵測 {fmt(ewm and next(x for x in D['lead_table'] if x['方法']=='EWMA')['偵測到的事件比例'])}、前置 {fmt(next(x for x in D['lead_table'] if x['方法']=='EWMA')['前置時間中位數（天）'], 0)} 天。模型的價值主要在「提前幾天」，偵測比例與 CUSUM 相近。</p>
  </section>

  <section>
    <div class="section-head"><span class="section-num">05</span><h2>案例</h2><span class="en">Case studies</span></div>
    <div class="grid-2">{cases_html}</div>
    <p class="source">兩個案例都顯示：流行起始前的兩次突破，模型機率在突破前數天已拉高；疫情高峰後閾值遠高於病例數（反應式規則的特性），不再有事件，模型機率也回落。</p>
  </section>

  <section>
    <div class="section-head"><span class="section-num">06</span><h2>應用情境與工作量</h2><span class="en">Operational use</span></div>
    <div class="grid-2">
      <div class="card"><h3>從機率到行動</h3><ol class="steps">
        <li>邊緘節點每日更新各收容所（此處為鄉鎮）的 7 日累計序列，先做通報完整度校正，再由模型輸出 14 天的分位數。</li>
        <li>換算「14 天內突破閾值的機率」；兩級門檻：<b>注意</b>（p* 低，只在儀表板標示）與<b>警示</b>（p* 高，產生查證工作單）。</li>
        <li>警示不自動觸發行動：工作單預填原始通報資料，推送給區級 focal point，依 EWARN 在 24 小時內查證。</li>
        <li>失效防護：若通報完整度異常（例如災後通報中斷）、機率校準偏離（第 4 節校準圖）或 context 分布位移，系統降級為 EWARN/EWMA 規則並強制人工覆核。</li></ol></div>
      <div class="card"><h3>把 p* 換成每週工作量（asof_adj，示例：一個轄區 {D['n_sites_example']} 個站點）</h3>
        <div class="sub">每站每週警示數 × 站數；假警報數 = 警示數 × (1 − PPV)。表中假警報率的分母為無事件的鄉鎮週。</div>
        <div class="tbl-wrap"><table class="tbl compact"><thead><tr><th class="num">p*</th><th class="num">敏感度</th><th class="num">假警報/100 週</th><th class="num">PPV</th><th class="num">轄區每週警示</th><th class="num">其中假警報</th><th class="num">偵測事件比例</th><th class="num">前置中位數</th></tr></thead><tbody>{work_rows}</tbody></table></div>
        <p class="source">建議起點：p* 0.4–0.5 作為「警示」（每週約 {fmt(D['workload'][sw['p'].index(0.4)]['alerts_week_district'] if 0.4 in sw['p'] else None, 1)}–{fmt(D['workload'][i5]['alerts_week_district'], 1)} 張查證單 / {D['n_sites_example']} 站），p* 0.2–0.3 作為「注意」。若防疫單位能提供一次查證成本 C 與漏報損失 L，預設 p* = C / L。</p></div>
    </div>
  </section>

  <section>
    <div class="section-head"><span class="section-num">07</span><h2>限制與下一步</h2><span class="en">Limits &amp; next</span></div>
    <div class="grid-2">
      <div class="card"><h3>限制</h3><ul class="findings readme">
        <li>用的是確定病例，不是情境一的自報症候群；病例數量級與雜訊都不同。</li>
        <li>閾值以起點時的 EWARN 規則定義，事件集中在大流行年（{D['n_events']} 個事件中絕大多數在 2014、2015、2023），平靜年幾乎沒有事件可評估敏感度。</li>
        <li>機率略保守、區間略寬（80% 涵蓋率約 0.87）；可做機率校準再用。</li>
        <li>TimesFM 3.0 為非商業授權且 330M 參數，邊緣端需蒸餾或改用 2.5；本頁只驗證「基礎模型層」的價值。</li></ul></div>
      <div class="card"><h3>下一步</h3><ul class="findings readme">
        <li>機率校準（isotonic）與兩級門檻的工作量表定案。</li>
        <li>同城市鄉鎮的多變量聯合預測、鄰區病例與雨量作為共變數。</li>
        <li>Farrington flexible 基準；每日起點；另一種「群聚起始」事件定義（7 日累計 ≥ 3 且前 14 天為 0）。</li>
        <li>把同一流程套到週層級的急性腹瀉、腸病毒、類流感（RODS），完成情境一四類症候群。</li></ul></div>
    </div>
  </section>
</main>
<footer class="site-footer">資料：疾病管制署登革熱每日確定病例（Internet Archive 保存之官方檔案）。模型：google/timesfm-3.0-pytorch（非商業授權）。子專案 dengue_ewarn；圖表依《疫情資料視覺化指引》v1.1。產生時間 {D['generated_at']}。</footer>
<script>
document.addEventListener('DOMContentLoaded', async () => {{
  const D = await (await fetch('data.json')).json();
  const H = D.hs.map(String);
  EPI.linesChart(document.getElementById('c-wis'), [
    {{label: 'TimesFM（asof_adj）', data: D.wis_by_h.asof_adj.tfm, kind: 'model'}},
    {{label: 'TimesFM（final）', data: D.wis_by_h.final.tfm, kind: 'model'}},
    {{label: 'naive（asof_adj）', data: D.wis_by_h.asof_adj.naive, kind: 'baseline'}},
    {{label: 'naive（final）', data: D.wis_by_h.final.naive, kind: 'baseline'}}], {{labels: H, xLabel: '預測天數 h', yLabel: 'WIS（越低越好）', fmt: v => Number(v).toFixed(1)}});
  const cal = D.calibration;
  EPI.linesChart(document.getElementById('c-cal'), [
    {{label: '實際發生比例（TimesFM asof_adj）', data: cal.map(c => c.observed), kind: 'model'}},
    {{label: '完美校準', data: cal.map(c => c.mid), kind: 'baseline'}}], {{labels: cal.map(c => c.bin), xLabel: '預測機率區間', yLabel: '該區間內實際突破比例', yMin: 0, yMax: 1, fmt: v => Number(v).toFixed(2)}});
  const sw = D.sweep; const P = sw.asof_adj.p.map(String);
  EPI.linesChart(document.getElementById('c-sens'), D.modes.map(m => ({{label: D.mode_label[m].split('（')[0], data: sw[m].sens, kind: m === 'asof' ? 'baseline' : 'model'}})), {{labels: P, xLabel: 'p*', yLabel: '敏感度', yMin: 0, yMax: 1, fmt: v => Number(v).toFixed(2)}});
  const det = Object.fromEntries(D.detectors.map(d => [d['方法'], d['每 100 鄉鎮週假警報']]));
  EPI.linesChart(document.getElementById('c-far'), D.modes.map(m => ({{label: D.mode_label[m].split('（')[0], data: sw[m].far, kind: m === 'asof' ? 'baseline' : 'model'}})), {{labels: P, xLabel: 'p*', yLabel: '假警報 / 100 鄉鎮週', fmt: v => Number(v).toFixed(1), refLines: {{horizontal: [{{at: det.CUSUM, label: 'CUSUM ' + det.CUSUM}}, {{at: det.EWMA, label: 'EWMA ' + det.EWMA}}]}}}});
  const order = ['EWARN 靜態規則', 'EWMA', 'CUSUM', 'TimesFM asof_adj p*=0.3', 'TimesFM asof_adj p*=0.5', 'TimesFM asof_adj p*=0.7', 'TimesFM final p*=0.5'];
  const lt = Object.fromEntries(D.lead_table.map(x => [x['方法'], x]));
  const rows = order.filter(k => lt[k]);
  EPI.hbarChart(document.getElementById('c-lead-det'), rows, rows.map(k => lt[k]['偵測到的事件比例']), {{label: '偵測比例', fmt: v => Number(v).toFixed(2), xMax: 1.15}});
  EPI.hbarChart(document.getElementById('c-lead-med'), rows, rows.map(k => lt[k]['前置時間中位數（天）'] ?? 0), {{label: '前置中位數（天）', fmt: v => Number(v).toFixed(1), xMax: 8}});
}});
</script>
</body></html>"""
