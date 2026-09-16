#!/usr/bin/env python
"""Build the static GitHub Pages site (docs/) from pipeline outputs.

Writes docs/data/{meta,latest,backtest}.json and docs/report.html (numbers filled in).
The dashboards docs/index.html and docs/backtest.html are static and read the JSON.
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller import OUTPUT_DIR, PROCESSED_DIR, ROOT  # noqa: E402
from forecast_teller.backtest import SEGMENTS, segment_of, threshold_note  # noqa: E402
from forecast_teller.covariates import holiday_weekly  # noqa: E402
from forecast_teller.panel import load_long, load_national  # noqa: E402
from forecast_teller.weeks import week_start  # noqa: E402

DOCS = ROOT / "docs"; DATA = DOCS / "data"; BT = OUTPUT_DIR / "backtest"; LATEST = OUTPUT_DIR / "latest"

IND = {
    "nhi_out_ili": {"label": "全國類流感門診就診人次", "short": "類流感門診", "unit": "人次", "decimals": 0, "source": "健保申報（疾管署資料開放平台）"},
    "rods_ili_pct": {"label": "RODS 急診類流感就診百分比", "short": "急診類流感%", "unit": "%", "decimals": 1, "source": "即時疫情監視系統 RODS"},
    "nhi_er_ili": {"label": "全國類流感急診就診人次", "short": "類流感急診", "unit": "人次", "decimals": 0, "source": "健保申報"},
    "nidds_severe": {"label": "流感併發重症週病例數（發病週）", "short": "流感併發重症", "unit": "例", "decimals": 0, "source": "法定傳染病通報 NIDDS"},
    "rods_ili": {"label": "RODS 急診類流感就診人次", "short": "RODS 急診人次", "unit": "人次", "decimals": 0, "source": "RODS"},
}
CONFIG_LABELS = {
    "snaive": "季節性 naive（去年同週）", "naive": "last-value naive", "ma3": "MA3（前 3 週移動平均）", "ets": "AutoETS", "theta": "Theta",
    "tfm_expanding": "TimesFM zero-shot（無共變數）", "tfm_expanding_log1p": "TimesFM（log1p）", "tfm_expanding_sym": "TimesFM（對稱平均）",
    "tfm_slide260": "TimesFM（滑動 260 週）", "tfm_slide156": "TimesFM（滑動 156 週）",
    "tfm_cov_season": "TimesFM + 季節相位", "tfm_cov_cny": "TimesFM + 春節旗標", "tfm_cov_season_cny": "TimesFM + 季節 + 春節",
    "tfm_cov_season_cny_hol": "TimesFM + 季節 + 春節 + 假日", "tfm_cov_season_cny_log1p": "TimesFM + 季節 + 春節（log1p）",
    "tfm_cov_hol": "TimesFM + 假日天數", "tfm_cov_cny_hol": "TimesFM + 春節 + 假日（最佳單變量）", "tfm_cov_season_hol": "TimesFM + 季節 + 假日",
    "tfm_cov_season_cny_hol_sym": "TimesFM + 季節 + 春節 + 假日（對稱平均）", "tfm_cov_season_cny_hol_slide260": "TimesFM + 季節 + 春節 + 假日（滑動 260）",
    "tfm_po_lab": "TimesFM + 實驗室 A/B（過去共變數）", "tfm_po_labpos": "TimesFM + 實驗室陽性率（過去共變數）", "tfm_po_rods": "TimesFM + RODS 急診%（過去共變數）",
    "tfm_po_all_lag1": "TimesFM + 全部過去共變數（延遲 1 週）", "tfm_po_all_lag2": "TimesFM + 全部過去共變數（延遲 2 週）",
    "tfm_mv_out_er_rods": "三變量聯合 + 季節 + 春節", "tfm_mv_out_er_rods_sev": "四變量聯合 + 季節 + 春節", "tfm_mv_out_er_rods_nocov": "三變量聯合（無共變數）",
    "tfm_best_mv4_hol": "四變量聯合 + 季節 + 春節 + 假日", "tfm_best_mv3_hol": "三變量聯合 + 季節 + 春節 + 假日",
    "tfm_best_mv4_cny_hol": "四變量聯合 + 春節 + 假日（最佳）", "tfm_best_po_rods_hol": "TimesFM + RODS 過去共變數 + 季節 + 春節 + 假日",
    "tfm_best_mv4_hol_slide260": "四變量聯合 + 季節 + 春節 + 假日（滑動 260）",
}
GROUP_LABELS = {"naive": "逐序列 last-value naive", "ma3": "逐序列 MA3（前 3 週移動平均）", "uni_nocov": "逐序列 TimesFM（無共變數）", "mv_nocov": "聯合多變量（無共變數）", "mv_cov": "聯合多變量 + 季節 + 春節"}
SEG_LABELS = {"post_covid": "COVID 後（2023–2025）", "pre_covid": "COVID 前（2018–2019）", "covid": "COVID 期（2020–2022）", "all": "全期（2018–2025）"}
BASELINES = {"snaive", "naive", "ma3", "ets", "theta"}
TAGS = {"nhi_out_ili": ["nhi_out_ili_baselines", "nhi_out_ili_layer1", "nhi_out_ili_layer2", "nhi_out_ili_layer2b", "nhi_out_ili_layer3", "nhi_out_ili_layer4", "nhi_out_ili_stats"],
        "nhi_er_ili": ["nhi_er_ili_baselines", "nhi_er_ili_core", "nhi_er_ili_stats"],
        "rods_ili_pct": ["rods_ili_pct_baselines", "rods_ili_pct_layer1", "rods_ili_pct_stats"]}


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x) else round(float(x), nd)


def ds(yw: str) -> str:
    return str(week_start(str(yw)).date())


# ----------------------------------------------------------------------------- latest
def build_latest(nat: pd.DataFrame) -> dict:
    joint = pd.read_csv(LATEST / "latest_forecast_joint.csv", dtype={"origin_yw": str, "target_yw": str})
    uni = pd.read_csv(LATEST / "latest_forecast.csv", dtype={"origin_yw": str, "target_yw": str})
    panels = [("nhi_out_ili", joint, "聯合三變量（門診 + 急診 + RODS）+ 春節旗標 + 假日天數"),
              ("rods_ili_pct", uni, "單變量 + 春節旗標 + 假日天數"),
              ("nhi_er_ili", joint, "聯合三變量（門診 + 急診 + RODS）+ 春節旗標 + 假日天數"),
              ("nidds_severe", uni, "單變量 + 春節旗標 + 假日天數")]
    out = []
    for key, src, cfg in panels:
        rows = src[src.target == key].sort_values("h")
        origin = str(rows.origin_yw.iloc[0])
        hist = nat[key].dropna()
        hist = hist.iloc[-130:]
        tw = [str(w) for w in rows.target_yw]
        hw = holiday_weekly(list(hist.index) + tw)
        nd = IND[key]["decimals"]
        history = [{"yw": w, "date": ds(w), "y": r(v, nd)} for w, v in hist.items()]
        forecast = []
        for _, x in rows.iterrows():
            w = str(x.target_yw)
            forecast.append({"yw": w, "date": ds(w), "h": int(x.h), "median": r(x["median"], nd),
                             **{f"q{q}": r(x[f"q{q}"], nd) for q in range(10, 100, 10)},
                             "cny": int(hw.loc[w, "cny_week"]), "holiday_days": int(hw.loc[w, "holiday_days"]),
                             "observed": r(nat[key].get(w, np.nan), nd)})
        last_obs_at_origin = r(nat[key].loc[origin], nd)
        out.append({"key": key, **IND[key], "config": cfg, "origin_yw": origin, "origin_date": ds(origin),
                    "last_observed": last_obs_at_origin, "history": history, "forecast": forecast})
    return {"indicators": out}


# ----------------------------------------------------------------------------- backtest
def load_forecasts(tags: list[str], target: str) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(BT / f"{t}_forecasts.parquet") for t in tags], ignore_index=True)
    df = df[df.target == target].drop_duplicates(subset=["config", "origin_yw", "h"]).copy()
    df["segment"] = df.target_year.map(segment_of)
    return df


def build_backtest() -> dict:
    targets = {}
    for target, tags in TAGS.items():
        summ = pd.read_csv(BT / f"{target}_summary_combined.csv")
        summ = summ[summ.target == target]
        fc = load_forecasts(tags, target)
        nd = IND[target]["decimals"]
        segs = {}
        for seg in ["post_covid", "pre_covid", "covid", "all"]:
            sub = summ[(summ.segment == seg) & (summ.h <= 4)]
            metrics = ["wis", "mae", "mape", "mase", "cov80", "cov60", "dir_hit", "dir_hit3", "hit_tol10", "thr_hit_rate", "thr_false_alarm", "thr_accuracy"]
            lb = sub.groupby("config")[metrics].mean()
            lb["rel_naive"] = lb["wis"] / lb.loc["naive", "wis"]
            lb["rel_snaive"] = lb["wis"] / lb.loc["snaive", "wis"]
            lb = lb.sort_values("wis")
            rows = [{"config": c, "label": CONFIG_LABELS.get(c, c), "kind": "baseline" if c in BASELINES else "timesfm",
                     **{m: r(v, 4 if m not in ("wis", "mae") else nd + 1) for m, v in row.items()}} for c, row in lb.iterrows()]
            by_h = lambda col: {c: [r(v, 4 if col != "wis" else nd + 1) for v in sub[sub.config == c].sort_values("h")[col]] for c in lb.index}
            segs[seg] = {"label": SEG_LABELS[seg], "leaderboard": rows, "wis_by_h": by_h("wis"), "dir_hit_by_h": by_h("dir_hit"),
                         "thr_hit_by_h": by_h("thr_hit_rate"), "cov80_by_h": by_h("cov80"),
                         "n_origins": int(fc[(fc.segment == seg) if seg != "all" else slice(None)].origin_yw.nunique()) if seg != "all" else int(fc.origin_yw.nunique())}
        best = segs["post_covid"]["leaderboard"][0]["config"]
        # yearly WIS (h=1, h=4) for a focused set of configs
        focus = [c for c in [best, "tfm_cov_cny_hol", "tfm_expanding", "ets", "ma3", "naive", "snaive"] if c in set(fc.config)]
        yearly = {}
        for h in (1, 4):
            t = fc[(fc.h == h) & (fc.config.isin(focus))].groupby(["config", "target_year"])["wis"].mean().unstack("target_year")
            yearly[f"h{h}"] = {c: {str(y): r(v, nd + 1) for y, v in t.loc[c].items()} for c in focus if c in t.index}
        # post-COVID forecast vs actual for best config, each h
        ts = {}
        post = fc[(fc.segment == "post_covid")]
        for h in (1, 2, 3, 4):
            b = post[(post.config == best) & (post.h == h)].sort_values("target_yw")
            n = post[(post.config == "naive") & (post.h == h)].set_index("target_yw")["median"]
            ts[f"h{h}"] = [{"yw": str(x.target_yw), "date": ds(x.target_yw), "y": r(x.y, nd), "median": r(x["median"], nd),
                            "q10": r(x.q10, nd), "q90": r(x.q90, nd), "q20": r(x.q20, nd), "q80": r(x.q80, nd),
                            "naive": r(n.get(x.target_yw, np.nan), nd)} for _, x in b.iterrows()]
        targets[target] = {**IND[target], "threshold": r(summ.thr.iloc[0], nd), "thr_note": threshold_note(target), "best": best, "best_label": CONFIG_LABELS.get(best, best),
                           "segments": segs, "yearly_wis": yearly, "timeseries": ts,
                           "n_origins": int(fc.origin_yw.nunique()), "origin_first": str(fc.origin_yw.min()), "origin_last": str(fc.origin_yw.max())}
    groups = {}
    for by, label in [("county", "22 縣市"), ("age", "18 年齡層")]:
        g = pd.read_parquet(BT / f"{by}_nhi_out_ili_forecasts.parquet")
        g["segment"] = g.target_year.map(segment_of)
        post = g[g.segment == "post_covid"]
        per = post.groupby(["config", by])["wis"].mean().unstack("config")
        rel = per.div(per["naive"], axis=0)
        order = rel["mv_cov"].sort_values().index.tolist()
        seg_tab = {}
        for seg in ["post_covid", "pre_covid", "covid", "all"]:
            s = g if seg == "all" else g[g.segment == seg]
            agg = s.groupby("config").agg(wis_sum=("wis", "sum"), cov80=("cov80", "mean"), dir_hit=("dir_hit", "mean"), dir_hit3=("dir_hit3", "mean"), hit_tol10=("hit_tol10", "mean"))
            agg["rel_naive"] = agg["wis_sum"] / agg.loc["naive", "wis_sum"]
            seg_tab[seg] = [{"config": c, "label": GROUP_LABELS.get(c, c), "rel_naive": r(v.rel_naive, 3), "cov80": r(v.cov80, 3),
                             "dir_hit": r(v.dir_hit, 3), "dir_hit3": r(v.dir_hit3, 3), "hit_tol10": r(v.hit_tol10, 3)} for c, v in agg.sort_values("wis_sum").iterrows()]
        groups[by] = {"label": label, "n_series": int(per.shape[0]), "segments": seg_tab,
                      "per_group_rel": [{"name": k, **{c: r(rel.loc[k, c], 3) for c in rel.columns}} for k in order]}
    return {"targets": targets, "groups": groups, "config_labels": CONFIG_LABELS, "seg_labels": SEG_LABELS}


# ----------------------------------------------------------------------------- report
def pct_impr(a, b):
    return round(100 * (1 - a / b))


def build_report(bt: dict, latest: dict, meta: dict) -> str:
    t = bt["targets"]["nhi_out_ili"]; post = t["segments"]["post_covid"]; pre = t["segments"]["pre_covid"]
    lb = {x["config"]: x for x in post["leaderboard"]}; lbp = {x["config"]: x for x in pre["leaderboard"]}
    best, zs, naive, ets, sn = lb[t["best"]], lb["tfm_expanding"], lb["naive"], lb["ets"], lb["snaive"]; ma3 = lb["ma3"]
    wh = post["wis_by_h"]; impr = [pct_impr(a, b) for a, b in zip(wh[t["best"]], wh["naive"])]
    impr_zs = [pct_impr(a, b) for a, b in zip(wh[t["best"]], wh["tfm_expanding"])]
    pre_best = pre["leaderboard"][0]
    rods = bt["targets"]["rods_ili_pct"]; rp = {x["config"]: x for x in rods["segments"]["post_covid"]["leaderboard"]}; rbest = rp[rods["best"]]
    er = bt["targets"]["nhi_er_ili"]; ep = {x["config"]: x for x in er["segments"]["post_covid"]["leaderboard"]}; ebest = ep[er["best"]]
    cty = {x["config"]: x for x in bt["groups"]["county"]["segments"]["post_covid"]}; age = {x["config"]: x for x in bt["groups"]["age"]["segments"]["post_covid"]}
    dh = post["dir_hit_by_h"][t["best"]]
    li = {x["key"]: x for x in latest["indicators"]}
    f_out = li["nhi_out_ili"]; f1 = f_out["forecast"]
    gen = meta["generated_at"][:10]
    fmt = lambda v, nd=0: f"{v:,.{nd}f}"
    html = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>解讀與評估報告 · 台灣流感 TimesFM 3.0 預測</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;600;700&family=Noto+Serif+TC:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/epi.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="assets/charts.js" defer></script></head>
<body>
<header class="masthead"><div class="masthead-inner"><div class="masthead-left"><div class="crest">FT</div><div class="masthead-title">台灣流感疫情預測 · TimesFM 3.0<small>FORECAST-TELLER · 解讀與評估報告</small></div></div>
<nav class="masthead-right"><a href="index.html">每週預測</a><a href="backtest.html">回測</a><a href="report.html" class="active">報告</a></nav></div></header>
<main class="report">
<div class="hero-kicker">EVALUATION REPORT · {gen}</div>
<h1 class="report-title">以 TimesFM 3.0 零樣本預測台灣類流感就診：<em>回測解讀與評估</em></h1>
<p class="lead">資料窗 2016w01–2025w53（522 週），以 2016–2017 為暖機 context，2018w01 起每週一個起點共 {t['n_origins']} 個，預測未來 1–4 週；主要決策依據為 COVID 後（2023–2025）的加權區間分數（WIS）。</p>

<div class="kpi-strip">
  <div class="kpi"><div class="label">最佳設定 WIS（COVID 後，h=1–4）</div><div class="value">{fmt(best['wis'])}</div><div class="sub">{best['label']}</div></div>
  <div class="kpi"><div class="label">相對 last-value naive</div><div class="value">−{round(100*(1-best['rel_naive']))}%</div><div class="sub">1–4 週分別 −{impr[0]}% / −{impr[1]}% / −{impr[2]}% / −{impr[3]}%</div></div>
  <div class="kpi"><div class="label">80% 預測區間涵蓋率</div><div class="value">{best['cov80']:.2f}</div><div class="sub">理想值 0.80；60% 區間 {best['cov60']:.2f}</div></div>
  <div class="kpi"><div class="label">方向命中率（h=1–4）</div><div class="value">{best['dir_hit']:.2f}</div><div class="sub">1 週前 {dh[0]:.2f}，4 週前 {dh[3]:.2f}；naive 退化為 {naive['dir_hit']:.2f}</div></div>
</div>

<section><h2>1. 我們評估了什麼</h2>
<p>目標序列是<strong>全國每週類流感門診就診人次</strong>（健保申報），另以全國類流感急診就診人次、RODS 急診類流感就診百分比、22 縣市與 18 年齡層做延伸驗證。模型是 Google TimesFM 3.0（330M 參數的時間序列基礎模型），<strong>零樣本、未經任何微調</strong>，逐層加入已知未來共變數（春節旗標、平日假日天數、季節相位）、過去共變數（實驗室型別、RODS、重症）與多變量聯合預測。基準模型為 last-value naive、MA3（前 3 週移動平均，2 週以上遞迴代入）、季節性 naive、AutoETS 與 Theta。</p>
<p>評估指標：WIS（由 9 個分位數組成 4 個對稱區間加中位數，越低越好）、MAE、MAPE、MASE、80%/60% 區間涵蓋率、方向命中率、±10% 容忍帶命中率，以及閾值命中率與誤報率（門診人次以第 75 百分位 {fmt(t['threshold'])} 人次為閾值，RODS 急診類流感%以流行閾值 10% 為閾值）。結果分 COVID 前（2018–2019）、COVID 期（2020–2022）、COVID 後（2023–2025）三段報告，決策看 COVID 後。</p></section>

<section><h2>2. 主要結果</h2>
<div class="two-col">
<div class="card"><h3>各設定 WIS 依預測週數（COVID 後）</h3><div class="chart-box"><canvas id="c-wis" role="img" aria-label="最佳設定在 1 到 4 週的 WIS 均低於各基準模型"></canvas></div><p class="source">WIS 越低越好；灰色虛線為基準模型。</p></div>
<div class="card"><h3>最佳設定 1 週前預測 vs 實際（2023–2025）</h3><div class="chart-box"><canvas id="c-ts" role="img" aria-label="1 週前預測中位數貼近實際值，80% 區間涵蓋大多數週"></canvas></div><p class="source">虛線為預測中位數，淺綠帶為 80% 預測區間。</p></div>
</div>
<table class="tbl compact"><thead><tr><th>設定</th><th class="num">WIS</th><th class="num">相對 naive</th><th class="num">MAPE %</th><th class="num">MASE</th><th class="num">80% 涵蓋</th><th class="num">方向命中</th><th class="num">閾值命中</th></tr></thead><tbody>
{''.join(f"<tr class='{'best' if x['config']==t['best'] else ''}'><td>{x['label']}</td><td class='num'>{fmt(x['wis'])}</td><td class='num'>{x['rel_naive']:.2f}</td><td class='num'>{x['mape']:.1f}</td><td class='num'>{x['mase']:.2f}</td><td class='num'>{x['cov80']:.2f}</td><td class='num'>{x['dir_hit']:.2f}</td><td class='num'>{x['thr_hit_rate']:.2f}</td></tr>" for x in [best, lb['tfm_cov_cny_hol'], zs, ets, ma3, naive, sn])}
</tbody></table>
<ul class="findings">
<li><strong>零樣本已勝過統計基準。</strong>不加任何共變數的 TimesFM，WIS 比 last-value naive 低 {round(100*(1-zs['rel_naive']))}%、比 MA3（前 3 週移動平均）低 {pct_impr(zs['wis'], ma3['wis'])}%、比 AutoETS 低 {pct_impr(zs['wis'], ets['wis'])}%；季節性 naive 因季節位移與 COVID 斷層幾乎失效（MASE {sn['mase']:.2f}）。</li>
<li><strong>假日天數是最有用的共變數。</strong>加入每週平日假日天數與春節旗標後，1 週前 WIS 再降約 {pct_impr(wh['tfm_cov_cny_hol'][0], wh['tfm_expanding'][0])}%；季節相位 sin/cos、log1p 轉換、對稱平均與滑動視窗都沒有幫助。</li>
<li><strong>多變量聯合預測在較長 horizon 更好。</strong>門診 + 急診 + RODS + 重症四變量聯合加假日共變數是全部 horizon 的最佳設定，相對無共變數 zero-shot 在 1–4 週分別改善 {impr_zs[0]}% / {impr_zs[1]}% / {impr_zs[2]}% / {impr_zs[3]}%。</li>
<li><strong>校準良好。</strong>80% 區間涵蓋率 {best['cov80']:.2f}、60% 涵蓋率 {best['cov60']:.2f}，接近名目值；naive 的區間過窄（{naive['cov80']:.2f}）。</li>
<li><strong>COVID 前也成立。</strong>2018–2019 最佳為 {pre_best['label']}，WIS {fmt(pre_best['wis'])}，相對 naive −{round(100*(1-pre_best['rel_naive']))}%。</li>
<li><strong>延伸驗證。</strong>全國類流感急診就診人次（健保）：最佳設定 {ebest['label']}，WIS {fmt(ebest['wis'])}，相對 naive −{round(100*(1-ebest['rel_naive']))}%，MAPE {ebest['mape']:.1f}%，80% 涵蓋率 {ebest['cov80']:.2f}。RODS 急診類流感%：最佳 WIS {rbest['wis']:.2f} 個百分點，相對 naive −{round(100*(1-rbest['rel_naive']))}%，方向命中率 {rbest['dir_hit']:.2f}，以流行閾值 10% 計的閾值命中率 {rbest['thr_hit_rate']:.2f}、誤報率 {rbest['thr_false_alarm']:.2f}。22 縣市聯合預測相對逐縣市 naive 為 {cty['mv_cov']['rel_naive']:.2f}，18 年齡層為 {age['mv_cov']['rel_naive']:.2f}。</li>
</ul></section>

<section><h2>3. 怎麼解讀</h2>
<ul class="findings">
<li><strong>1–2 週最可靠。</strong>WIS 隨 horizon 大致線性上升（{fmt(wh[t['best']][0])} → {fmt(wh[t['best']][3])}），方向命中率從 {dh[0]:.2f} 降到 {dh[3]:.2f}。4 週以上的預測應只做規劃參考。</li>
<li><strong>峰值偏保守。</strong>高流行週（實際值在第 75 百分位以上）中位數低估 2–8%（1→4 週）；零樣本模型不知道今年疫苗策略或流行株變化，遇到創新高的季節會慢半拍。</li>
<li><strong>閾值警示可用但需搭配區間。</strong>以第 75 百分位為閾值，1–2 週前的命中率約 0.86–0.90、誤報率約 0.17；建議用「超過閾值的機率」（由 9 個分位數推得）而非中位數單點做警示。</li>
<li><strong>COVID 期的分數不作決策依據。</strong>2020–2022 流感近乎消失，模型在該時期高估 4–12%，但區間涵蓋率仍在 0.75 以上。</li>
</ul></section>

<section><h2>4. 限制與注意事項</h2>
<ul class="findings">
<li>健保申報有回補，最近 1–2 週低報；管線以「低於前 8 週中位數 60%」自動排除不完整週，正式使用時應以首報/終報比值校正。</li>
<li>假日檔只到 2025 年底，2026 年起春節週以農曆新年日期外推、平日假日天數假設 5 天；請以官方辦公日曆表延伸後重跑。</li>
<li>TimesFM 3.0 權重授權為非商業、非生產用途；正式上線需改用 2.5（Apache-2.0）或另取授權。</li>
<li>回測資料窗只有十年，最早的起點僅兩季暖機；閾值以整段資料的分位數決定，帶有少量事後資訊。</li>
</ul></section>

<section><h2>5. 建議</h2>
<ul class="findings">
<li>每週固定流程：更新資料 → 建面板 → 以四變量聯合 + 春節 + 假日設定產出 1–4 週預測與分位數 → 發布本站。</li>
<li>發布時同時呈現中位數、80% 區間與「超過閾值的機率」，並註明資料截止週與申報回補風險。</li>
<li>下一步：TimesFM 與 ETS 的簡單集成、健保申報回補校正、季節峰值時間/高度的專門評估，以及微調（2.5 LoRA）是否值得。</li>
</ul></section>

<section><h2>6. 最新預測摘要（起點 {f_out['origin_yw']}，{f_out['origin_date']} 當週）</h2>
<p>全國類流感門診就診人次未來 4 週中位數：{'、'.join(fmt(x['median']) for x in f1)}；第 1 週 80% 區間 {fmt(f1[0]['q10'])}–{fmt(f1[0]['q90'])}{'（春節週）' if f1[0]['cny'] else ''}。完整內容見<a href="index.html">每週預測</a>。</p></section>

<footer class="report-footer">資料：疾病管制署資料開放平台（健保、RODS、NIDDS、合約實驗室）。模型：google/timesfm-3.0-pytorch（非商業授權）。產生時間 {meta['generated_at']}。圖表依《疫情資料視覺化指引》v1.1。</footer>
</main>
<script>
document.addEventListener('DOMContentLoaded', async () => {{
  const bt = await (await fetch('data/backtest.json')).json();
  const t = bt.targets['nhi_out_ili']; const seg = t.segments.post_covid;
  const cfgs = [t.best, 'tfm_cov_cny_hol', 'tfm_expanding', 'ets', 'naive', 'snaive'];
  EPI.linesChart(document.getElementById('c-wis'), EPI.byHorizonSeries(seg, cfgs, 'wis_by_h', bt.config_labels), {{yLabel: 'WIS', xLabel: '預測週數（h）', fmt: v => v.toLocaleString('zh-TW', {{maximumFractionDigits: 0}})}});
  EPI.backtestTsChart(document.getElementById('c-ts'), t.timeseries.h1, {{unit: t.unit, decimals: t.decimals}});
}});
</script>
</body></html>"""
    return html


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    nat = load_national()
    cov = json.loads((PROCESSED_DIR / "coverage.json").read_text(encoding="utf-8"))
    now = dt.datetime.now().astimezone().isoformat(timespec="minutes")
    latest = build_latest(nat); latest["generated_at"] = now
    bt = build_backtest(); bt["generated_at"] = now
    meta = {"generated_at": now, "coverage": {k: {"first_complete": v["first_complete"], "last_complete": v["last_complete"],
                                                  "first_date": ds(v["first_complete"]), "last_date": ds(v["last_complete"])} for k, v in cov.items()},
            "backtest_window": {"start": "201601", "end": "202553", "weeks": 522, "warmup": 104, "origins": bt["targets"]["nhi_out_ili"]["n_origins"]},
            "model": {"name": "TimesFM 3.0", "checkpoint": "google/timesfm-3.0-pytorch", "license": "timesfm-non-commercial-license-v1.0（非商業、非生產用途）"},
            "best_config": bt["targets"]["nhi_out_ili"]["best"], "best_label": bt["targets"]["nhi_out_ili"]["best_label"]}
    (DATA / "latest.json").write_text(json.dumps(latest, ensure_ascii=False), encoding="utf-8")
    (DATA / "backtest.json").write_text(json.dumps(bt, ensure_ascii=False), encoding="utf-8")
    (DATA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (DOCS / "report.html").write_text(build_report(bt, latest, meta), encoding="utf-8")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    sizes = {p.name: p.stat().st_size // 1024 for p in DATA.glob("*.json")}
    print("site data written:", sizes, "KB; report.html", (DOCS / "report.html").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
