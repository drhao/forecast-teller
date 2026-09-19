"""Build the enterovirus pages of the GitHub Pages site (docs/ev/): data JSON, rule-based narrative, report."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from forecast_teller.backtest import SEGMENTS, segment_of
from forecast_teller.covariates import holiday_weekly
from forecast_teller.weeks import week_start

from . import DOCS_DIR, OUTPUT_DIR, PROCESSED_DIR
from .covariates import school_calendar
from .panel import load_national
from .thresholds import RULE_TEXT, epidemic_flags, load_thresholds, periods, threshold_for

DATA = DOCS_DIR / "data"; BT = OUTPUT_DIR / "backtest"; LATEST = OUTPUT_DIR / "latest"
PRIMARY = "ev_oe"
IND = {
    "ev_oe": {"label": "全國腸病毒門急診就診人次（健保門診 + RODS 急診）", "short": "門急診合計", "unit": "人次", "decimals": 0,
              "source": "健保申報門診腸病毒就診人次（疾管署資料開放平台）+ RODS 急診腸病毒就診人次（急診部分暫以 RODS 代替）"},
    "ev_out": {"label": "全國腸病毒健保門診就診人次", "short": "門診", "unit": "人次", "decimals": 0, "source": "健保申報（疾管署資料開放平台）"},
    "ev_rods": {"label": "RODS 急診腸病毒就診人次", "short": "RODS 急診", "unit": "人次", "decimals": 0, "source": "即時疫情監視及預警系統 RODS（資料開放平台）"},
    "ev_rods_pct": {"label": "RODS 急診腸病毒就診百分比", "short": "急診腸病毒%", "unit": "%", "decimals": 2,
                    "source": "RODS 腸病毒人次 ÷ RODS 急診總人次（分母來自流感面板的 RODS_RS.csv）"},
}
CONFIG_LABELS = {
    "snaive": "季節性 naive（去年同週）", "naive": "last-value naive", "ma3": "MA3（前 3 週移動平均）", "ets": "AutoETS", "theta": "Theta",
    "tfm_expanding": "TimesFM zero-shot（無共變數）", "tfm_expanding_log1p": "TimesFM（log1p）", "tfm_expanding_sym": "TimesFM（對稱平均）",
    "tfm_slide156": "TimesFM（滑動 156 週）", "tfm_slide260": "TimesFM（滑動 260 週）",
    "tfm_cov_school": "TimesFM + 學校行事曆", "tfm_cov_hol": "TimesFM + 假日天數", "tfm_cov_cny_hol": "TimesFM + 春節 + 假日",
    "tfm_cov_school_hol": "TimesFM + 學校行事曆 + 假日", "tfm_cov_school_hol_log1p": "TimesFM + 學校行事曆 + 假日（log1p）",
    "tfm_cov_season2": "TimesFM + 雙諧波季節相位", "tfm_cov_season2_school_hol": "TimesFM + 雙諧波季節 + 學校行事曆 + 假日",
    "tfm_cov_season_school_hol": "TimesFM + 單諧波季節 + 學校行事曆 + 假日",
    "tfm_mv_rods": "二變量聯合（+ RODS）+ 學校行事曆 + 假日", "tfm_mv_rods_inp": "三變量聯合（+ RODS + 住院）+ 學校行事曆 + 假日",
    "tfm_mv_rods_inp_log1p": "三變量聯合（+ RODS + 住院）+ 學校行事曆 + 假日（log1p）",
    "tfm_po_rods": "TimesFM + RODS（過去共變數）+ 學校行事曆 + 假日", "tfm_po_rods_inp": "TimesFM + RODS + 住院（過去共變數）+ 學校行事曆 + 假日",
    "tfm_mv_all": "多變量聯合（門診/RODS/住院）+ 學校行事曆 + 假日",
    "tfm_mv_out_rods": "三變量聯合（門急診合計 + 門診 + RODS）+ 學校行事曆 + 假日",
    "tfm_mv_out_rods_inp": "四變量聯合（門急診合計 + 門診 + RODS + 住院）+ 學校行事曆 + 假日",
    "tfm_mv_out_rods_inp_log1p": "四變量聯合（門急診合計 + 門診 + RODS + 住院）+ 學校行事曆 + 假日（log1p）",
}
SEG_LABELS = {"post_covid": "COVID 後（2023–2025）", "pre_covid": "COVID 前（2018–2019）", "covid": "COVID 期（2020–2022）", "all": "全期（2018–2025）"}
BASELINES = {"snaive", "naive", "ma3", "ets", "theta"}
TAGS = {"ev_oe": ["baselines_ev_oe", "core_ev_oe", "stats_ev_oe"],
        "ev_out": ["baselines_ev_out", "layer1_ev_out", "layer2_ev_out", "layer3_ev_out", "stats_ev_out"]}
# 疾管署新聞稿公布的「門急診就診人次」（首報值），用來校驗「門診 + RODS」合成指標
OFFICIAL = {"201617": 11174, "202317": 10230, "202318": 11252, "202335": 12366, "202336": 12163, "202339": 12463, "202340": 12643,
            "202418": 16564, "202419": 17895, "202449": 22279, "202450": 17508, "202546": 11223, "202620": 5153, "202621": 5573,
            "202635": 8641, "202636": 10203}


def r(x, nd=3):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def ds(yw: str) -> str:
    return str(week_start(str(yw)).date())


# ----------------------------------------------------------------------------- latest
def build_latest(nat: pd.DataFrame, config_labels: dict[str, str]) -> dict:
    joint = pd.read_csv(LATEST / "latest_forecast_joint.csv", dtype={"origin_yw": str, "target_yw": str})
    uni = pd.read_csv(LATEST / "latest_forecast.csv", dtype={"origin_yw": str, "target_yw": str})
    thr_tab = load_thresholds()
    out = []
    for key in ["ev_oe", "ev_out", "ev_rods", "ev_rods_pct"]:
        src = joint if key in set(joint.target) else uni
        rows = src[src.target == key].sort_values("h")
        if rows.empty:
            continue
        origin = str(rows.origin_yw.iloc[0])
        hist = nat[key].dropna().loc[:origin].iloc[-130:]
        tw = [str(w) for w in rows.target_yw]
        nd = IND[key]["decimals"]
        history = [{"yw": w, "date": ds(w), "y": r(v, nd)} for w, v in hist.items()]
        forecast = []
        for _, x in rows.iterrows():
            w = str(x.target_yw)
            forecast.append({"yw": w, "date": ds(w), "h": int(x.h), "median": r(x["median"], nd),
                             **{f"q{q}": r(x[f"q{q}"], nd) for q in range(10, 100, 10)},
                             "cny": int(x.cny), "holiday_days": int(x.holiday_days), "summer_break": int(x.summer_break),
                             "winter_break": int(x.winter_break), "reopen": int(x.reopen),
                             "observed": r(nat[key].get(w, np.nan), nd), "threshold": threshold_for(w, thr_tab) if key == PRIMARY else None})
        extra = {}
        if key == PRIMARY:
            thr = threshold_for(tw[0], thr_tab)
            extra = {"threshold": thr, "threshold_label": f"流行閾值 {thr:,.0f} 人次（{tw[0][:4]} 年）"}
        out.append({"key": key, **IND[key], "config": config_labels.get(key, ""), "mode": str(rows["mode"].iloc[0]), "origin_yw": origin,
                    "origin_date": ds(origin), "last_observed": r(nat[key].loc[origin], nd), "history": history, "forecast": forecast, **extra})
    # epidemic-period status from the rule
    oe = nat[PRIMARY].dropna()
    fl = epidemic_flags(oe)
    per = periods(fl)
    status = {"in_period": bool(fl["in_period"].iloc[-1]), "week": str(oe.index[-1]), "threshold": float(fl["thr"].iloc[-1]),
              "periods": [{"start": a, "end": b} for a, b in per[-6:]], "rule": RULE_TEXT,
              "thresholds_by_year": {str(y): float(v) for y, v in load_thresholds()["threshold"].items()}}
    return {"indicators": out, "status": status}


# ----------------------------------------------------------------------------- narrative
def prob_ge(f: dict, thr: float) -> float:
    qs = [f[f"q{k}"] for k in range(10, 100, 10)]; lv = [k / 100 for k in range(10, 100, 10)]
    if thr <= qs[0]:
        return 0.95
    if thr >= qs[-1]:
        return 0.05
    for i in range(8):
        if qs[i] <= thr <= qs[i + 1]:
            frac = (thr - qs[i]) / (qs[i + 1] - qs[i]) if qs[i + 1] > qs[i] else 0.0
            return 1 - (lv[i] + frac * (lv[i + 1] - lv[i]))
    return 0.5


def pct_txt(p: float) -> str:
    return f"約 {int(round(p * 20) * 5)}%"


def trend_word(pct: float) -> str:
    if pct >= 15: return "快速上升"
    if pct >= 5: return "上升"
    if pct > -5: return "大致持平"
    if pct > -15: return "下降"
    return "快速下降"


def consecutive(series: pd.Series, cond) -> int:
    n = 0
    for v in series[::-1]:
        if cond(v):
            n += 1
        else:
            break
    return n


def build_narrative(nat: pd.DataFrame, latest: dict, calib: dict) -> dict:
    li = {x["key"]: x for x in latest["indicators"]}
    oe, out, rods = li["ev_oe"], li["ev_out"], li["ev_rods"]
    st = latest["status"]
    origin = oe["origin_yw"]; o_date = oe["origin_date"]; thr = st["threshold"]
    s_oe = nat["ev_oe"].dropna().loc[:origin]; s_out = nat["ev_out"].dropna().loc[:origin]
    s_rods = nat["ev_rods"].dropna().loc[:origin]; s_pct = nat["ev_rods_pct"].dropna().loc[:origin]; s_inp = nat["ev_inp"].dropna().loc[:origin]
    yr = int(origin[:4])
    fmt0 = lambda v: f"{v:,.0f}"
    cur, fut, cav = [], [], []

    # --- current: composite indicator vs threshold
    wow = 100 * (s_oe.iloc[-1] / s_oe.iloc[-2] - 1); w3 = 100 * (s_oe.iloc[-1] / s_oe.iloc[-4] - 1)
    rising = consecutive(s_oe.diff().dropna(), lambda d: d > 0); falling = consecutive(s_oe.diff().dropna(), lambda d: d < 0)
    yrs = pd.Series(nat.index.str[:4].astype(int), index=nat.index)
    same = nat[(nat.index.str[4:] == origin[4:]) & yrs.between(yr - 3, yr - 1)]["ev_oe"].dropna()
    hist = s_oe.loc["201601":]; pct_rank = 100 * (hist < s_oe.iloc[-1]).mean()
    fl_all = epidemic_flags(s_oe)
    above = consecutive(fl_all["in_period"], lambda v: bool(v)); below = consecutive(fl_all["in_period"], lambda v: not v)
    above_thr = consecutive(s_oe, lambda v: v >= thr)
    txt = (f"全國腸病毒門急診就診人次（門診 + RODS 急診）{origin}（{o_date} 起）為 {fmt0(s_oe.iloc[-1])} 人次，"
           f"較前一週{trend_word(wow)}（{wow:+.1f}%），較 3 週前 {w3:+.0f}%")
    if rising >= 2: txt += f"，已連續 {rising} 週上升"
    elif falling >= 2: txt += f"，已連續 {falling} 週下降"
    if len(same):
        txt += f"；為前三年同一週平均（{fmt0(same.mean())}）的 {s_oe.iloc[-1] / same.mean():.1f} 倍，在 2016 年以來所有週中位於第 {pct_rank:.0f} 百分位"
    txt += f"。{yr} 年流行閾值為 {fmt0(thr)} 人次，"
    if st["in_period"]:
        txt += f"目前處於流行期第 {above} 週（依規則：{'已連續 %d 週高於閾值' % above_thr if above_thr else '上週低於閾值 1 週，尚未脫離'}）。"
    else:
        gap = thr - s_oe.iloc[-1]
        txt += f"目前未達閾值（差 {fmt0(gap)} 人次，{100 * gap / thr:.0f}%），依規則已連續 {below} 週為非流行期。" if gap > 0 else "本週已達閾值。"
    cur.append(txt)
    # --- current: outpatient & RODS
    o_wow = 100 * (s_out.iloc[-1] / s_out.iloc[-2] - 1); r_wow = 100 * (s_rods.iloc[-1] / s_rods.iloc[-2] - 1)
    cur.append(f"健保門診 {fmt0(s_out.iloc[-1])} 人次（較前一週 {o_wow:+.1f}%）；RODS 急診腸病毒 {fmt0(s_rods.iloc[-1])} 人次（{r_wow:+.1f}%），"
               f"占 RODS 急診總人次 {s_pct.iloc[-1]:.2f}%。住院 {fmt0(s_inp.iloc[-1])} 人次。")
    # --- current: age & school context
    age = pd.read_parquet(PROCESSED_DIR / "age_nhi_weekly.parquet")
    a = age[age.yw == origin].set_index("age")["ev_out"]; a4 = age[age.yw == s_out.index[-5]].set_index("age")["ev_out"] if len(s_out) > 5 else a
    tot = a.sum()
    if tot > 0:
        parts = [f"{g} 歲 {100 * a.get(g, 0) / tot:.0f}%（4 週變化 {100 * (a.get(g, 0) / max(a4.get(g, 1), 1) - 1):+.0f}%）" for g in ["0~2", "3~6", "7~12"]]
        cur.append("門診年齡分布：" + "、".join(parts) + "。")
    sc = school_calendar([origin]).iloc[0]
    if sc.reopen:
        cur.append("本週為開學後前兩週，歷史上開學後 3–4 週腸病毒就診人次每週平均上升 3–7%。")
    elif sc.summer_break:
        cur.append("本週在暑假期間，歷史上暑假前半就診人次每週平均下降 5–6%。")
    elif sc.winter_break:
        cur.append("本週在寒假期間，歷史上寒假與春節後就診人次每週下降 10–15%。")
    # --- outlook
    fc = oe["forecast"]; meds = [f["median"] for f in fc]; last = oe["last_observed"]
    ch = [100 * (m / last - 1) for m in meds]; imax = int(np.argmax(meds))
    probs = [prob_ge(f, f["threshold"] or thr) for f in fc]
    p_up1 = prob_ge(fc[0], last)
    if all(np.diff(meds) > 0):
        shape = f"未來 4 週持續上升，第 4 週中位數 {fmt0(meds[3])} 人次（較目前 {ch[3]:+.0f}%）"
    elif all(np.diff(meds) < 0):
        shape = f"未來 4 週持續下降，第 4 週中位數 {fmt0(meds[3])} 人次（較目前 {ch[3]:+.0f}%）"
    elif 0 < imax < 3:
        shape = (f"先升後緩：中位數在第 {imax + 1} 週（{fc[imax]['yw']}，{fc[imax]['date']} 起）達高點 {fmt0(meds[imax])} 人次（較目前 {ch[imax]:+.0f}%），"
                 f"第 4 週回到 {fmt0(meds[3])} 人次")
    elif imax == 0 and max(abs(c) for c in ch) < 5:
        shape = "未來 4 週大致持平（中位數在目前水準 ±5% 內）"
    else:
        shape = f"第 1 週 {fmt0(meds[0])} 人次後轉為下降，第 4 週 {fmt0(meds[3])} 人次（較目前 {ch[3]:+.0f}%）"
    fut.append(f"門急診合計：{shape}。下週高於本週的機率{pct_txt(p_up1)}，第 1 週 80% 區間 {fmt0(fc[0]['q10'])}–{fmt0(fc[0]['q90'])} 人次。")
    ptxt = "、".join(f"第 {i + 1} 週{pct_txt(p)}" for i, p in enumerate(probs))
    first = next((i for i, p in enumerate(probs) if p >= 0.5), None)
    if st["in_period"]:
        fut.append(f"維持在流行閾值 {fmt0(thr)} 人次以上的機率：{ptxt}。" + (f"中位數自第 {[i for i, m in enumerate(meds) if m < thr][0] + 1} 週起低於閾值，若連續 2 週低於即脫離流行期。" if any(m < thr for m in meds) else "中位數 4 週內都在閾值之上。"))
    else:
        fut.append(f"達到流行閾值 {fmt0(thr)} 人次的機率：{ptxt}。" + (f"以中位數判斷最快在第 {first + 1} 週（{fc[first]['yw']}，{fc[first]['date']} 起）達閾值、進入流行期。" if first is not None else "4 週內中位數都未達閾值。"))
    of = out["forecast"]; rf = rods["forecast"]
    fut.append(f"門診中位數 {' → '.join(fmt0(f['median']) for f in of)} 人次；RODS 急診 {' → '.join(fmt0(f['median']) for f in rf)} 人次。")
    w4 = (fc[3]["q90"] - fc[3]["q10"]) / meds[3] * 100
    fut.append(f"不確定性：第 4 週 80% 區間寬度為中位數的 {w4:.0f}%（{fmt0(fc[3]['q10'])}–{fmt0(fc[3]['q90'])}），3–4 週的預測僅供規劃參考。")
    # --- caveats
    flags = []
    for f in fc:
        tags = [t for t, on in (("春節週", f["cny"]), ("暑假", f["summer_break"]), ("寒假", f["winter_break"]), ("開學前兩週", f["reopen"])) if on]
        if f["holiday_days"]: tags.append(f"平日假日 {f['holiday_days']} 天")
        if tags: flags.append(f"{f['yw']}（{'、'.join(tags)}）")
    if flags:
        cav.append("預測範圍內的行事曆因素：" + "；".join(flags) + "。模型以學校行事曆與假日天數共變數納入這些效果。")
    cav.append(f"急診部分暫以 RODS 人次代替健保急診：與疾管署新聞稿公布的門急診人次相比，本合成指標在 2016–2023 年偏低 2–9%，2024 年後在 −2% 至 +6% 內（新聞稿為首報值，開放資料為回補後的數值）。詳見報告第 2 節。")
    cav.append(f"流行期判定規則（暫定）：{RULE_TEXT}")
    cav.append("健保申報有回補，最近一週的觀測值日後可能上修；本頁所有判讀由程式依規則自動產生。")
    # --- headline
    lvl = "高" if pct_rank >= 90 else ("偏高" if pct_rank >= 75 else ("中等" if pct_rank >= 40 else "低"))
    stage = ("上升期" if rising >= 2 and wow >= 5 else ("下降期" if falling >= 2 and wow <= -5 else "高原期" if lvl in ("高", "偏高") else "低度活動"))
    if 0 < imax < 3:
        outlook = f"模型預期第 {imax + 1} 週（{fc[imax]['date']} 起）前後觸頂，之後緩降"
    elif all(np.diff(meds) > 0):
        outlook = "模型預期未來 4 週持續上升"
    elif all(np.diff(meds) < 0):
        outlook = "模型預期未來 4 週持續下降"
    else:
        outlook = "模型預期未來 4 週大致持平"
    thr_part = (f"目前處於流行期第 {above} 週（閾值 {fmt0(thr)}）" if st["in_period"] else f"未達 {yr} 年流行閾值 {fmt0(thr)} 人次")
    headline = (f"腸病毒疫情處於{stage}、活動度{lvl}：門急診合計 {fmt0(s_oe.iloc[-1])} 人次，較前一週 {wow:+.0f}%，{thr_part}；"
                f"{outlook}，" + (f"最快第 {first + 1} 週達閾值（機率{pct_txt(probs[first])}）。" if (first is not None and not st["in_period"]) else f"第 4 週高於閾值的機率{pct_txt(probs[3])}。"))
    return {"headline": headline, "current": cur, "outlook": fut, "caveats": cav, "origin_yw": origin, "origin_date": o_date}


# ----------------------------------------------------------------------------- backtest
def load_summary(target: str) -> pd.DataFrame:
    parts = [pd.read_csv(BT / f"{t}_summary.csv") for t in TAGS[target] if (BT / f"{t}_summary.csv").exists()]
    s = pd.concat(parts, ignore_index=True)
    s = s[s.target == target].drop_duplicates(subset=["config", "segment", "h"], keep="last")
    ref = s[s.config == "naive"].set_index(["segment", "h"])["wis"]
    s["rel_wis_vs_naive"] = [x.wis / ref.get((x.segment, x.h), np.nan) for x in s.itertuples()]
    return s


def load_forecasts(target: str) -> pd.DataFrame:
    parts = [pd.read_parquet(BT / f"{t}_forecasts.parquet") for t in TAGS[target] if (BT / f"{t}_forecasts.parquet").exists()]
    df = pd.concat(parts, ignore_index=True)
    df = df[df.target == target].drop_duplicates(subset=["config", "origin_yw", "h"], keep="last").copy()
    df["segment"] = df.target_year.map(segment_of)
    return df


def build_backtest() -> dict:
    targets = {}
    for target in TAGS:
        summ = load_summary(target); fc = load_forecasts(target); nd = IND[target]["decimals"]
        segs = {}
        for seg in ["post_covid", "pre_covid", "covid", "all"]:
            sub = summ[(summ.segment == seg) & (summ.h <= 4)]
            metrics = ["wis", "mae", "mape", "mase", "cov80", "cov60", "dir_hit", "dir_hit3", "hit_tol10", "thr_hit_rate", "thr_false_alarm", "thr_accuracy"]
            lb = sub.groupby("config")[metrics].mean()
            lb["rel_naive"] = lb["wis"] / lb.loc["naive", "wis"]; lb["rel_snaive"] = lb["wis"] / lb.loc["snaive", "wis"]
            lb = lb.sort_values("wis")
            rows = [{"config": c, "label": CONFIG_LABELS.get(c, c), "kind": "baseline" if c in BASELINES else "timesfm",
                     **{m: r(v, 4 if m not in ("wis", "mae") else nd + 1) for m, v in row.items()}} for c, row in lb.iterrows()]
            by_h = lambda col: {c: [r(v, 4 if col != "wis" else nd + 1) for v in sub[sub.config == c].sort_values("h")[col]] for c in lb.index}
            n_or = int(fc.origin_yw.nunique()) if seg == "all" else int(fc[fc.segment == seg].origin_yw.nunique())
            segs[seg] = {"label": SEG_LABELS[seg], "leaderboard": rows, "wis_by_h": by_h("wis"), "dir_hit_by_h": by_h("dir_hit"),
                         "thr_hit_by_h": by_h("thr_hit_rate"), "cov80_by_h": by_h("cov80"), "n_origins": n_or}
        best = segs["post_covid"]["leaderboard"][0]["config"]
        focus = [c for c in [best, "tfm_cov_school_hol", "tfm_expanding", "ets", "ma3", "naive", "snaive"] if c in set(fc.config)]
        yearly = {}
        for h in (1, 4):
            t = fc[(fc.h == h) & (fc.config.isin(focus))].groupby(["config", "target_year"])["wis"].mean().unstack("target_year")
            yearly[f"h{h}"] = {c: {str(y): r(v, nd + 1) for y, v in t.loc[c].items()} for c in focus if c in t.index}
        ts = {}
        post = fc[fc.segment == "post_covid"]
        for h in (1, 2, 3, 4):
            b = post[(post.config == best) & (post.h == h)].sort_values("target_yw")
            n = post[(post.config == "naive") & (post.h == h)].set_index("target_yw")["median"]
            ts[f"h{h}"] = [{"yw": str(x.target_yw), "date": ds(x.target_yw), "y": r(x.y, nd), "median": r(x["median"], nd),
                            "q10": r(x.q10, nd), "q90": r(x.q90, nd), "q20": r(x.q20, nd), "q80": r(x.q80, nd),
                            "naive": r(n.get(x.target_yw, np.nan), nd)} for _, x in b.iterrows()]
        thr_note = "各年疾管署流行閾值（回測窗 2016–2025 均為 11,000 人次）" if target == PRIMARY else "資料窗內實際值第 75 百分位"
        targets[target] = {**IND[target], "threshold": r(summ.thr.iloc[0], nd), "thr_note": thr_note, "best": best, "best_label": CONFIG_LABELS.get(best, best),
                           "segments": segs, "yearly_wis": yearly, "timeseries": ts,
                           "n_origins": int(fc.origin_yw.nunique()), "origin_first": str(fc.origin_yw.min()), "origin_last": str(fc.origin_yw.max())}
    return {"targets": targets, "groups": None, "config_labels": CONFIG_LABELS, "seg_labels": SEG_LABELS}


# ----------------------------------------------------------------------------- calibration of the composite indicator
def calibration(nat: pd.DataFrame) -> dict:
    rows = []
    for w, v in OFFICIAL.items():
        if w in nat.index and not pd.isna(nat.loc[w, "ev_oe"]):
            x = nat.loc[w]
            rows.append({"yw": w, "official": v, "out": int(x.ev_out), "rods": int(x.ev_rods), "composite": int(x.ev_oe),
                         "diff_pct": round(100 * (x.ev_oe - v) / v, 1), "implied_er": int(v - x.ev_out)})
    d = pd.DataFrame(rows)
    return {"rows": rows, "mean_diff_pct": r(d.diff_pct.mean(), 1), "min_diff_pct": r(d.diff_pct.min(), 1), "max_diff_pct": r(d.diff_pct.max(), 1)}


# ----------------------------------------------------------------------------- report
def pct_impr(a, b):
    return round(100 * (1 - a / b))


def build_report(bt: dict, latest: dict, meta: dict, calib: dict) -> str:
    t = bt["targets"][PRIMARY]; post = t["segments"]["post_covid"]; pre = t["segments"]["pre_covid"]
    lb = {x["config"]: x for x in post["leaderboard"]}
    best = lb[t["best"]]; zs = lb.get("tfm_expanding"); naive = lb["naive"]; ma3 = lb["ma3"]; sn = lb["snaive"]; ets = lb.get("ets")
    wh = post["wis_by_h"]; impr = [pct_impr(a, b) for a, b in zip(wh[t["best"]], wh["naive"])]
    pre_best = pre["leaderboard"][0]
    dh = post["dir_hit_by_h"][t["best"]]
    tout = bt["targets"]["ev_out"]; lo = {x["config"]: x for x in tout["segments"]["post_covid"]["leaderboard"]}; obest = lo[tout["best"]]
    li = {x["key"]: x for x in latest["indicators"]}; f1 = li[PRIMARY]["forecast"]; st = latest["status"]
    gen = meta["generated_at"][:10]
    fmt = lambda v, nd=0: "—" if v is None else f"{v:,.{nd}f}"
    cov = meta["coverage"]
    thr_rows = "".join(f"<tr><td>{y}</td><td class='num'>{v:,.0f}</td></tr>" for y, v in st["thresholds_by_year"].items())
    cal_rows = "".join(f"<tr><td>{x['yw']}</td><td class='num'>{x['official']:,}</td><td class='num'>{x['out']:,}</td><td class='num'>{x['rods']:,}</td><td class='num'>{x['composite']:,}</td><td class='num'>{x['diff_pct']:+.1f}%</td></tr>" for x in calib["rows"])
    lb_rows = "".join(f"<tr class='{'best' if x['config'] == t['best'] else ''}'><td>{x['label']}</td><td class='num'>{fmt(x['wis'])}</td><td class='num'>{x['rel_naive']:.2f}</td><td class='num'>{x['mape']:.1f}</td><td class='num'>{x['cov80']:.2f}</td><td class='num'>{x['dir_hit']:.2f}</td><td class='num'>{fmt(x['thr_hit_rate'], 2)}</td><td class='num'>{fmt(x['thr_false_alarm'], 2)}</td></tr>"
                      for x in [best] + [lb[c] for c in ["tfm_cov_school_hol", "tfm_expanding", "ets", "ma3", "naive", "snaive"] if c in lb and c != t["best"]])
    layer_rows = "".join(f"<tr class='{'best' if x['config'] == tout['best'] else ''} {'baseline' if x['kind'] == 'baseline' else ''}'><td>{x['label']}</td><td class='num'>{fmt(x['wis'])}</td><td class='num'>{x['rel_naive']:.2f}</td><td class='num'>{x['mape']:.1f}</td><td class='num'>{x['cov80']:.2f}</td><td class='num'>{x['dir_hit']:.2f}</td></tr>"
                         for x in tout["segments"]["post_covid"]["leaderboard"] if x["config"] != "snaive")
    head = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>解讀與評估報告 · 台灣腸病毒 TimesFM 3.0 預測</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;600;700&family=Noto+Serif+TC:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../assets/epi.css?v=ev1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="../assets/charts.js?v=ev1" defer></script></head>
<body>
<header class="masthead"><div class="masthead-inner"><div class="masthead-left"><div class="crest">FT</div><div class="masthead-title">台灣腸病毒疫情預測 · TimesFM 3.0<small>FORECAST-TELLER · EV · 解讀與評估報告</small></div></div>
<nav class="masthead-right"><a href="../index.html">流感</a><a href="index.html">每週預測</a><a href="backtest.html">回測</a><a href="report.html" class="active">報告</a><a href="../dengue/">登革熱預警</a></nav></div></header>
<main class="report">
"""
    body = f"""<div class="hero-kicker">EVALUATION REPORT · {gen}</div>
<h1 class="report-title">以 TimesFM 3.0 零樣本預測台灣腸病毒門急診就診：<em>回測解讀與評估</em></h1>
<p class="lead">資料窗 2016w01–2025w53（522 週），以 2016–2017 為暖機 context，2018w01 起每週一個起點共 {t['n_origins']} 個，預測未來 1–4 週；主要決策依據為 COVID 後（2023–2025）的加權區間分數（WIS）。目標序列為全國腸病毒門急診就診人次，其中急診部分暫以 RODS 急診腸病毒人次代替。</p>

<div class="kpi-strip">
  <div class="kpi"><div class="label">最佳設定 WIS（COVID 後，h=1–4）</div><div class="value">{fmt(best['wis'])}</div><div class="sub">{best['label']}</div></div>
  <div class="kpi"><div class="label">相對 last-value naive</div><div class="value">−{round(100*(1-best['rel_naive']))}%</div><div class="sub">1–4 週分別 −{impr[0]}% / −{impr[1]}% / −{impr[2]}% / −{impr[3]}%</div></div>
  <div class="kpi"><div class="label">80% 預測區間涵蓋率</div><div class="value">{best['cov80']:.2f}</div><div class="sub">理想值 0.80；60% 區間 {best['cov60']:.2f}</div></div>
  <div class="kpi"><div class="label">流行閾值命中率 · 誤報率</div><div class="value">{fmt(best['thr_hit_rate'], 2)}<small>/ {fmt(best['thr_false_alarm'], 2)}</small></div><div class="sub">閾值 11,000（2016–2025）；naive {fmt(naive['thr_hit_rate'], 2)} / {fmt(naive['thr_false_alarm'], 2)}</div></div>
</div>

<section><h2>1. 我們評估了什麼</h2>
<p>目標序列是<strong>全國每週腸病毒門急診就診人次</strong>：健保申報的門診腸病毒就診人次（疾管署資料開放平台）加上 RODS 急診腸病毒就診人次。這是疾管署判定腸病毒流行期的指標；開放資料的健保檔只有門診與住院，沒有急診，因此急診部分<strong>暫以 RODS 人次代替</strong>（使用者決定，2026-09-19），日後取得健保急診檔再替換。另以健保門診人次單獨做完整的分層實驗（第 3 節）。</p>
<p>模型是 Google TimesFM 3.0（330M 參數的時間序列基礎模型），<strong>零樣本、未經微調</strong>，逐層加入已知未來共變數（學校行事曆：暑假、寒假、開學前兩週；每週平日假日天數；春節旗標；季節相位）、過去共變數（RODS、住院）與多變量聯合預測。基準為 last-value naive、MA3、季節性 naive、AutoETS、Theta。評估指標：WIS、MAE、MAPE、MASE、80%/60% 涵蓋率、方向命中率、±10% 命中率、流行閾值命中率與誤報率；結果分 COVID 前／期／後三段。</p></section>

<section><h2>2. 使用的資料與流行閾值</h2>
<table class="tbl compact"><thead><tr><th>來源</th><th>內容</th><th>粒度</th><th>完整週範圍</th></tr></thead><tbody>
<tr><td>健保門診及住院就診人次統計－腸病毒（開放資料）</td><td>腸病毒健保就診人次、健保就診總人次；門診與住院</td><td>週 × 9 年齡層 × 22 縣市</td><td class="num">{cov['nhi']['first_complete']}–{cov['nhi']['last_complete']}</td></tr>
<tr><td>急診傳染病監測統計－腸病毒（開放資料，RODS）</td><td>急診腸病毒就診人次（無分母）</td><td>週 × 5 年齡層 × 22 縣市</td><td class="num">{cov['rods']['first_complete']}–{cov['rods']['last_complete']}</td></tr>
<tr><td>RODS 急診總人次（流感面板 RODS_RS.csv）</td><td>急診就診總人次，用來算腸病毒急診就診百分比</td><td>週 × 醫院</td><td class="num">{cov['rods_total']['first']}–{cov['rods_total']['last']}</td></tr>
<tr><td>行事曆</td><td>學校暑假／寒假／開學週（近似規則）、平日假日天數、春節（tw_holiday.csv）</td><td>週</td><td class="num">2016–2027</td></tr>
</tbody></table>
<div class="two-col">
<div class="card"><h3>流行閾值（疾管署每年設定，依新聞稿記錄）</h3>
<table class="tbl compact"><thead><tr><th>年</th><th class="num">門急診就診人次</th></tr></thead><tbody>{thr_rows}</tbody></table>
<p class="source">2016、2023、2024、2025、2026 年有新聞稿佐證；其餘年份推定沿用 11,000。流行期規則（暫定）：{RULE_TEXT}</p></div>
<div class="card"><h3>合成指標（門診 + RODS）與疾管署公布值的比較</h3>
<table class="tbl compact"><thead><tr><th>週</th><th class="num">公布門急診</th><th class="num">開放資料門診</th><th class="num">RODS</th><th class="num">合成</th><th class="num">差</th></tr></thead><tbody>{cal_rows}</tbody></table>
<p class="source">公布值為新聞稿首報數字，開放資料為回補後數值。合成指標在 2016–2023 年偏低 2–9%（RODS 只涵蓋部分急診），2024 年後在 ±5% 內（平均 {calib['mean_diff_pct']:+.1f}%，範圍 {calib['min_diff_pct']:+.1f}% 至 {calib['max_diff_pct']:+.1f}%）。以合成指標套用官方閾值時，早年略偏保守。</p></div>
</div></section>

<section><h2>3. 主要結果</h2>
<div class="two-col">
<div class="card"><h3>各設定 WIS 依預測週數（門急診合計，COVID 後）</h3><div class="chart-box"><canvas id="c-wis"></canvas></div><p class="source">WIS 越低越好；灰色虛線為基準模型。</p></div>
<div class="card"><h3>最佳設定 1 週前預測 vs 實際（2023–2025）</h3><div class="chart-box"><canvas id="c-ts"></canvas></div><p class="source">虛線為預測中位數，淺綠帶為 80% 預測區間。</p></div>
</div>
<table class="tbl compact"><thead><tr><th>設定（門急診合計）</th><th class="num">WIS</th><th class="num">相對 naive</th><th class="num">MAPE %</th><th class="num">80% 涵蓋</th><th class="num">方向命中</th><th class="num">閾值命中</th><th class="num">誤報率</th></tr></thead><tbody>{lb_rows}</tbody></table>
<ul class="findings">
<li><strong>零樣本已勝過統計基準。</strong>不加共變數的 TimesFM，WIS 比 last-value naive 低 {round(100*(1-zs['rel_naive'])) if zs else '—'}%、比 MA3 低 {pct_impr(zs['wis'], ma3['wis']) if zs else '—'}%{('、比 AutoETS 低 %d%%' % pct_impr(zs['wis'], ets['wis'])) if (zs and ets) else ''}。季節性 naive 的 WIS 是 naive 的 {sn['rel_naive']:.1f} 倍：腸病毒一年兩波且波形逐年不同，去年同週幾乎沒有參考價值。</li>
<li><strong>最佳設定：{best['label']}</strong>，相對 naive 在 1–4 週分別改善 {impr[0]}% / {impr[1]}% / {impr[2]}% / {impr[3]}%；方向命中率 1 週前 {dh[0]:.2f}、4 週前 {dh[3]:.2f}。</li>
<li><strong>校準。</strong>80% 區間涵蓋率 {best['cov80']:.2f}、60% 涵蓋率 {best['cov60']:.2f}；naive 為 {naive['cov80']:.2f}。</li>
<li><strong>COVID 前（2018–2019）</strong>最佳為 {pre_best['label']}，WIS {fmt(pre_best['wis'])}，相對 naive −{round(100*(1-pre_best['rel_naive']))}%。</li>
<li><strong>流行閾值（11,000）命中。</strong>最佳設定 1–4 週平均命中率 {fmt(best['thr_hit_rate'], 2)}、誤報率 {fmt(best['thr_false_alarm'], 2)}；naive 為 {fmt(naive['thr_hit_rate'], 2)} / {fmt(naive['thr_false_alarm'], 2)}。</li>
</ul>
<h3 style="margin-top:18px">門診人次的分層實驗（COVID 後，h = 1–4 平均）</h3>
<table class="tbl compact"><thead><tr><th>設定（門診）</th><th class="num">WIS</th><th class="num">相對 naive</th><th class="num">MAPE %</th><th class="num">80% 涵蓋</th><th class="num">方向命中</th></tr></thead><tbody>{layer_rows}</tbody></table>
<p class="source">門診最佳：{obest['label']}（相對 naive {obest['rel_naive']:.2f}）。灰字為基準模型；季節性 naive 略去。</p></section>

<section><h2>4. 怎麼解讀</h2>
<ul class="findings">
<li><strong>1–2 週最可靠。</strong>WIS 隨 horizon 上升（{fmt(wh[t['best']][0])} → {fmt(wh[t['best']][3])}）。4 週的預測應只做規劃參考。</li>
<li><strong>用機率而非單點判斷是否進入流行期。</strong>每週頁以 9 個分位數推得「達閾值的機率」；建議以機率 ≥ 50% 作為「可能進入流行期」的預警，並與規則（連續 2 週低於閾值才脫離）一起看。</li>
<li><strong>型別轉換是模型看不到的。</strong>流行規模與重症由 EV-A71、CVA6、CVA16、E11、EV-D68 等型別組成決定，零樣本模型只從就診人次的形狀外推；遇到型別轉換造成的異常季節（如 2025 年秋末才進入流行期）會慢半拍。</li>
<li><strong>COVID 期分數不作決策依據。</strong>2020–2022 幾乎沒有腸病毒流行。</li>
</ul></section>

<section><h2>5. 限制與注意事項</h2>
<ul class="findings">
<li>急診以 RODS 代替：RODS 只涵蓋部分急診，合成指標早年偏低 2–9%；取得健保急診檔後應重跑。</li>
<li>開放資料有回補：新聞稿首報值與日後開放資料相差可達 3–5%；建議每週保存快照以估計回補係數。</li>
<li>學校行事曆為近似規則（暑假 7/1–8/29、寒假 1/21 至春節後 6 天、開學前兩週），非各學年實際日期。</li>
<li>閾值 2017–2019 年未找到新聞稿佐證，推定為 11,000；2026 年 12,000 依新聞稿，均待疾管署正式文件確認。</li>
<li>TimesFM 3.0 權重為非商業、非生產用途授權。</li>
</ul></section>

<section><h2>6. 最新預測摘要（起點 {li[PRIMARY]['origin_yw']}，{li[PRIMARY]['origin_date']} 當週）</h2>
<p>門急診合計未來 4 週中位數：{'、'.join(fmt(x['median']) for x in f1)} 人次；第 1 週 80% 區間 {fmt(f1[0]['q10'])}–{fmt(f1[0]['q90'])}。{f1[0]['yw'][:4]} 年流行閾值 {fmt(st['threshold'])} 人次；目前{'處於' if st['in_period'] else '未進入'}流行期。完整內容見<a href="index.html">每週預測</a>。</p></section>

<footer class="report-footer">資料：疾病管制署資料開放平台（健保門診腸病毒、RODS 急診腸病毒）、RODS_RS.csv（急診總人次）。模型：google/timesfm-3.0-pytorch（非商業授權）。產生時間 {meta['generated_at']}。圖表依《疫情資料視覺化指引》v1.1。</footer>
</main>
"""
    script = """<script>
document.addEventListener('DOMContentLoaded', async () => {
  const bt = await (await fetch('data/backtest.json')).json();
  const t = bt.targets['ev_oe']; const seg = t.segments.post_covid;
  const cfgs = [t.best, 'tfm_cov_school_hol', 'tfm_expanding', 'ets', 'ma3', 'naive'].filter((c, i, a) => a.indexOf(c) === i && seg.wis_by_h[c]);
  EPI.linesChart(document.getElementById('c-wis'), EPI.byHorizonSeries(seg, cfgs, 'wis_by_h', bt.config_labels), {yLabel: 'WIS', xLabel: '預測週數（h）', fmt: v => v.toLocaleString('zh-TW', {maximumFractionDigits: 0})});
  EPI.backtestTsChart(document.getElementById('c-ts'), t.timeseries.h1, {unit: t.unit, decimals: t.decimals});
});
</script>
</body></html>"""
    return head + body + script


def main(config_labels: dict[str, str]):
    DATA.mkdir(parents=True, exist_ok=True)
    nat = load_national()
    cov = json.loads((PROCESSED_DIR / "coverage.json").read_text(encoding="utf-8"))
    now = dt.datetime.now().astimezone().isoformat(timespec="minutes")
    calib = calibration(nat)
    latest = build_latest(nat, config_labels); latest["generated_at"] = now
    latest["narrative"] = build_narrative(nat, latest, calib)
    bt = build_backtest(); bt["generated_at"] = now
    best = bt["targets"][PRIMARY]["best"]
    meta = {"generated_at": now, "coverage": cov, "calibration": calib,
            "backtest_window": {"start": "201601", "end": "202553", "weeks": 522, "warmup": 104, "origins": bt["targets"][PRIMARY]["n_origins"]},
            "model": {"name": "TimesFM 3.0", "checkpoint": "google/timesfm-3.0-pytorch", "license": "timesfm-non-commercial-license-v1.0（非商業、非生產用途）"},
            "best_config": best, "best_label": CONFIG_LABELS.get(best, best), "rule": RULE_TEXT, "thresholds_by_year": latest["status"]["thresholds_by_year"]}
    (DATA / "latest.json").write_text(json.dumps(latest, ensure_ascii=False), encoding="utf-8")
    (DATA / "backtest.json").write_text(json.dumps(bt, ensure_ascii=False), encoding="utf-8")
    (DATA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (DOCS_DIR / "report.html").write_text(build_report(bt, latest, meta, calib), encoding="utf-8")
    print("site data written:", {p.name: p.stat().st_size // 1024 for p in DATA.glob('*.json')}, "KB; best", best)
    print("headline:", latest["narrative"]["headline"])
