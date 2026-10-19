"""Build the COVID-19 pages of the GitHub Pages site (docs/covid/): data JSON, rule-based narrative, report."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from forecast_teller.backtest import SEGMENTS, segment_of
from forecast_teller.threshold_probs import threshold_outlook
from forecast_teller.site_common import THRESHOLD_PROB_LINE, method_page, page_head, reading_items, table, ul
from forecast_teller.weeks import week_start

from . import DATA_DIR, DOCS_DIR, OUTPUT_DIR, PROCESSED_DIR
from .panel import load_national
from .thresholds import RULE_TEXT, epidemic_flags, load_thresholds, periods

DATA = DOCS_DIR / "data"; BT = OUTPUT_DIR / "backtest"; LATEST = OUTPUT_DIR / "latest"
PRIMARY = "covid_oe"
SEGMENTS.clear(); SEGMENTS.update({"y2024": (2024, 2024), "y2025": (2025, 2025), "y2026": (2026, 2026)})
IND = {
    "covid_oe": {"label": "全國新冠門急診就診人次（健保門診 + 急診）", "short": "門急診合計", "unit": "人次", "decimals": 0,
                 "source": "NIDSS 傳染病統計資料查詢系統「健保門急診就診」新冠病毒感染（疾管署新聞稿使用的指標）"},
    "covid_out": {"label": "全國新冠健保門診就診人次", "short": "門診", "unit": "人次", "decimals": 0, "source": "健保申報（疾管署資料開放平台 NHI_COVID-19）"},
    "covid_rods": {"label": "RODS 急診新冠就診人次", "short": "RODS 急診", "unit": "人次", "decimals": 0, "source": "即時疫情監視及預警系統 RODS（資料開放平台 RODS_COVID-19）"},
    "covid_rods_pct": {"label": "RODS 急診新冠就診百分比", "short": "急診新冠%", "unit": "%", "decimals": 2,
                       "source": "RODS 新冠人次 ÷ RODS 急診總人次（分母來自流感面板的 RODS_RS.csv）"},
    "covid_inp": {"label": "全國新冠健保住院人次", "short": "住院", "unit": "人次", "decimals": 0, "source": "健保申報（資料開放平台 NHI_COVID-19）"},
    "covid_oe_alt": {"label": "門急診合成指標（開放資料門診 + RODS 急診）", "short": "門診 + RODS", "unit": "人次", "decimals": 0,
                     "source": "開放資料門診 + RODS 急診；與疾管署新聞稿首報值最接近的合成（對照用）"},
}
CONFIG_LABELS = {
    "snaive": "季節性 naive（去年同週）", "naive": "last-value naive", "ma3": "MA3（前 3 週移動平均）", "ets": "AutoETS", "theta": "Theta",
    "tfm_expanding": "TimesFM zero-shot（無共變數）", "tfm_expanding_log1p": "TimesFM（log1p）", "tfm_expanding_sym": "TimesFM（對稱平均）",
    "tfm_slide104": "TimesFM（滑動 104 週）", "tfm_ctx2023": "TimesFM（context 自 2023w12）",
    "tfm_cov_hol": "TimesFM + 假日天數", "tfm_cov_cny_hol": "TimesFM + 春節 + 假日", "tfm_cov_season_cny_hol": "TimesFM + 季節相位 + 春節 + 假日",
    "tfm_cov_cny_hol_log1p": "TimesFM + 春節 + 假日（log1p）",
    "tfm_mv_out_rods": "三變量聯合（門急診 + 門診 + RODS）+ 春節 + 假日", "tfm_mv_out_rods_inp": "四變量聯合（門急診 + 門診 + RODS + 住院）+ 春節 + 假日",
    "tfm_mv_out_rods_nocov": "三變量聯合（門急診 + 門診 + RODS，無共變數）", "tfm_po_rods": "TimesFM + RODS（過去共變數）+ 春節 + 假日",
}
SEG_LABELS = {"all": "全期（2024–2026）", "y2026": "2026 年", "y2025": "2025 年", "y2024": "2024 年"}
SEG_ORDER = ["all", "y2026", "y2025", "y2024"]
DECISION_SEG = "all"
BASELINES = {"snaive", "naive", "ma3", "ets", "theta"}
TAGS = {"covid_oe": ["baselines_covid_oe", "stats_covid_oe", "layer1_covid_oe", "layer2_covid_oe", "layer3_covid_oe", "core_covid_oe"],
        "covid_oe_alt": ["baselines_covid_oe_alt", "stats_covid_oe_alt", "core_covid_oe_alt"],
        "covid_out": ["baselines_covid_out", "stats_covid_out", "core_covid_out"]}
# 疾管署新聞稿公布的「門急診就診人次」（首報值），用來校驗 NIDSS 門急診與「門診 + RODS」合成指標
OFFICIAL = {"202622": 1000, "202628": 4766, "202629": 10605, "202630": 18990, "202631": 24645, "202633": 24995, "202635": 20199}


def r(x, nd=3):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def ds(yw: str) -> str:
    return str(week_start(str(yw)).date())


def load_variants() -> pd.DataFrame:
    p = DATA_DIR / "variants.csv"
    if not p.exists():
        return pd.DataFrame(columns=["yw", "variant", "share_pct", "source"])
    return pd.read_csv(p, dtype={"yw": str}, encoding="utf-8-sig").sort_values("yw")


# ----------------------------------------------------------------------------- latest
def build_latest(nat: pd.DataFrame, config_labels: dict[str, str]) -> dict:
    joint = pd.read_csv(LATEST / "latest_forecast_joint.csv", dtype={"origin_yw": str, "target_yw": str})
    uni = pd.read_csv(LATEST / "latest_forecast.csv", dtype={"origin_yw": str, "target_yw": str})
    src0 = joint if PRIMARY in set(joint.target) else uni
    origin0 = str(src0[src0.target == PRIMARY].origin_yw.iloc[0])
    oe = nat[PRIMARY].dropna().loc[:origin0]          # status as of the forecast origin (NIDSS may already have one more week)
    fl = epidemic_flags(oe)
    level = float(fl["level"].iloc[-1]); in_p = bool(fl["in_period"].iloc[-1])
    out = []
    for key in ["covid_oe", "covid_out", "covid_rods", "covid_rods_pct", "covid_inp", "covid_oe_alt"]:
        src = joint if key in set(joint.target) else uni
        rows = src[src.target == key].sort_values("h")
        if rows.empty:
            continue
        origin = str(rows.origin_yw.iloc[0])
        hist = nat[key].dropna().loc[:origin].iloc[-130:]
        nd = IND[key]["decimals"]
        history = [{"yw": w, "date": ds(w), "y": r(v, nd)} for w, v in hist.items()]
        forecast = []
        for _, x in rows.iterrows():
            w = str(x.target_yw)
            forecast.append({"yw": w, "date": ds(w), "h": int(x.h), "median": r(x["median"], nd),
                             **{f"q{q}": r(x[f"q{q}"], nd) for q in range(10, 100, 10)},
                             "cny": int(x.cny), "holiday_days": int(x.holiday_days),
                             "observed": r(nat[key].get(w, np.nan), nd)})
        extra = {}
        if key == PRIMARY:
            extra = {"level": r(level, 0), "level_label": ("暫行脫離水準（本波峰值一半）" if in_p else "暫行進入水準（本波低點 2 倍）") + f" {level:,.0f} 人次"}
        out.append({"key": key, **IND[key], "config": config_labels.get(key, ""), "mode": str(rows["mode"].iloc[0]), "origin_yw": origin,
                    "origin_date": ds(origin), "last_observed": r(nat[key].loc[origin], nd), "history": history, "forecast": forecast, **extra})
    per = periods(fl)
    status = {"in_period": in_p, "week": str(oe.index[-1]), "level": level, "trough": float(fl["trough"].iloc[-1]), "peak": float(fl["peak"].iloc[-1]),
              "rises": int(fl["rises"].iloc[-1]), "falls": int(fl["falls"].iloc[-1]),
              "periods": [{"start": a, "end": b} for a, b in per[-6:]], "rule": RULE_TEXT,
              "thresholds_by_year": {str(y): (None if pd.isna(v) else float(v)) for y, v in load_thresholds()["threshold"].items()}}
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
    if pct >= 25: return "快速上升"
    if pct >= 5: return "上升"
    if pct > -5: return "大致持平"
    if pct > -25: return "下降"
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
    oe, out, rods = li["covid_oe"], li["covid_out"], li["covid_rods"]
    st = latest["status"]
    origin = oe["origin_yw"]; o_date = oe["origin_date"]
    s_oe = nat["covid_oe"].dropna().loc[:origin]; s_out = nat["covid_out"].dropna().loc[:origin]
    s_rods = nat["covid_rods"].dropna().loc[:origin]; s_pct = nat["covid_rods_pct"].dropna().loc[:origin]; s_inp = nat["covid_inp"].dropna().loc[:origin]
    yr = int(origin[:4])
    fmt0 = lambda v: f"{v:,.0f}"
    cur, fut, cav = [], [], []

    # --- current: primary indicator and wave status
    wow = 100 * (s_oe.iloc[-1] / s_oe.iloc[-2] - 1); w3 = 100 * (s_oe.iloc[-1] / s_oe.iloc[-4] - 1)
    rising = consecutive(s_oe.diff().dropna(), lambda d: d > 0); falling = consecutive(s_oe.diff().dropna(), lambda d: d < 0)
    yrs = pd.Series(nat.index.str[:4].astype(int), index=nat.index)
    same = nat[(nat.index.str[4:] == origin[4:]) & yrs.between(yr - 2, yr - 1)]["covid_oe"].dropna()
    hist = s_oe.loc["202312":]; pct_rank = 100 * (hist < s_oe.iloc[-1]).mean()
    peak_w = s_oe.loc["202301":].idxmax() if False else s_oe.iloc[-30:].idxmax()
    txt = (f"全國新冠門急診就診人次 {origin}（{o_date} 起）為 {fmt0(s_oe.iloc[-1])} 人次，較前一週{trend_word(wow)}（{wow:+.1f}%），較 3 週前 {w3:+.0f}%")
    if rising >= 2: txt += f"，已連續 {rising} 週上升"
    elif falling >= 2: txt += f"，已連續 {falling} 週下降"
    if len(same):
        txt += f"；為前兩年同一週平均（{fmt0(same.mean())}）的 {s_oe.iloc[-1] / same.mean():.1f} 倍，在 2023 年 3 月取消全面通報以來的所有週中位於第 {pct_rank:.0f} 百分位"
    if st["in_period"]:
        txt += f"。依暫行規則目前處於流行期（本波峰值 {fmt0(st['peak'])} 人次，{peak_w} 當週），脫離水準為峰值一半 {fmt0(st['level'])} 人次。"
    else:
        txt += f"。依暫行規則目前不在流行期；本波低點 {fmt0(st['trough'])} 人次，進入水準為低點 2 倍 {fmt0(st['level'])} 人次。"
    cur.append(txt)
    # --- current: components
    o_wow = 100 * (s_out.iloc[-1] / s_out.iloc[-2] - 1); r_wow = 100 * (s_rods.iloc[-1] / s_rods.iloc[-2] - 1)
    er_est = nat["covid_er_est"].dropna().loc[:origin]
    cur.append(f"健保門診 {fmt0(s_out.iloc[-1])} 人次（較前一週 {o_wow:+.1f}%），健保急診約 {fmt0(er_est.iloc[-1])} 人次（占門急診 {100 * er_est.iloc[-1] / s_oe.iloc[-1]:.0f}%）；"
               f"RODS 急診新冠 {fmt0(s_rods.iloc[-1])} 人次（{r_wow:+.1f}%），占 RODS 急診總人次 {s_pct.iloc[-1]:.2f}%。住院 {fmt0(s_inp.iloc[-1])} 人次。")
    # --- current: age
    age = pd.read_parquet(PROCESSED_DIR / "age_nhi_weekly.parquet")
    a = age[age.yw == s_out.index[-1]].set_index("age")["covid_out"]; a4 = age[age.yw == s_out.index[-5]].set_index("age")["covid_out"] if len(s_out) > 5 else a
    tot = a.sum()
    if tot > 0:
        parts = [f"{g} 歲 {100 * a.get(g, 0) / tot:.0f}%（4 週變化 {100 * (a.get(g, 0) / max(a4.get(g, 1), 1) - 1):+.0f}%）" for g in ["65+", "25~64"]]
        kids = sum(a.get(g, 0) for g in ["0~2", "3~6", "7~12", "13~15", "16~18"])
        cur.append("門診年齡分布：" + "、".join(parts) + f"、18 歲以下 {100 * kids / tot:.0f}%。")
    # --- current: severe, RESP, variant
    sev = nat["covid_severe"].dropna()
    if len(sev) >= 8:
        cur.append(f"新冠併發重症（發病週，通報延遲約 3 週）最新完整週 {sev.index[-1]} 為 {fmt0(sev.iloc[-1])} 例，近 4 週平均 {sev.iloc[-4:].mean():.0f} 例，"
                   f"較前 4 週 {100 * (sev.iloc[-4:].mean() / max(sev.iloc[-8:-4].mean(), 1) - 1):+.0f}%。")
    if "resp_covid" in nat.columns:
        rr = nat[["resp_covid", "resp_total_pos", "resp_pos_pct"]].dropna()
        if len(rr):
            x = rr.iloc[-1]
            cur.append(f"社區合約實驗室多重 PCR（RESP_LAB）{rr.index[-1]}：{fmt0(x.resp_total_pos)} 件陽性檢體中 SARS-CoV-2 占 {100 * x.resp_covid / max(x.resp_total_pos, 1):.0f}%（{fmt0(x.resp_covid)} 件），總陽性率 {x.resp_pos_pct:.0f}%。")
    var = load_variants()
    if len(var):
        v = var.iloc[-1]
        cur.append(f"主流變異株（人工記錄，{v.source}）：{v.variant}。")
    # --- outlook
    fc = oe["forecast"]; meds = [f["median"] for f in fc]; last = oe["last_observed"]
    ch = [100 * (m / last - 1) for m in meds]; imax = int(np.argmax(meds))
    p_up = [prob_ge(f, last) for f in fc]
    p_lvl = [prob_ge(f, st["level"]) for f in fc]
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
    fut.append(f"門急診合計：{shape}。下週高於本週的機率{pct_txt(p_up[0])}，第 1 週 80% 區間 {fmt0(fc[0]['q10'])}–{fmt0(fc[0]['q90'])} 人次。")
    fut.append("高於目前水準（" + fmt0(last) + " 人次）的機率：" + "、".join(f"第 {i + 1} 週{pct_txt(p)}" for i, p in enumerate(p_up)) + "。")
    if st["in_period"]:
        fut.append(f"維持在暫行脫離水準 {fmt0(st['level'])} 人次以上的機率：" + "、".join(f"第 {i + 1} 週{pct_txt(p)}" for i, p in enumerate(p_lvl)) +
                   "。" + (f"中位數自第 {[i for i, m in enumerate(meds) if m < st['level']][0] + 1} 週起低於脫離水準，若同時連續 2 週下降即脫離流行期。" if any(m < st["level"] for m in meds) else "中位數 4 週內都在脫離水準之上。"))
    else:
        first = next((i for i, p in enumerate(p_lvl) if p >= 0.5), None)
        fut.append(f"達暫行進入水準 {fmt0(st['level'])} 人次的機率：" + "、".join(f"第 {i + 1} 週{pct_txt(p)}" for i, p in enumerate(p_lvl)) +
                   "。" + (f"以中位數判斷最快在第 {first + 1} 週達到，但還需連續 3 週上升才符合進入規則。" if first is not None else "4 週內中位數都未達進入水準。"))
    if latest.get("threshold_outlook"):
        fut.append(latest["threshold_outlook"]["text"])
    of = out["forecast"]; rf = rods["forecast"]
    fut.append(f"門診中位數 {' → '.join(fmt0(f['median']) for f in of)} 人次；RODS 急診 {' → '.join(fmt0(f['median']) for f in rf)} 人次。")
    w4 = (fc[3]["q90"] - fc[3]["q10"]) / meds[3] * 100
    fut.append(f"不確定性：第 4 週 80% 區間寬度為中位數的 {w4:.0f}%（{fmt0(fc[3]['q10'])}–{fmt0(fc[3]['q90'])}），新冠波動大，3–4 週的預測僅供規劃參考。")
    # --- caveats
    flags = []
    for f in fc:
        tags = (["春節週"] if f["cny"] else []) + ([f"平日假日 {f['holiday_days']} 天"] if f["holiday_days"] else [])
        if tags:
            flags.append(f"{f['yw']}（{'、'.join(tags)}）")
    if flags:
        cav.append("預測範圍內含假日：" + "；".join(flags) + "。門診量會因休診下降、急診相對上升，模型已透過春節旗標與假日天數共變數納入。")
    cav.append(f"門急診人次為 NIDSS 查詢系統的回補後數值，比疾管署新聞稿首報值高約 {calib['mean_diff_nidss_pct']:+.0f}%（「開放資料門診 + RODS」合成指標與首報值差 {calib['mean_diff_alt_pct']:+.0f}%）；本頁每週保存首報快照以估計回補。詳見報告第 2 節。")
    cav.append(RULE_TEXT)
    cav.append("回測只涵蓋 2024–2026 三個夏季波與兩個冬春波（141 個起點），結論屬初步；每週預測均存檔並與實際值比對。本頁所有判讀由程式依規則自動產生。")
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
    period_part = "依暫行規則處於流行期" if st["in_period"] else "依暫行規則不在流行期"
    headline = (f"新冠疫情處於{stage}、活動度{lvl}：門急診 {fmt0(s_oe.iloc[-1])} 人次，較前一週 {wow:+.0f}%，{period_part}；"
                f"{outlook}，下週續升機率{pct_txt(p_up[0])}。")
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
        if not any((BT / f"{t}_summary.csv").exists() for t in TAGS[target]):
            continue
        summ = load_summary(target); fc = load_forecasts(target); nd = IND[target]["decimals"]
        segs = {}
        for seg in SEG_ORDER:
            sub = summ[(summ.segment == seg) & (summ.h <= 4)]
            if sub.empty:
                continue
            metrics = ["wis", "mae", "mape", "mase", "cov80", "cov60", "dir_hit", "dir_hit3", "hit_tol10", "thr_hit_rate", "thr_false_alarm", "thr_accuracy"]
            lb = sub.groupby("config")[metrics].mean()
            lb["rel_naive"] = lb["wis"] / lb.loc["naive", "wis"]; lb["rel_snaive"] = lb["wis"] / lb.loc["snaive", "wis"] if "snaive" in lb.index else np.nan
            lb = lb.sort_values("wis")
            rows = [{"config": c, "label": CONFIG_LABELS.get(c, c), "kind": "baseline" if c in BASELINES else "timesfm",
                     **{m: r(v, 4 if m not in ("wis", "mae") else nd + 1) for m, v in row.items()}} for c, row in lb.iterrows()]
            by_h = lambda col: {c: [r(v, 4 if col != "wis" else nd + 1) for v in sub[sub.config == c].sort_values("h")[col]] for c in lb.index}
            n_or = int(fc.origin_yw.nunique()) if seg == "all" else int(fc[fc.segment == seg].origin_yw.nunique())
            segs[seg] = {"label": SEG_LABELS[seg], "leaderboard": rows, "wis_by_h": by_h("wis"), "dir_hit_by_h": by_h("dir_hit"),
                         "thr_hit_by_h": by_h("thr_hit_rate"), "cov80_by_h": by_h("cov80"), "n_origins": n_or}
        best = segs[DECISION_SEG]["leaderboard"][0]["config"]
        focus = [c for c in [best, "tfm_cov_cny_hol", "tfm_expanding", "ets", "ma3", "naive", "snaive"] if c in set(fc.config)]
        yearly = {}
        for h in (1, 4):
            t = fc[(fc.h == h) & (fc.config.isin(focus))].groupby(["config", "target_year"])["wis"].mean().unstack("target_year")
            yearly[f"h{h}"] = {c: {str(y): r(v, nd + 1) for y, v in t.loc[c].items()} for c in focus if c in t.index}
        ts = {}
        for h in (1, 2, 3, 4):
            b = fc[(fc.config == best) & (fc.h == h)].sort_values("target_yw")
            n = fc[(fc.config == "naive") & (fc.h == h)].set_index("target_yw")["median"]
            ts[f"h{h}"] = [{"yw": str(x.target_yw), "date": ds(x.target_yw), "y": r(x.y, nd), "median": r(x["median"], nd),
                            "q10": r(x.q10, nd), "q90": r(x.q90, nd), "q20": r(x.q20, nd), "q80": r(x.q80, nd),
                            "naive": r(n.get(x.target_yw, np.nan), nd)} for _, x in b.iterrows()]
        targets[target] = {**IND[target], "threshold": r(summ.thr.iloc[0], nd), "thr_note": "資料窗內實際值第 75 百分位（疾管署流行閾值待確認）",
                           "best": best, "best_label": CONFIG_LABELS.get(best, best),
                           "segments": segs, "yearly_wis": yearly, "timeseries": ts,
                           "n_origins": int(fc.origin_yw.nunique()), "origin_first": str(fc.origin_yw.min()), "origin_last": str(fc.origin_yw.max())}
    return {"targets": targets, "groups": None, "config_labels": CONFIG_LABELS, "seg_labels": SEG_LABELS, "seg_order": SEG_ORDER, "decision_seg": DECISION_SEG}


# ----------------------------------------------------------------------------- calibration vs press releases
def calibration(nat: pd.DataFrame) -> dict:
    rows = []
    for w, v in OFFICIAL.items():
        if w in nat.index and not pd.isna(nat.loc[w, "covid_oe"]):
            x = nat.loc[w]
            rows.append({"yw": w, "official": v, "nidss_oe": int(x.covid_oe), "out": int(x.covid_out), "rods": int(x.covid_rods), "alt": int(x.covid_oe_alt),
                         "diff_nidss_pct": round(100 * (x.covid_oe - v) / v, 1), "diff_alt_pct": round(100 * (x.covid_oe_alt - v) / v, 1)})
    d = pd.DataFrame(rows)
    return {"rows": rows, "mean_diff_nidss_pct": r(d.diff_nidss_pct.mean(), 1), "mean_diff_alt_pct": r(d.diff_alt_pct.mean(), 1),
            "min_diff_alt_pct": r(d.diff_alt_pct.min(), 1), "max_diff_alt_pct": r(d.diff_alt_pct.max(), 1)}


# ----------------------------------------------------------------------------- report
def pct_impr(a, b):
    return round(100 * (1 - a / b))


def build_report(bt: dict, latest: dict, meta: dict, calib: dict) -> str:
    t = bt["targets"][PRIMARY]; seg = t["segments"][DECISION_SEG]
    lb = {x["config"]: x for x in seg["leaderboard"]}
    best = lb[t["best"]]; zs = lb.get("tfm_expanding"); naive = lb["naive"]; ma3 = lb["ma3"]; sn = lb.get("snaive"); ets = lb.get("ets")
    wh = seg["wis_by_h"]; impr = [pct_impr(a, b) for a, b in zip(wh[t["best"]], wh["naive"])]
    dh = seg["dir_hit_by_h"][t["best"]]
    yr_best = {s: t["segments"][s]["leaderboard"][0] for s in ["y2024", "y2025", "y2026"] if s in t["segments"]}
    alt = bt["targets"].get("covid_oe_alt"); alt_best = alt["segments"][DECISION_SEG]["leaderboard"][0] if alt else None
    li = {x["key"]: x for x in latest["indicators"]}; f1 = li[PRIMARY]["forecast"]; st = latest["status"]
    gen = meta["generated_at"][:10]
    fmt = lambda v, nd=0: "—" if v is None else f"{v:,.{nd}f}"
    cov = meta["coverage"]
    cal_rows = "".join(f"<tr><td>{x['yw']}</td><td class='num'>{x['official']:,}</td><td class='num'>{x['nidss_oe']:,}</td><td class='num'>{x['diff_nidss_pct']:+.1f}%</td><td class='num'>{x['out']:,}</td><td class='num'>{x['rods']:,}</td><td class='num'>{x['alt']:,}</td><td class='num'>{x['diff_alt_pct']:+.1f}%</td></tr>" for x in calib["rows"])
    lb_rows = "".join(f"<tr class='{'best' if x['config'] == t['best'] else ''} {'baseline' if x['kind'] == 'baseline' else ''}'><td>{x['label']}</td><td class='num'>{fmt(x['wis'])}</td><td class='num'>{x['rel_naive']:.2f}</td><td class='num'>{x['mape']:.1f}</td><td class='num'>{x['cov80']:.2f}</td><td class='num'>{x['dir_hit']:.2f}</td></tr>"
                      for x in seg["leaderboard"] if x["config"] != "snaive")
    yr_rows = "".join(f"<tr><td>{SEG_LABELS[s]}</td><td>{b['label']}</td><td class='num'>{fmt(b['wis'])}</td><td class='num'>{b['rel_naive']:.2f}</td><td class='num'>{t['segments'][s]['n_origins']}</td></tr>" for s, b in yr_best.items())
    head = page_head("報告 · 新冠 · 台灣重點傳染病預測", "covid", "report") + '<main class="report">\n'
    body = f"""<div class="hero-kicker">EVALUATION REPORT · {gen}</div>
<h1 class="report-title">以 TimesFM 3.0 零樣本預測台灣新冠門急診就診：<em>回測解讀與評估</em></h1>
<p class="lead">資料自 2021w27（健保新冠門診申報起）至 {meta['backtest_window']['end']}，context 含 2022 年 Omicron 大流行；起點 2024w01 起每週一個共 {t['n_origins']} 個，預測未來 1–4 週。主要決策依據為全期（2024–2026）的加權區間分數（WIS），並檢查三年各段是否一致。目標序列為全國新冠門急診就診人次（NIDSS 健保門診 + 急診，疾管署新聞稿使用的指標）。</p>

<div class="kpi-strip">
  <div class="kpi"><div class="label">最佳設定 WIS（全期，h=1–4）</div><div class="value">{fmt(best['wis'])}</div><div class="sub">{best['label']}</div></div>
  <div class="kpi"><div class="label">相對 last-value naive</div><div class="value">−{round(100*(1-best['rel_naive']))}%</div><div class="sub">1–4 週分別 −{impr[0]}% / −{impr[1]}% / −{impr[2]}% / −{impr[3]}%</div></div>
  <div class="kpi"><div class="label">80% 預測區間涵蓋率</div><div class="value">{best['cov80']:.2f}</div><div class="sub">理想值 0.80；60% 區間 {best['cov60']:.2f}</div></div>
  <div class="kpi"><div class="label">方向命中率（1 週 / 4 週）</div><div class="value">{dh[0]:.2f}<small>/ {dh[3]:.2f}</small></div><div class="sub">naive 永不預測上漲：{naive['dir_hit']:.2f}</div></div>
</div>

<section><h2>1. 我們評估了什麼</h2>
<p>目標序列是<strong>全國每週新冠門急診就診人次</strong>（健保門診 + 急診），取自疾管署 NIDSS「傳染病統計資料查詢系統」的新冠病毒感染健保就診查詢；這是疾管署每週新聞稿與宣布流行期所用的指標。疾管署開放資料的健保檔只有門診與住院、RODS 檔沒有急診總人次，因此另以「開放資料門診 + RODS 急診」合成指標作對照（第 2 節），兩者都做回測。</p>
<p>模型是 Google TimesFM 3.0（330M 參數的時間序列基礎模型），<strong>零樣本、未經微調</strong>，逐層加入已知未來共變數（春節旗標、每週平日假日天數、季節相位）、過去共變數（RODS）與多變量聯合預測（門急診 + 門診 + RODS + 住院）。基準為 last-value naive、MA3、季節性 naive、AutoETS、Theta。評估指標：WIS、MAE、MAPE、MASE、80%/60% 涵蓋率、方向命中率、±10% 命中率；結果依年（2024、2025、2026）與全期報告。新冠與流感、腸病毒最大的不同是<strong>可用歷史只有 5 年、沒有固定季節週期、波峰逐年下降</strong>（2023 下半年起每年一波夏季流行加一波較小的冬春波，夏季波峰 19.6 萬 → 12.7 萬 → 7.5 萬 → 2.8 萬人次），因此回測窗不能沿用 2016–2025，起點只有 {t['n_origins']} 個，結論屬初步。</p></section>

<section><h2>2. 使用的資料與指標校驗</h2>
<table class="tbl compact"><thead><tr><th>來源</th><th>內容</th><th>粒度</th><th>完整週範圍</th></tr></thead><tbody>
<tr><td>NIDSS 傳染病統計資料查詢系統（健保門急診就診，新冠病毒感染）</td><td>門診 + 急診、門診 就診人次與就診率</td><td>週 × 全國 / 6 區 × 9 年齡層</td><td class="num">{cov['nidss']['first_complete']}–{cov['nidss']['last_complete']}</td></tr>
<tr><td>健保門診及住院就診人次統計－COVID-19（開放資料）</td><td>COVID-19 健保就診人次、健保就診總人次；門診與住院</td><td>週 × 9 年齡層 × 22 縣市</td><td class="num">{cov['nhi']['first_complete']}–{cov['nhi']['last_complete']}</td></tr>
<tr><td>急診傳染病監測統計－COVID-19（開放資料，RODS）</td><td>急診新冠就診人次（無分母）</td><td>週 × 5 年齡層 × 22 縣市</td><td class="num">{cov['rods']['first_complete']}–{cov['rods']['last_complete']}</td></tr>
<tr><td>RODS 急診總人次（流感面板 RODS_RS.csv）</td><td>急診就診總人次，用來算新冠急診就診百分比</td><td>週 × 醫院</td><td class="num">{cov['rods_total']['first']}–{cov['rods_total']['last']}</td></tr>
<tr><td>NIDSS 新冠併發重症（法定傳染病統計）</td><td>週確定病例數（發病週；2024/9/1 新病例定義）</td><td>週 × 全國</td><td class="num">{cov.get('severe', {}).get('first_complete', '—')}–{cov.get('severe', {}).get('last_complete', '—')}（再保守多排 {cov.get('severe', {}).get('lag_weeks', 3)} 週）</td></tr>
<tr><td>社區合約實驗室呼吸道 PCR（RESP_LAB，流感面板）</td><td>SARS-CoV-2 週陽性數、總陽性率</td><td>週 × 全國</td><td class="num">{cov.get('resp', {}).get('first', '—')}–{cov.get('resp', {}).get('last', '—')}</td></tr>
<tr><td>行事曆</td><td>平日假日天數、春節（tw_holiday.csv）</td><td>週</td><td class="num">2016–2026</td></tr>
</tbody></table>
<div class="card"><h3>NIDSS 門急診、「門診 + RODS」合成指標 與疾管署新聞稿公布值（首報）的比較</h3>
<table class="tbl compact"><thead><tr><th>週</th><th class="num">新聞稿門急診</th><th class="num">NIDSS 門急診</th><th class="num">差</th><th class="num">開放資料門診</th><th class="num">RODS</th><th class="num">門診 + RODS</th><th class="num">差</th></tr></thead><tbody>{cal_rows}</tbody></table>
<p class="source">新聞稿為首報值，NIDSS 與開放資料為查詢當日（{gen}）的回補後數值。NIDSS 門急診平均比首報高 {calib['mean_diff_nidss_pct']:+.1f}%，「門診 + RODS」平均差 {calib['mean_diff_alt_pct']:+.1f}%（範圍 {calib['min_diff_alt_pct']:+.1f}% 至 {calib['max_diff_alt_pct']:+.1f}%）。差異來自申報回補或新聞稿的急診定義，本專案每週保存首報快照（data/nidss_visits_history.csv）以後續分辨。主指標採 NIDSS 門急診（疾管署定義），合成指標並列對照。</p></div>
<p class="source"><strong>流行期判定。</strong>{RULE_TEXT}</p></section>

<section><h2>3. 主要結果</h2>
<div class="two-col">
<div class="card"><h3>各設定 WIS 依預測週數（門急診，全期）</h3><div class="chart-box"><canvas id="c-wis"></canvas></div><p class="source">WIS 越低越好；灰色虛線為基準模型。</p></div>
<div class="card"><h3>最佳設定 1 週前預測 vs 實際（2024–2026）</h3><div class="chart-box"><canvas id="c-ts"></canvas></div><p class="source">虛線為預測中位數，淺綠帶為 80% 預測區間。</p></div>
</div>
<table class="tbl compact"><thead><tr><th>設定（門急診，全期 h = 1–4 平均）</th><th class="num">WIS</th><th class="num">相對 naive</th><th class="num">MAPE %</th><th class="num">80% 涵蓋</th><th class="num">方向命中</th></tr></thead><tbody>{lb_rows}</tbody></table>
<ul class="findings">
<li><strong>零樣本已勝過統計基準。</strong>不加共變數的 TimesFM，WIS 比 last-value naive 低 {round(100*(1-zs['rel_naive'])) if zs else '—'}%、比 MA3 低 {pct_impr(zs['wis'], ma3['wis']) if zs else '—'}%{('、比 AutoETS 低 %d%%' % pct_impr(zs['wis'], ets['wis'])) if (zs and ets) else ''}。{('季節性 naive 的 WIS 是 naive 的 %.1f 倍：新冠沒有固定季節週期，去年同週沒有參考價值。' % sn['rel_naive']) if sn else ''}</li>
<li><strong>最佳設定：{best['label']}</strong>，相對 naive 在 1–4 週分別改善 {impr[0]}% / {impr[1]}% / {impr[2]}% / {impr[3]}%；方向命中率 1 週前 {dh[0]:.2f}、4 週前 {dh[3]:.2f}。</li>
<li><strong>校準。</strong>80% 區間涵蓋率 {best['cov80']:.2f}、60% 涵蓋率 {best['cov60']:.2f}；naive 為 {naive['cov80']:.2f}。涵蓋率略高於名目值表示區間偏寬，可作區間校正。</li>
<li><strong>各年一致。</strong></li>
</ul>
<table class="tbl compact"><thead><tr><th>分段</th><th>最佳設定</th><th class="num">WIS</th><th class="num">相對 naive</th><th class="num">起點數</th></tr></thead><tbody>{yr_rows}</tbody></table>
{f'<p class="source">合成指標「開放資料門診 + RODS」的最佳設定：{alt_best["label"]}，WIS {fmt(alt_best["wis"])}（相對 naive {alt_best["rel_naive"]:.2f}）。</p>' if alt_best else ''}
</section>

<section><h2>4. 怎麼解讀</h2>
<ul class="findings">
<li><strong>1–2 週最可靠。</strong>WIS 隨 horizon 上升（{fmt(wh[t['best']][0])} → {fmt(wh[t['best']][3])}）。新冠一波之內每週可變動 30–100%，4 週的預測只做規劃參考。</li>
<li><strong>看「續升機率」而非單點。</strong>每週頁以 9 個分位數推得「下週高於本週」與「高於暫行進入／脫離水準」的機率；在疾管署流行閾值確認前，不以單一數值判定流行期。</li>
<li><strong>變異株是模型看不到的。</strong>每一波由新變異株驅動，零樣本模型只從就診人次的形狀外推；波的起點通常慢 1–2 週才被模型捕捉，上升期的 1 週預測最有價值。</li>
<li><strong>2022 年在 context 內。</strong>單週 88 萬人次的 Omicron 大流行留在歷史裡；回測顯示含與不含（context 自 2023w12）結果相當，保留較長歷史。</li>
</ul></section>

<section><h2>5. 限制與注意事項</h2>
<ul class="findings">
<li>回測只有 2024–2026 三個夏季波與兩個冬春波，{t['n_origins']} 個起點；每週預測存檔（forecast_log）累積的事前表現才是最終依據。</li>
<li>NIDSS 與開放資料為回補後數值，回測對「首報」情境偏樂觀；首報快照累積後可估回補係數。</li>
<li>流行閾值數值尚未取得；暫行規則的參數（連續 3 週上升、低點 2 倍、峰值一半）為工作假設，確認後改依官方定義。</li>
<li>重症序列自 2024w34 起（新病例定義），只做敘事；舊定義（2023/3/20–2024/8/31）未納入。</li>
<li>TimesFM 3.0 權重為非商業、非生產用途授權。</li>
</ul></section>

<section><h2>6. 最新預測摘要（起點 {li[PRIMARY]['origin_yw']}，{li[PRIMARY]['origin_date']} 當週）</h2>
<p>門急診未來 4 週中位數：{'、'.join(fmt(x['median']) for x in f1)} 人次；第 1 週 80% 區間 {fmt(f1[0]['q10'])}–{fmt(f1[0]['q90'])}。依暫行規則目前{'處於' if st['in_period'] else '不在'}流行期。完整內容見<a href="index.html">每週預測</a>。</p></section>

<footer class="report-footer">資料：NIDSS 傳染病統計資料查詢系統、疾病管制署資料開放平台（健保門診新冠、RODS 急診新冠）、RODS_RS.csv（急診總人次）、NIDSS 新冠併發重症。模型：google/timesfm-3.0-pytorch（非商業授權）。產生時間 {meta['generated_at']}。圖表依《疫情資料視覺化指引》v1.1。</footer>
</main>
"""
    script = """<script>
document.addEventListener('DOMContentLoaded', async () => {
  const bt = await (await fetch('data/backtest.json')).json();
  const t = bt.targets['covid_oe']; const seg = t.segments[bt.decision_seg];
  const cfgs = [t.best, 'tfm_cov_cny_hol', 'tfm_expanding', 'ets', 'ma3', 'naive'].filter((c, i, a) => a.indexOf(c) === i && seg.wis_by_h[c]);
  EPI.linesChart(document.getElementById('c-wis'), EPI.byHorizonSeries(seg, cfgs, 'wis_by_h', bt.config_labels), {yLabel: 'WIS', xLabel: '預測週數（h）', fmt: v => v.toLocaleString('zh-TW', {maximumFractionDigits: 0})});
  EPI.backtestTsChart(document.getElementById('c-ts'), t.timeseries.h1, {unit: t.unit, decimals: t.decimals});
});
</script>
</body></html>"""
    return head + body + script


def build_method(bt: dict, latest: dict, meta: dict, calib: dict) -> str:
    """docs/covid/method.html"""
    cov = meta["coverage"]; li = {x["key"]: x for x in latest["indicators"]}; st = latest["status"]
    t = bt["targets"][PRIMARY]; seg = {x["config"]: x for x in t["segments"][DECISION_SEG]["leaderboard"]}; best = seg[t["best"]]
    rng = lambda k: f"{cov[k]['first_complete']}–{cov[k]['last_complete']}" if k in cov and "first_complete" in cov[k] else (f"{cov[k]['first']}–{cov[k]['last']}" if k in cov else "—")
    s1 = (table(["指標", "定義與來源", "預測設定", "最新起點"], [[IND[k]["label"] + ("（主指標）" if k == PRIMARY else ""), IND[k]["source"], li[k]["config"], li[k]["origin_yw"]]
                                                       for k in ["covid_oe", "covid_out", "covid_rods", "covid_rods_pct", "covid_inp", "covid_oe_alt"] if k in li]) +
          '<p class="source">主指標為 NIDSS 查詢系統的健保門診 + 急診就診人次，即疾管署新聞稿與宣布流行期使用的「門急診就診人次」。開放資料的健保檔沒有急診、RODS 檔沒有分母，因此「開放資料門診 + RODS 急診」合成指標並列對照。</p>')
    s2 = table(["來源", "內容", "粒度", "完整週範圍"], [
        ["NIDSS 傳染病統計資料查詢系統（健保門急診，新冠病毒感染）", "門診 + 急診、門診 就診人次與就診率", "週 × 全國／6 區 × 9 年齡層", rng("nidss")],
        ["健保門診及住院就診人次統計－COVID-19（開放資料）", "健保就診人次、健保就診總人次；門診與住院", "週 × 9 年齡層 × 22 縣市", rng("nhi")],
        ["急診傳染病監測統計－COVID-19（開放資料，RODS）", "急診新冠就診人次（無分母）", "週 × 5 年齡層 × 22 縣市", rng("rods")],
        ["RODS 急診總人次（類流感面板 RODS_RS.csv）", "急診新冠% 的分母", "週 × 醫院", rng("rods_total")],
        ["NIDSS 新冠併發重症", "週確定病例數（發病週；2024/9/1 新病例定義）", "週 × 全國", rng("severe")],
        ["社區合約實驗室呼吸道 PCR（RESP_LAB）", "SARS-CoV-2 週陽性數、總陽性率（敘事用）", "週 × 全國", rng("resp")],
        ["行事曆", "平日假日天數、春節（tw_holiday.csv）", "週", "2016–2026"]]) + ul([
        "NIDSS 門急診與重症每週以程式抓取，每次抓到的值留存於 history 檔，日後用來估計首報與回補的差距。",
        "完整週判定：本週尚未結束的週一律不用；人次序列在下降期每週可跌 20–40%，不用比值測試，改以同一來源的分母（健保總就診人次、RODS 急診總人次）是否完整來把關。RODS 2022w20–w34 只有零星資料，自動排除。",
        f"與疾管署新聞稿首報值比較：NIDSS 門急診平均高 {calib['mean_diff_nidss_pct']:+.1f}%，「門診 + RODS」平均差 {calib['mean_diff_alt_pct']:+.1f}%；校驗表見<a href=\"report.html\">報告</a>第 2 節。"])
    s3 = table(["項目", "設定"], [
        ["模型", "Google TimesFM 3.0，零樣本、未微調（MLX 後端）"],
        ["context", "擴張視窗，自 2021w27（健保新冠門診申報起，含 2022 年 Omicron 大流行）；含 RODS 的設定自 2022w35"],
        ["預測", "未來 1–4 週，每週中位數與 9 個分位數"],
        ["每週流程設定", "門急診（主指標）單變量 + 春節 + 假日天數；門診與 RODS 二變量聯合 + 同樣共變數；急診新冠%、住院、合成指標單變量"],
        ["回測最佳設定", f"{best['label']}（全期 2024–2026 的 WIS 為 naive 的 {best['rel_naive']:.2f}）；多變量聯合因 RODS 只能自 2022w35 起算、失去較長 context 而無增益"],
        ["回測", f"起點 2024w01 起至最新完整週共 {t['n_origins']} 個，所有設定同一組起點，h = 1–4；依年（2024、2025、2026）與全期分段"],
        ["基準模型", "last-value naive、MA3、季節性 naive、AutoETS、Theta"]])
    s4 = ul(["<b>流行閾值數值尚未取得。</b>疾管署新聞稿只稱「疫情達流行閾值，進入流行期」（例如 2026 年第 28 週門急診 4,766 人次、連續第 6 週上升時宣布），未公布數值；記錄於 <code>covid_forecast/data/thresholds.csv</code>（待確認）。",
             f"<b>暫行規則</b>（2026-10-13 決定）：{RULE_TEXT}",
             f"目前{'處於' if st['in_period'] else '不在'}流行期；{'脫離水準（本波峰值一半）' if st['in_period'] else '進入水準（本波低點 2 倍）'} {st['level']:,.0f} 人次。每週頁扇形圖上的灰色虛線即此暫行水準，不用紅色以免與官方閾值混淆。"])
    s5 = ul(reading_items(THRESHOLD_PROB_LINE + " 新冠以門急診對暫行進入／脫離水準計算，並標示「暫行」。"))
    s6 = ul([f"回測只有 2024–2026 三個夏季波與兩個冬春波（{t['n_origins']} 個起點），結論屬初步；每週預測存檔累積的事前表現才是最終依據。",
             "NIDSS 與開放資料為回補後數值，回測對「首報」情境偏樂觀。",
             "每一波由新變異株驅動，零樣本模型只從就診人次的形狀外推；波的起點通常慢 1–2 週才被模型捕捉。",
             "重症序列自 2024w34 起（新病例定義），只做敘事；舊定義（2023/3/20–2024/8/31）未納入。",
             "TimesFM 3.0 權重為非商業、非生產用途授權。"])
    return method_page("covid", "方法 · 新冠 · 台灣重點傳染病預測", "Method", "新冠預測的方法與資料",
                       "說明每週新冠預測用了哪些資料、模型怎麼設定、在官方閾值確認前如何判定流行期，以及預測輸出該怎麼讀。回測成績見<a href=\"backtest.html\">回測</a>，完整解讀見<a href=\"report.html\">報告</a>。",
                       [("預測目標與指標", "Targets", s1), ("資料來源與完整週", "Data", s2), ("模型與設定", "Model", s3),
                        ("流行閾值與流行期", "Thresholds", s4), ("預測輸出怎麼讀", "How to read", s5), ("限制", "Limitations", s6)],
                       f"資料：NIDSS 傳染病統計資料查詢系統、疾病管制署資料開放平台、RODS_RS.csv、NIDSS 新冠併發重症。模型：{meta['model']['checkpoint']}（{meta['model']['license']}）。產生時間 {meta['generated_at']}。")


def main(config_labels: dict[str, str]):
    DATA.mkdir(parents=True, exist_ok=True)
    nat = load_national()
    cov = json.loads((PROCESSED_DIR / "coverage.json").read_text(encoding="utf-8"))
    now = dt.datetime.now().astimezone().isoformat(timespec="minutes")
    calib = calibration(nat)
    latest = build_latest(nat, config_labels); latest["generated_at"] = now
    prim = next(i for i in latest["indicators"] if i["key"] == PRIMARY); st = latest["status"]
    lvl_label = ("暫行脫離水準" if st["in_period"] else "暫行進入水準") + f" {st['level']:,.0f} 人次"
    latest["threshold_outlook"] = threshold_outlook(prim["forecast"], st["level"], prim["last_observed"], "門急診", lvl_label, "人次", 0, provisional=True)
    latest["narrative"] = build_narrative(nat, latest, calib)
    bt = build_backtest(); bt["generated_at"] = now
    best = bt["targets"][PRIMARY]["best"]
    end = str(nat[PRIMARY].dropna().index[-1])
    meta = {"generated_at": now, "coverage": cov, "calibration": calib,
            "backtest_window": {"start": "202127", "end": end, "origin_start": "202401", "origins": bt["targets"][PRIMARY]["n_origins"]},
            "model": {"name": "TimesFM 3.0", "checkpoint": "google/timesfm-3.0-pytorch", "license": "timesfm-non-commercial-license-v1.0（非商業、非生產用途）"},
            "best_config": best, "best_label": CONFIG_LABELS.get(best, best), "rule": RULE_TEXT, "thresholds_by_year": latest["status"]["thresholds_by_year"]}
    (DATA / "latest.json").write_text(json.dumps(latest, ensure_ascii=False), encoding="utf-8")
    (DATA / "backtest.json").write_text(json.dumps(bt, ensure_ascii=False), encoding="utf-8")
    (DATA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (DOCS_DIR / "report.html").write_text(build_report(bt, latest, meta, calib), encoding="utf-8")
    (DOCS_DIR / "method.html").write_text(build_method(bt, latest, meta, calib), encoding="utf-8")
    print("site data written:", {p.name: p.stat().st_size // 1024 for p in DATA.glob('*.json')}, "KB; best", best)
    print("headline:", latest["narrative"]["headline"])
