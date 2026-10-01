#!/usr/bin/env python
"""Charts for the presentation decks (flu + enterovirus), drawn with the epidemic dataviz style guide palette.

Outputs (outputs/slides/):
  flu: wis_by_horizon.png, backtest_h1.png, latest_fan.png, county_rel.png
  ev : ev_wis_by_horizon.png, ev_backtest_h1.png, ev_latest_fan.png
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller import OUTPUT_DIR, ROOT  # noqa: E402

P = {300: "#B4C9B1", 500: "#739A6D", 600: "#5D7F58", 800: "#374C34"}
N = {200: "#E4E7E4", 300: "#CACFC9", 400: "#A2ABA0", 500: "#7A8778", 600: "#5D675B", 700: "#444C43", 900: "#181B18"}
LINE = {"primary": "#5D7F58", "blue": "#587A9D", "yellow": "#A8821F", "teal": "#356B70", "alert": "#BE373C"}
OUT = OUTPUT_DIR / "slides"; OUT.mkdir(parents=True, exist_ok=True)

avail = {f.name for f in font_manager.fontManager.ttflist}
fam = [f for f in ["Noto Sans TC", "Noto Sans CJK TC", "PingFang TC", "PingFang HK", "Heiti TC", "Hiragino Sans", "Arial Unicode MS"] if f in avail] + ["DejaVu Sans"]
plt.rcParams.update({"font.family": fam, "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": N[200], "grid.linewidth": 0.6, "axes.edgecolor": N[300], "axes.labelcolor": N[700],
                     "xtick.color": N[700], "ytick.color": N[700], "font.size": 11, "legend.frameon": False})
fmt_k = lambda v, _=None: (f"{v/1000:g}k" if v >= 1000 else f"{v:g}")  # 1.5k、12k；避免 2k/2k 重複刻度


def wis_by_horizon(seg, series, title, path):
    """series: list of (config, color, marker, linewidth, linestyle, label)."""
    fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=200); ax.grid(axis="x", visible=False)
    for cfg, color, mk, lw, ls, lab in series:
        ax.plot([1, 2, 3, 4], seg["wis_by_h"][cfg], color=color, marker=mk, markersize=6, linewidth=lw, linestyle=ls, label=lab)
    ax.set_ylim(0, None); ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["1 週前", "2 週前", "3 週前", "4 週前"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_k)); ax.set_ylabel("WIS（越低越好）")
    ax.legend(ncol=2, fontsize=9, loc="upper left")
    ax.set_title(title, loc="left", fontsize=11.5, color=N[900], fontweight="600")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def backtest_h1(rows, title, path, threshold=None, thr_label=None):
    d = pd.to_datetime([r["date"] for r in rows])
    y = np.array([r["y"] for r in rows]); med = np.array([r["median"] for r in rows]); lo = np.array([r["q10"] for r in rows]); hi = np.array([r["q90"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=200); ax.grid(axis="x", visible=False)
    ax.fill_between(d, lo, hi, color=P[300], alpha=0.35, linewidth=0, label="80% 預測區間")
    ax.plot(d, med, color=LINE["primary"], linewidth=2.2, linestyle=(0, (6, 3)), label="1 週前預測中位數")
    ax.plot(d, y, color=N[700], linewidth=1.8, label="實際值")
    if threshold:
        ax.axhline(threshold, color=LINE["alert"], linestyle="--", linewidth=1.2, label=thr_label)
    ax.set_ylim(0, None); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_k)); ax.set_ylabel("人次")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax.legend(loc="upper left", fontsize=9, ncol=4 if threshold else 3)
    ax.set_title(title, loc="left", fontsize=11.5, color=N[900], fontweight="600")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def latest_fan(ind, title, path, n_hist=60, threshold=None, thr_label=None, label_fmt=None):
    h = ind["history"][-n_hist:]; f = ind["forecast"]
    hd = pd.to_datetime([x["date"] for x in h]); hy = [x["y"] for x in h]
    fd = pd.to_datetime([ind["origin_date"]] + [x["date"] for x in f]); lo = ind["last_observed"]
    med = [lo] + [x["median"] for x in f]; q10 = [lo] + [x["q10"] for x in f]; q90 = [lo] + [x["q90"] for x in f]
    q20 = [lo] + [x["q20"] for x in f]; q80 = [lo] + [x["q80"] for x in f]
    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=200); ax.grid(axis="x", visible=False)
    ax.plot(hd, hy, color=LINE["primary"], linewidth=2.4, label="觀測值")
    ax.fill_between(fd, q10, q90, color=P[300], alpha=0.25, linewidth=0, label="80% 預測區間")
    ax.fill_between(fd, q20, q80, color=P[300], alpha=0.5, linewidth=0, label="60% 預測區間")
    ax.plot(fd, med, color=LINE["primary"], linewidth=2.4, linestyle=(0, (6, 3)), label="預測中位數")
    if threshold:
        ax.axhline(threshold, color=LINE["alert"], linestyle="--", linewidth=1.2, label=thr_label)
    ax.axvline(pd.Timestamp(ind["origin_date"]), color=N[400], linestyle="--", linewidth=1)
    ax.set_ylim(0, max(max(q90), max(hy), threshold or 0) * 1.18)
    ax.text(pd.Timestamp(ind["origin_date"]), ax.get_ylim()[1] * 0.03, f"預測起點 {ind['origin_date']} ", color=N[600], fontsize=9, va="bottom", ha="right")
    label_fmt = label_fmt or (lambda v: f"{v:,.0f}")
    for k, x in enumerate(f):  # 標籤放在點的右上 / 右下交錯，避免相鄰週重疊
        ax.annotate(label_fmt(x["median"]), (pd.Timestamp(x["date"]), x["median"]), textcoords="offset points",
                    xytext=(5, 9 if k % 2 == 0 else -15), ha="left", fontsize=8.5, color=N[700])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt_k)); ax.set_ylabel("人次")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.set_title(title, loc="left", fontsize=11.5, color=N[900], fontweight="600")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ============================================================ flu
bt = json.loads((ROOT / "docs/data/backtest.json").read_text(encoding="utf-8"))
latest = json.loads((ROOT / "docs/data/latest.json").read_text(encoding="utf-8"))
t = bt["targets"]["nhi_out_ili"]; seg = t["segments"]["post_covid"]
wis_by_horizon(seg, [(t["best"], LINE["primary"], "o", 2.6, "-", "四變量聯合＋春節＋假日（最佳）"), ("tfm_cov_cny_hol", LINE["blue"], "s", 2.0, "-", "單變量＋春節＋假日"),
                     ("tfm_expanding", LINE["yellow"], "^", 2.0, "-", "TimesFM zero-shot"), ("ets", N[400], "D", 1.5, "--", "AutoETS"),
                     ("naive", N[400], "x", 1.5, ":", "last-value naive"), ("ma3", N[400], "+", 1.5, "-.", "MA3")],
               "全國類流感門診人次：各設定的 WIS 依預測週數（COVID 後 2023–2025）", OUT / "wis_by_horizon.png")
backtest_h1(t["timeseries"]["h1"], "最佳設定 1 週前預測 vs 實際（2023–2025，每週滾動）", OUT / "backtest_h1.png")
ind = next(i for i in latest["indicators"] if i["key"] == "nhi_out_ili")
latest_fan(ind, f"全國類流感門診人次：起點 {ind['origin_yw']}，未來 4 週預測（四變量聯合＋春節＋假日）", OUT / "latest_fan.png", label_fmt=lambda v: f"{v/1000:.0f}k")

# county relative WIS (mv_cov), sorted
rel = bt["groups"]["county"]["per_group_rel"]; names = [r["name"] for r in rel][::-1]; vals = [r["mv_cov"] for r in rel][::-1]
fig, ax = plt.subplots(figsize=(4.6, 5.2), dpi=200); ax.grid(axis="y", visible=False)
ax.barh(names, vals, color=P[500], height=0.65)
ax.axvline(1.0, color=N[500], linestyle="--", linewidth=1.2); ax.text(1.0, len(names) - 0.4, " naive = 1.0", color=N[600], fontsize=9, va="center")
for i, v in enumerate(vals):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8.5, color=N[700])
ax.set_xlim(0, 1.12); ax.set_xlabel("相對 naive 的 WIS（< 1 優於 naive）"); ax.tick_params(axis="y", labelsize=9)
ax.set_title("22 縣市聯合預測（COVID 後）", loc="left", fontsize=11.5, color=N[900], fontweight="600")
fig.tight_layout(); fig.savefig(OUT / "county_rel.png"); plt.close(fig)

# ============================================================ enterovirus
ebt = json.loads((ROOT / "docs/ev/data/backtest.json").read_text(encoding="utf-8"))
elatest = json.loads((ROOT / "docs/ev/data/latest.json").read_text(encoding="utf-8"))
et = ebt["targets"]["ev_oe"]; eseg = et["segments"]["post_covid"]
wis_by_horizon(eseg, [(et["best"], LINE["primary"], "o", 2.6, "-", "三變量聯合＋學校行事曆＋假日（最佳）"), ("tfm_cov_school_hol", LINE["blue"], "s", 2.0, "-", "單變量＋學校行事曆＋假日"),
                      ("tfm_expanding", LINE["yellow"], "^", 2.0, "-", "TimesFM zero-shot"), ("ets", N[400], "D", 1.5, "--", "AutoETS"),
                      ("naive", N[400], "x", 1.5, ":", "last-value naive"), ("ma3", N[400], "+", 1.5, "-.", "MA3")],
               "腸病毒門急診合計：各設定的 WIS 依預測週數（COVID 後 2023–2025）", OUT / "ev_wis_by_horizon.png")
backtest_h1(et["timeseries"]["h1"], "最佳設定 1 週前預測 vs 實際（2023–2025，每週滾動）", OUT / "ev_backtest_h1.png",
            threshold=et["threshold"], thr_label=f"流行閾值 {et['threshold']:,.0f}")
eind = next(i for i in elatest["indicators"] if i["key"] == "ev_oe")
latest_fan(eind, f"腸病毒門急診合計：起點 {eind['origin_yw']}，未來 4 週預測（三變量聯合＋學校行事曆＋假日）", OUT / "ev_latest_fan.png",
           threshold=eind["threshold"], thr_label=eind["threshold_label"])
print("charts →", sorted(p.name for p in OUT.glob("*.png")), "| font:", fam[0])
