#!/usr/bin/env python
"""Assemble outputs/backtest/REPORT.md and figures from saved backtest results.

  python scripts/make_report.py --tags nhi_out_ili_layer1 nhi_out_ili_layer2 --title "全國類流感門診人次"
"""
import argparse, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller import OUTPUT_DIR  # noqa: E402
from forecast_teller.backtest import SEGMENTS, summarize, threshold_note  # noqa: E402
from forecast_teller.report import leaderboard, plot_fan_over_time, plot_wis_by_horizon, wis_by_horizon_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--title", default="")
ap.add_argument("--name", default=None, help="report file stem (default: first tag)")
ap.add_argument("--target", default="nhi_out_ili")
ap.add_argument("--threshold", type=float, default=None, help="epidemic threshold for threshold hit rate (default: 75th percentile of actuals)")
args = ap.parse_args()

bt = OUTPUT_DIR / "backtest"; figs = OUTPUT_DIR / "figures"; figs.mkdir(parents=True, exist_ok=True)
results = pd.concat([pd.read_parquet(bt / f"{t}_forecasts.parquet") for t in args.tags], ignore_index=True)
results = results[results.target == args.target]
results = results.drop_duplicates(subset=["config", "origin_yw", "h"])  # baselines may repeat across suites
summary = summarize(results, thresholds={args.target: args.threshold} if args.threshold is not None else None)
stem = args.name or args.tags[0]
summary.to_csv(bt / f"{stem}_summary_combined.csv", index=False)

n_orig = results.groupby("config")["origin_yw"].nunique().max()
first, last = results["origin_yw"].min(), results["origin_yw"].max()
lines = [f"# 回測報告：{args.title or results['target'].iloc[0]}", "",
         f"- 目標序列：`{results['target'].iloc[0]}`；資料窗 2016w01–2025w53；起點 {n_orig} 個（最後 context 週 {first}–{last}）；h = 1–4",
         f"- 分段：COVID 前 = 目標週落在 2018–2019、COVID 期 = 2020–2022、COVID 後 = 2023–2025",
         "- WIS 越低越好；rel_wis = 相對季節性 naive 的 WIS（< 1 表示優於季節性 naive）；mape 為百分比；cov80 / cov60 = 80% / 60% 區間涵蓋率（理想 0.80 / 0.60）", ""]
seg_names = {"post_covid": "COVID 後（2023–2025，主要決策依據）", "pre_covid": "COVID 前（2018–2019）", "covid": "COVID 期（2020–2022）", "all": "全期（2018–2025）"}
for seg in ["post_covid", "pre_covid", "covid", "all"]:
    lines += [f"## {seg_names[seg]}：h = 1–4 平均", "", leaderboard(summary, seg).round(3).to_markdown(), ""]
    lines += [f"### WIS 依 horizon（{seg}）", "", wis_by_horizon_table(summary, seg).round(0).to_markdown(), ""]
    plot_wis_by_horizon(summary, seg, figs / f"{stem}_wis_{seg}.png", title=f"{args.title} WIS by horizon ({seg})")
    lines += [f"![]( ../figures/{stem}_wis_{seg}.png)", ""]
# yearly table for h=1 and h=4 (post-covid focus)
yr = results.assign(year=results.target_year).groupby(["config", "year", "h"])["wis"].mean().unstack("year")
lines += ["## 各年 WIS（h = 1 與 h = 4）", ""]
for h in (1, 4):
    lines += [f"### h = {h}", "", yr.xs(h, level="h").round(0).to_markdown(), ""]
best = leaderboard(summary, "post_covid").index[0]
# hit rates
thr_val = summary["thr"].iloc[0]
lines += ["## 命中率（hit rate）", "",
          "定義：`dir_hit` = 方向命中率（中位數相對起點週的漲/跌方向與實際一致）；`dir_hit3` = 三分類（漲 / 持平 / 跌，±5% 為持平帶）命中率；"
          "`hit_tol10` = 中位數落在實際值 ±10% 內的比例；`hit80` = 實際值落在 q10–q90 內的比例（= 80% 涵蓋率）；"
          f"`thr_hit_rate` = 閾值命中率（敏感度：實際 ≥ 閾值的週中，中位數也 ≥ 閾值的比例），`thr_false_alarm` = 誤報率，`thr_accuracy` = 整體準確率；閾值 = {thr_val:,.1f}（{threshold_note(args.target, args.threshold)}）。",
          "last-value naive 的中位數等於起點值，方向命中率因此退化（永遠不預測上漲、三分類永遠預測持平），僅供對照。", ""]
hit_cols = ["dir_hit", "dir_hit3", "hit_tol10", "cov80", "thr_hit_rate", "thr_false_alarm", "thr_accuracy"]
for seg in ["post_covid", "all"]:
    sub = summary[(summary.segment == seg) & (summary.h.isin([1, 2, 3, 4]))]
    tab = sub.groupby("config")[hit_cols].mean().rename(columns={"cov80": "hit80"})
    tab = tab.loc[leaderboard(summary, seg).index]  # order by WIS
    lines += [f"### {seg_names[seg]}：h = 1–4 平均", "", tab.round(3).to_markdown(), ""]
    piv = sub.pivot(index="config", columns="h", values="dir_hit").loc[leaderboard(summary, seg).index]
    lines += [f"方向命中率依 horizon（{seg}）：", "", piv.round(3).to_markdown(), ""]
# diagnostics: signed bias, MAPE, high-activity weeks
d = results.copy()
d["segment"] = d.target_year.map(lambda y: next((k for k, (a, b) in SEGMENTS.items() if a <= y <= b), "other"))
d["bias_pct"] = 100 * (d["median"] - d["y"]) / d["y"]
d["ape"] = 100 * d["abs_err"] / d["y"]
focus = [c for c in [best, "naive", "ma3", "snaive"] if c in set(d.config)]
lines += ["## 診斷：偏差與 MAPE（最佳設定 vs 基準）", "",
          "### 平均帶號偏差 %（中位數 − 實際）/ 實際；正值 = 高估", "",
          d[d.config.isin(focus)].pivot_table(index=["config", "segment"], columns="h", values="bias_pct", aggfunc="mean").round(1).to_markdown(), "",
          "### MAPE %", "",
          d[d.config.isin(focus)].pivot_table(index=["config", "segment"], columns="h", values="ape", aggfunc="mean").round(1).to_markdown(), ""]
post = d[d.segment == "post_covid"]
thr = post[post.config == best]["y"].quantile(0.75)
hi = post[post.y >= thr]
lines += [f"### COVID 後高流行週（實際值 ≥ 第 75 百分位 {thr:,.0f}；{hi[hi.config == best].shape[0] // 4} 週）", "",
          "WIS：", "", hi.pivot_table(index="config", columns="h", values="wis", aggfunc="mean").round(0).to_markdown(), "",
          "偏差 %：", "", hi.pivot_table(index="config", columns="h", values="bias_pct", aggfunc="mean").round(1).to_markdown(), ""]
for h in (1, 4):
    plot_fan_over_time(results, best, h, figs / f"{stem}_fan_{best}_h{h}.png", title=f"{best}: {h}-week-ahead ({args.title})")
    lines += [f"## 最佳設定 `{best}` 的 {h} 週前預測 vs 實際（2023 起）", "", f"![]( ../figures/{stem}_fan_{best}_h{h}.png)", ""]
plot_fan_over_time(results, "snaive", 4, figs / f"{stem}_fan_snaive_h4.png", title=f"snaive: 4-week-ahead ({args.title})")
lines += ["## 對照：季節性 naive 的 4 週前預測", "", f"![]( ../figures/{stem}_fan_snaive_h4.png)", ""]
(bt / f"{stem}_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(f"report: {bt / f'{stem}_REPORT.md'}\nbest (post_covid): {best}")
print(leaderboard(summary, "post_covid").round(3).to_string())
