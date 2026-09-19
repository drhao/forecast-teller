#!/usr/bin/env python
"""Print markdown tables of the backtest results (for PLAN.md §12)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from ev_forecast.site import CONFIG_LABELS, TAGS, load_summary  # noqa: E402

for target in TAGS:
    s = load_summary(target)
    post = s[(s.segment == "post_covid") & (s.h <= 4)]
    lb = post.groupby("config")[["wis", "mape", "cov80", "dir_hit", "thr_hit_rate", "thr_false_alarm"]].mean()
    lb["rel_naive"] = lb["wis"] / lb.loc["naive", "wis"]
    byh = post.pivot_table(index="config", columns="h", values="wis")
    rel_h = byh.div(byh.loc["naive"])
    lb = lb.sort_values("wis")
    print(f"\n### {target}: COVID 後（2023–2025），h = 1–4 平均\n")
    print("| 設定 | WIS | 相對 naive | h1 / h2 / h3 / h4 相對 naive | MAPE % | 80% 涵蓋 | 方向命中 | 閾值命中 / 誤報 |")
    print("|---|---|---|---|---|---|---|---|")
    for c, r in lb.iterrows():
        rh = " / ".join(f"{rel_h.loc[c, h]:.2f}" for h in (1, 2, 3, 4))
        thr = f"{r.thr_hit_rate:.2f} / {r.thr_false_alarm:.2f}" if pd.notna(r.thr_hit_rate) else "—"
        print(f"| {CONFIG_LABELS.get(c, c)} | {r.wis:,.0f} | {r.rel_naive:.2f} | {rh} | {r.mape:.1f} | {r.cov80:.2f} | {r.dir_hit:.2f} | {thr} |")
    pre = s[(s.segment == "pre_covid") & (s.h <= 4)].groupby("config")["wis"].mean().sort_values()
    cov = s[(s.segment == "covid") & (s.h <= 4)].groupby("config")["wis"].mean().sort_values()
    print(f"\nCOVID 前最佳：{CONFIG_LABELS.get(pre.index[0], pre.index[0])} WIS {pre.iloc[0]:,.0f}（naive {pre['naive']:,.0f}，相對 {pre.iloc[0]/pre['naive']:.2f}）；"
          f"COVID 期最佳：{CONFIG_LABELS.get(cov.index[0], cov.index[0])} WIS {cov.iloc[0]:,.0f}（naive {cov['naive']:,.0f}，相對 {cov.iloc[0]/cov['naive']:.2f}）")
