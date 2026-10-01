#!/usr/bin/env python
"""
evaluate_forecasts.py — 把 outputs/forecast_log/forecast_log.csv 的歷次即時預測與
data_processed/national_weekly.csv 的實際值對上，評估「真正事前」的預測表現。

輸出：
  outputs/forecast_log/evaluation.csv          逐列：預測 + 實際 + 誤差（abs_err、ape、in60、in80、wis）
  outputs/forecast_log/evaluation_summary.csv  依 mode × target × h 彙整（MAE、MAPE、涵蓋率、WIS、n）
只評估目標週已在面板「完整週」內的列；實際值可能因回補而變動，summary 會記錄評估日期。

用法：python scripts/evaluate_forecasts.py [--since 202636] [--quiet] [--project ev_forecast]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forecast_teller.metrics import wis, covered  # noqa: E402

QCOLS = [f"q{q}" for q in range(10, 100, 10)]


def actuals(panel: Path) -> pd.DataFrame:
    nat = pd.read_csv(panel, dtype={"yw": str}).set_index("yw")
    if {"nhi_out_ili", "nhi_er_ili"} <= set(nat.columns):
        nat["nhi_oe_ili"] = nat["nhi_out_ili"] + nat["nhi_er_ili"]
    return nat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="只評估此起點週以後（含）的預測，例如 202636")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--project", default="", help="子專案目錄名（如 ev_forecast）；預設為流感主線")
    args = ap.parse_args()
    base = ROOT / args.project.strip("/") if args.project else ROOT
    log_dir = base / "outputs" / "forecast_log"
    panel = base / "data_processed" / "national_weekly.csv"

    log = pd.read_csv(log_dir / "forecast_log.csv", dtype={"origin_yw": str, "target_yw": str})
    if args.since:
        log = log[log["origin_yw"] >= args.since]
    nat = actuals(panel)

    log = log[log["target"].isin(nat.columns)].copy()
    log["actual"] = [nat[t].get(w, np.nan) for t, w in zip(log["target"], log["target_yw"])]
    ev = log.dropna(subset=["actual"]).copy()
    if ev.empty:
        print("目前沒有任何目標週已有實際值，無可評估的列。")
        return 0
    q = ev[QCOLS].to_numpy(float)
    y = ev["actual"].to_numpy(float)
    ev["abs_err"] = np.abs(y - ev["median"].to_numpy(float))
    ev["ape"] = ev["abs_err"] / np.where(y == 0, np.nan, np.abs(y))
    ev["in60"] = covered(y, q, 60)
    ev["in80"] = covered(y, q, 80)
    ev["wis"] = wis(y, q)
    ev["evaluated_on"] = date.today().isoformat()
    ev = ev.sort_values(["target", "mode", "origin_yw", "h"]).reset_index(drop=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ev.to_csv(log_dir / "evaluation.csv", index=False, encoding="utf-8-sig")

    summ = (ev.groupby(["mode", "target", "h"])
              .agg(n=("actual", "size"), mae=("abs_err", "mean"), mape=("ape", "mean"),
                   cov60=("in60", "mean"), cov80=("in80", "mean"), wis=("wis", "mean"))
              .reset_index())
    summ["mape"] = 100 * summ["mape"]
    summ["evaluated_on"] = date.today().isoformat()
    summ.to_csv(log_dir / "evaluation_summary.csv", index=False, encoding="utf-8-sig")

    if not args.quiet:
        pd.set_option("display.width", 160)
        print(f"可評估 {len(ev)} 列（{log['actual'].isna().sum()} 列目標週尚無實際值）；"
              f"起點 {ev['origin_yw'].min()}–{ev['origin_yw'].max()}")
        show = ev[["mode", "target", "origin_yw", "target_yw", "h", "median", "actual", "ape", "in80", "wis"]].copy()
        show["ape"] = (100 * show["ape"]).round(1)
        print(show.round(1).to_string(index=False))
        print("\n彙整（mode × target × h）：")
        print(summ.round({"mae": 1, "mape": 1, "cov60": 2, "cov80": 2, "wis": 1}).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
