#!/usr/bin/env python
"""Isotonic calibration (leave-one-year-out) of the 'crossing within 14 days' probability, calibrated p* sweep,
lead times, and the two-tier (watch / alert) workload table. Writes CSVs + dengue_CALIBRATION_REPORT.md."""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import OUTPUT_DIR, PROCESSED_DIR  # noqa: E402
from dengue_ewarn.alerts import auc, episodes  # noqa: E402
from dengue_ewarn.calibration import Isotonic, brier, ece, episode_leads, logloss, loyo_calibrate, reliability, sweep, two_tier_table, weekly_table  # noqa: E402
from dengue_ewarn.data import ewarn_threshold, rolling7  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="dengue"); ap.add_argument("--mode", default="asof_adj"); ap.add_argument("--n-sites", type=int, default=20)
ap.add_argument("--window", type=int, default=14)
args = ap.parse_args()
bt = OUTPUT_DIR / "backtest"
res = pd.read_parquet(bt / f"{args.tag}_{args.mode}_forecasts.parquet"); res["origin"] = pd.to_datetime(res.origin)
rows = res[(res.h == 0) & (res.model == "tfm")].copy().reset_index(drop=True)
years = sorted(rows.year.unique())

# --- 1) leave-one-year-out isotonic calibration of the per-origin probability
rows["prob_cal"] = loyo_calibrate(rows, "prob", "event", "year", method="isotonic")
rows["prob_platt"] = loyo_calibrate(rows, "prob", "event", "year", method="platt")
iso_all = Isotonic().fit(rows.prob.to_numpy(), rows.event.to_numpy(float))            # deployable map (all years)
grid = np.round(np.arange(0, 1.0001, 0.01), 2)
pd.DataFrame({"raw": grid, "calibrated": iso_all.predict(grid)}).to_csv(bt / f"{args.tag}_calibration_map.csv", index=False)
y = rows.event.to_numpy(float)
metrics = pd.DataFrame([{"version": "raw", "brier": brier(rows.prob, y), "logloss": logloss(rows.prob, y), "ece": ece(rows.prob, y), "auc": auc(rows.event.to_numpy(bool), rows.prob.to_numpy())},
                        {"version": "isotonic (LOYO)", "brier": brier(rows.prob_cal, y), "logloss": logloss(rows.prob_cal, y), "ece": ece(rows.prob_cal, y), "auc": auc(rows.event.to_numpy(bool), rows.prob_cal.to_numpy())},
                        {"version": "Platt (LOYO)", "brier": brier(rows.prob_platt, y), "logloss": logloss(rows.prob_platt, y), "ece": ece(rows.prob_platt, y), "auc": auc(rows.event.to_numpy(bool), rows.prob_platt.to_numpy())}])
per_year = pd.DataFrame([{"year": yr, "n": int((rows.year == yr).sum()), "event_rate": float(rows.event[rows.year == yr].mean()),
                          "brier_raw": brier(rows.prob[rows.year == yr], y[rows.year == yr]), "brier_cal": brier(rows.prob_cal[rows.year == yr], y[rows.year == yr]),
                          "ece_raw": ece(rows.prob[rows.year == yr], y[rows.year == yr]), "ece_cal": ece(rows.prob_cal[rows.year == yr], y[rows.year == yr])} for yr in years])
rel_raw, rel_cal = reliability(rows.prob, y), reliability(rows.prob_cal, y)
rel = rel_raw.merge(rel_cal, on="bin", suffixes=("_raw", "_cal"))
metrics.to_csv(bt / f"{args.tag}_calibration_metrics.csv", index=False); rel.to_csv(bt / f"{args.tag}_calibration_bins.csv", index=False); per_year.to_csv(bt / f"{args.tag}_calibration_by_year.csv", index=False)
rows[["series", "origin", "year", "thr", "prob", "prob_cal", "prob_platt", "event"]].to_parquet(bt / f"{args.tag}_{args.mode}_calibrated.parquet")

# --- 2) weekly sweeps (raw vs calibrated) and lead times with calibrated probabilities
pgrid = np.round(np.arange(0.1, 0.91, 0.05), 2)
wk_raw, wk_cal = weekly_table(rows, "prob"), weekly_table(rows, "prob_cal")
sw_raw, sw_cal = sweep(wk_raw, pgrid).assign(version="raw"), sweep(wk_cal, pgrid).assign(version="calibrated")
counts = pd.read_parquet(PROCESSED_DIR / "dengue_township_daily.parquet"); dates = counts.index; series = list(counts.columns)
C = counts.to_numpy().T.astype(float); S7 = rolling7(C)
thr_final = np.stack([ewarn_threshold(C, t) for t in range(C.shape[1])], axis=1)
ep = [(s_i, t) for s_i in range(C.shape[0]) for t in episodes(S7[s_i], thr_final[s_i]) if dates[t].year in years and 6 <= dates[t].month <= 12]
lk_raw = dict(zip(zip(rows.series, rows.origin), rows.prob)); lk_cal = dict(zip(zip(rows.series, rows.origin), rows.prob_cal))
for name, lk, sw in [("raw", lk_raw, sw_raw), ("calibrated", lk_cal, sw_cal)]:
    det, med, ge3 = [], [], []
    for p in pgrid:
        lv = episode_leads(ep, lk, series, dates, p, args.window); det.append(np.mean(~np.isnan(lv))); med.append(np.nanmedian(lv) if np.any(~np.isnan(lv)) else np.nan); ge3.append(np.nanmean(lv >= 3) if np.any(~np.isnan(lv)) else np.nan)
    sw["events_detected"], sw["lead_median"], sw["lead_ge3"] = det, med, ge3
sweeps = pd.concat([sw_raw, sw_cal], ignore_index=True); sweeps.to_csv(bt / f"{args.tag}_sweep_calibrated.csv", index=False)

# --- 3) two-tier table (calibrated probabilities)
pairs = [(pw, pa) for pw in (0.2, 0.3, 0.4) for pa in (0.4, 0.5, 0.6) if pa > pw]
tiers = two_tier_table(wk_raw, ep, lk_raw, series, dates, pairs, n_sites=args.n_sites, window=args.window)   # raw probabilities (see §1: recalibration does not help)
tiers.to_csv(bt / f"{args.tag}_two_tier.csv", index=False)
# cost-loss guidance: p* = C/L
cl = pd.DataFrame([{"C_over_L": r_, "p_star": r_, **sw_raw.iloc[(np.abs(sw_raw.p_star - r_)).argmin()][["sensitivity", "false_alarm_per100", "ppv", "events_detected", "lead_median"]].to_dict()} for r_ in (0.1, 0.2, 0.3, 0.5)])
cl.to_csv(bt / f"{args.tag}_cost_loss.csv", index=False)

# --- 4) markdown
f2 = lambda v: "—" if pd.isna(v) else f"{v:.2f}"
lines = ["# 機率校準與兩級門檻工作量（asof_adj，TimesFM 3.0）", "",
         f"樣本：{len(rows):,} 個（鄉鎮 × 起點），事件率 {y.mean():.3f}；校準 = 等張回歸，留一年交叉驗證（每年的校準機率都由其他年份的資料擬合）。", "",
         "## 1. 校準前後", "", metrics.round(4).to_markdown(index=False), "", "各年（Brier / ECE，越低越好）：", "", per_year.round(3).to_markdown(index=False), "",
         "可靠度（10 個機率區間）：", "", rel[["bin", "n_raw", "mean_pred_raw", "observed_raw", "mean_pred_cal", "observed_cal"]].round(3).to_markdown(index=False), "",
         "## 2. 校準後的 p* 掃描（鄉鎮週）與前置時間", "", sw_cal[["p_star", "alert_rate_per_site_week", "sensitivity", "false_alarm_per100", "ppv", "events_detected", "lead_median", "lead_ge3"]].round(3).to_markdown(index=False), "",
         "校準前（同一 p* 的原始機率）：", "", sw_raw[["p_star", "alert_rate_per_site_week", "sensitivity", "false_alarm_per100", "ppv", "events_detected", "lead_median"]].round(3).to_markdown(index=False), "",
         f"## 3. 兩級門檻工作量（原始機率；示例轄區 {args.n_sites} 站）", "",
         "注意 = 機率 ≥ watch_p，只在儀表板標示；警示 = 機率 ≥ alert_p，產生查證工作單。watch_earlier_than_alert = 有警示的事件中，注意標示比警示更早出現的比例；days_watch_to_alert_median_if_earlier = 這些事件從注意到警示的天數中位數；watch_only_events_detected = 只有注意、沒有升級成警示的事件比例。", "",
         tiers.round(3).to_markdown(index=False), "", "## 4. 成本損失法的預設門檻（p* = C / L）", "", cl.round(3).to_markdown(index=False), ""]
(bt / f"{args.tag}_CALIBRATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:14])); print("...\n"); print(sw_cal[["p_star", "sensitivity", "false_alarm_per100", "ppv", "events_detected", "lead_median"]].round(3).to_string(index=False)); print("\n", tiers[["watch_p", "alert_p", f"watch_flags_week_{args.n_sites}", f"alert_orders_week_{args.n_sites}", f"false_orders_week_{args.n_sites}", "watch_sensitivity", "alert_sensitivity", "alert_ppv", "events_detected_watch", "events_detected_alert", "lead_median_watch", "lead_median_alert", "watch_earlier_than_alert", "days_watch_to_alert_median_if_earlier", "watch_only_events_detected"]].round(2).to_string(index=False))
