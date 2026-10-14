#!/usr/bin/env python
"""
forecast_now.py — 登革熱鄉鎮即時預警：以今天為起點，對台南 + 高雄 + 屏東各鄉鎮的 7 日累計本土確定病例做
14 天分布預測，換算「14 天內突破 EWARN 閾值」的機率，輸出兩級標示（注意 / 警示）。

流程（與回測 asof_adj 模式一致，PLAN §12–13）：
  1. 讀 data/nidss_latest.csv（fetch_nidss.py）發病日序列 = 今天看得到的確定病例（as-of 快照）。
  2. 完整度校正：最近 DMAX 天的計數除以歷史通報完整度 c[k]（下限 0.2），c[k] 來自個案檔的延遲分布
     （data_processed/dengue_delay_hist.npz 最後 730 天，發病日至 2025-07；假設延遲分布至今未變，待日後用
     每日快照重估）。
  3. 閾值 = max(前 3 週週均值 × 2, 3 例)，用校正後的已知資料算（部署時系統看得到的）。
  4. TimesFM 3.0 對 7 日累計序列（context 730 天）預測 14 天，9 個分位數；P(突破) 以分位數內插，
     取 14 天內最大者（與回測的 h = 0 列相同）。
  4b. 鄰近 KDE（PLAN_erakde §10.6，2026-10-14 起）：各鄉鎮近 60 天校正後計數依近期性衰減（半衰期 14 天）加總，放在
     該鄉鎮歷史病例座標的加權平均點（data/kde_geometry.csv），固定 2 km 高斯核在各村里點取強度 → 鄉鎮占比 × 全域
     總量趨勢投影，與自身持續各半（kde_mix_fix_town）→ 負二項分位數 → P(突破)。
     集成 = TimesFM 與 KDE 的「14 天內突破」機率平均（回測 ensp_tfm_kde_mix_fix_town），標示依集成機率。
  5. 標示：p ≥ alert → 警示、p ≥ watch → 注意（預設 0.3 / 0.4，第二版兩級門檻建議，集成門檻待重算）；另列「已達閾值」規則。

輸出：
  outputs/latest/dengue_alerts.csv                 每區一列（含 7 日累計、閾值、p 各 horizon、標示）
  outputs/forecast_log/dengue_forecast_log.csv     逐 origin × 區 × h（1,3,7,14 與 any）的分位數與機率，累積
  docs/dengue/latest.json                          站台「最新預警」用
用法：python scripts/forecast_now.py [--context 730] [--horizon 14] [--watch 0.3] [--alert 0.4] [--floor 3]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import DATA_DIR, OUTPUT_DIR, PROCESSED_DIR, SITE_DIR  # noqa: E402
from dengue_ewarn.alerts import prob_ge  # noqa: E402
from dengue_ewarn.data import DMAX, completeness, ewarn_threshold, rolling7  # noqa: E402

LOG_HS = [1, 3, 7, 14]
QCOLS = [f"q{q}" for q in range(10, 100, 10)]


def load_snapshot(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, dict]:
    meta = json.loads((data_dir / "nidss_latest_meta.json").read_text(encoding="utf-8"))
    df = pd.read_csv(data_dir / "nidss_latest.csv", dtype={"date": str})
    return df, meta


def matrix(df: pd.DataFrame, meta: dict, date_type: str, col: str) -> tuple[pd.DatetimeIndex, list[str], np.ndarray]:
    days = pd.date_range(meta["start"], meta["end"], freq="D")
    series = meta["towns"]
    M = np.zeros((len(series), len(days)), dtype=np.float64)
    idx = {s: i for i, s in enumerate(series)}; didx = {d.strftime("%Y-%m-%d"): j for j, d in enumerate(days)}
    sub = df[df.date_type == date_type]
    for r in sub.itertuples(index=False):
        s = f"{r.county}|{r.town}"
        if s in idx and r.date in didx:
            M[idx[s], didx[r.date]] += getattr(r, col)
    return days, series, M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=730)
    ap.add_argument("--horizon", type=int, default=14)
    ap.add_argument("--watch", type=float, default=0.3)
    ap.add_argument("--alert", type=float, default=0.4)
    ap.add_argument("--floor", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-model", action="store_true", help="只算閾值與已知資料，不跑 TimesFM（測試用）")
    ap.add_argument("--no-kde", action="store_true", help="不算鄰近 KDE 與集成，標示只依 TimesFM（2026-10-14 前的行為）")
    ap.add_argument("--data-dir", default=None, help="改讀別的快照目錄（測試用，例如注入病例的合成快照）")
    ap.add_argument("--dry-run", action="store_true", help="只印結果，不寫 outputs / docs（測試用）")
    args = ap.parse_args()

    df, meta = load_snapshot(Path(args.data_dir) if args.data_dir else DATA_DIR)
    days, series, known = matrix(df, meta, "onset", "confirmed")
    _, _, pending = matrix(df, meta, "onset", "pending")
    S, T = known.shape; d = T - 1; origin = days[-1]
    H, L = args.horizon, args.context

    # 完整度校正（asof_adj）：最近 K 天依「距今天數 k」除以 c[k]
    hp = PROCESSED_DIR / "dengue_all_delay_hist.npz"   # 全台面板的延遲分布（含屏東）；沒有就用台南高雄的
    hz = np.load(hp if hp.exists() else PROCESSED_DIR / "dengue_delay_hist.npz")
    hist = hz["hist"]; hist_days = pd.to_datetime(hz["days"])
    c = np.maximum(completeness(hist, hist.shape[1] - 1), 0.2)                 # (K,)
    K = len(c); kk = np.minimum(d - np.arange(d + 1), K - 1)
    adj = 1.0 / c[kk[-K:]] if d + 1 >= K else 1.0 / c[kk]
    known_adj = known.copy(); known_adj[:, -len(adj):] *= adj
    s7_raw = rolling7(known); s7_adj = rolling7(known_adj)
    thr = ewarn_threshold(known_adj, d, floor=args.floor)                     # (S,)
    ctxs = [s7_adj[i, max(0, d + 1 - L): d + 1].astype(np.float32) for i in range(S)]

    if args.no_model:
        Q = np.repeat(s7_adj[:, -1][:, None, None], 9, axis=2).repeat(H, axis=1)
    else:
        from dengue_ewarn.model import TimesFM3Model
        try:
            import mlx.core as mx
            mx.set_memory_limit(6 * 1024 ** 3); mx.set_cache_limit(1 * 1024 ** 3)
        except Exception:
            pass
        model = TimesFM3Model(backend="mlx", batch_size=args.batch)
        Q = np.stack([q for _, q in model.forecast(ctxs, H)])                   # (S, H, 9)
    Q = np.maximum(Q, 0)
    P_tfm = prob_ge(np.transpose(Q, (1, 0, 2)), np.broadcast_to(thr, (H, S)))  # (H, S)
    p_any_tfm = P_tfm.max(axis=0)

    # 鄰近 KDE + 集成（PLAN_erakde §10.6；幾何檔由 build_kde_geometry.py 產生，不需要個案檔）
    method = "tfm"; Qk = None; P_kde = np.full_like(P_tfm, np.nan); p_any_kde = np.full(S, np.nan)
    geo_path = DATA_DIR / "kde_geometry.csv"
    if not args.no_kde and geo_path.exists():
        from dengue_ewarn.kde import VARIANTS, KDEForecaster
        geo = pd.read_csv(geo_path); sidx = {s: i for i, s in enumerate(series)}
        tw = geo[(geo.kind == "town") & geo.series.isin(sidx)]; vg = geo[(geo.kind == "village") & geo.series.isin(sidx)]
        cx = np.full(S, np.nan); cy = np.full(S, np.nan); cx[tw.series.map(sidx).to_numpy()] = tw.x_km.to_numpy(); cy[tw.series.map(sidx).to_numpy()] = tw.y_km.to_numpy()
        pts = {"cx": cx, "cy": cy, "ex": vg.x_km.to_numpy(float), "ey": vg.y_km.to_numpy(float), "es": vg.series.map(sidx).to_numpy(int), "eshare": vg.w.to_numpy(float)}
        fc = KDEForecaster(pts, hist, S, VARIANTS["kde_mix_fix_town"])
        Qk = np.maximum(fc.forecast(d, "asof_adj", known_adj, H), 0)                     # (S, H, 9)
        P_kde = prob_ge(np.transpose(Qk, (1, 0, 2)), np.broadcast_to(thr, (H, S))); p_any_kde = P_kde.max(axis=0)
        method = "ensp_tfm_kde_mix_fix_town"
        print(f"鄰近 KDE：{len(tw)} 區有代表點、{len(vg)} 個村里評估點；{int((~np.isfinite(cx)).sum())} 區無歷史病例（只用自身持續）")
    if method == "tfm":
        P, p_any = P_tfm, p_any_tfm
    else:
        P = (P_tfm + P_kde) / 2; p_any = (p_any_tfm + p_any_kde) / 2             # 機率平均（與回測 ensp 相同：各自取 14 天內最大再平均）
    tier = np.where(p_any >= args.alert, "警示", np.where(p_any >= args.watch, "注意", ""))
    pend14 = pending[:, -14:].sum(axis=1); conf14 = known[:, -14:].sum(axis=1)
    last_case = [days[np.flatnonzero(known[i] > 0)[-1]].strftime("%Y-%m-%d") if (known[i] > 0).any() else None for i in range(S)]

    rows = []
    for i, s in enumerate(series):
        county, town = s.split("|")
        rows.append({"origin": origin.strftime("%Y-%m-%d"), "county": county, "town": town,
                     "known7": int(s7_raw[i, -1]), "adj7": round(float(s7_adj[i, -1]), 2), "thr": round(float(thr[i]), 2),
                     "confirmed14": int(conf14[i]), "pending14": int(pend14[i]), "last_onset": last_case[i],
                     "rule_now": bool(s7_raw[i, -1] >= thr[i]),
                     "p_h7": round(float(P[6, i]), 3), "p_h14": round(float(P[H - 1, i]), 3), "p_any14": round(float(p_any[i]), 3),
                     "p_any14_tfm": round(float(p_any_tfm[i]), 3), "p_any14_kde": round(float(p_any_kde[i]), 3) if np.isfinite(p_any_kde[i]) else None,
                     "median_h7": round(float(Q[i, 6, 4]), 2), "q90_h7": round(float(Q[i, 6, 8]), 2),
                     "median_h14": round(float(Q[i, H - 1, 4]), 2), "q90_h14": round(float(Q[i, H - 1, 8]), 2),
                     "kde_median_h7": round(float(Qk[i, 6, 4]), 2) if Qk is not None else None, "kde_median_h14": round(float(Qk[i, H - 1, 4]), 2) if Qk is not None else None,
                     "method": method, "tier": tier[i]})
    tab = pd.DataFrame(rows).sort_values(["p_any14", "known7"], ascending=False).reset_index(drop=True)
    if args.dry_run:
        print(f"[dry-run] 起點 {origin.date()}：注意 {int((tier == '注意').sum())}、警示 {int((tier == '警示').sum())}；最高 p {p_any.max():.2f}")
        print(tab.head(15)[["county", "town", "known7", "adj7", "thr", "p_any14", "p_any14_tfm", "p_any14_kde", "median_h14", "kde_median_h14", "tier"]].to_string(index=False))
        return 0
    out_dir = OUTPUT_DIR / "latest"; out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_dir / "dengue_alerts.csv", index=False, encoding="utf-8-sig")

    # forecast log（累積，key = origin × series × h）
    run_at = dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M%z")
    lrows = []
    for i, s in enumerate(series):
        for h in LOG_HS + [0]:
            if h:
                q = Q[i, h - 1, :]; lrows.append({"run_at": run_at, "origin": origin.strftime("%Y-%m-%d"), "series": s, "h": h,
                                                   "target": (origin + pd.Timedelta(days=h)).strftime("%Y-%m-%d"), "known7": int(s7_raw[i, -1]), "adj7": round(float(s7_adj[i, -1]), 2),
                                                   "thr": round(float(thr[i]), 2), "median": round(float(q[4]), 2), **{c_: round(float(q[j]), 2) for j, c_ in enumerate(QCOLS)}, "prob": round(float(P[h - 1, i]), 3),
                                                   "prob_tfm": round(float(P_tfm[h - 1, i]), 3), "prob_kde": round(float(P_kde[h - 1, i]), 3) if np.isfinite(P_kde[h - 1, i]) else np.nan,
                                                   "kde_median": round(float(Qk[i, h - 1, 4]), 2) if Qk is not None else np.nan, "method": method})
            else:
                lrows.append({"run_at": run_at, "origin": origin.strftime("%Y-%m-%d"), "series": s, "h": 0, "target": (origin + pd.Timedelta(days=H)).strftime("%Y-%m-%d"),
                              "known7": int(s7_raw[i, -1]), "adj7": round(float(s7_adj[i, -1]), 2), "thr": round(float(thr[i]), 2), "median": np.nan, **{c_: np.nan for c_ in QCOLS}, "prob": round(float(p_any[i]), 3),
                              "prob_tfm": round(float(p_any_tfm[i]), 3), "prob_kde": round(float(p_any_kde[i]), 3) if np.isfinite(p_any_kde[i]) else np.nan, "kde_median": np.nan, "method": method})
    log_dir = OUTPUT_DIR / "forecast_log"; log_dir.mkdir(parents=True, exist_ok=True); lp = log_dir / "dengue_forecast_log.csv"
    new = pd.DataFrame(lrows)
    if lp.exists():
        old = pd.read_csv(lp); old = old[~((old.origin == new.origin.iloc[0]))]
        if "method" not in old.columns:      # rows logged before 2026-10-14: TimesFM only
            old["method"] = "tfm"; old["prob_tfm"] = old["prob"]; old["prob_kde"] = np.nan; old["kde_median"] = np.nan
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(lp, index=False, encoding="utf-8-sig")

    # 站台 JSON
    flagged = tab[tab.tier != ""]
    latest = {"origin": origin.strftime("%Y-%m-%d"), "generated_at": run_at, "fetched_at": meta["fetched_at"], "n_series": S,
              "params": {"context_days": L, "horizon_days": H, "watch_p": args.watch, "alert_p": args.alert, "threshold": f"max(2 × 前 3 週週均值, {args.floor:g} 例)",
                         "mode": "asof_adj", "method": method,
                         "method_text": ("TimesFM 3.0（asof_adj）與鄰近 KDE（鄉鎮代表點、固定 2 km、與自身持續各半）的 14 天內突破機率平均（回測 ensp_tfm_kde_mix_fix_town）" if method != "tfm" else "TimesFM 3.0（asof_adj）"),
                         "completeness_window": f"{hist_days[max(0, hist.shape[1]-1-730)].date()}–{hist_days[-1].date()}（個案檔延遲分布）",
                         "completeness_recent7": [round(float(x), 2) for x in c[:7]]},
              "totals": {"confirmed14": int(conf14.sum()), "pending14": int(pend14.sum()), "known7": int(s7_raw[:, -1].sum()),
                         "n_watch": int((tier == "注意").sum()), "n_alert": int((tier == "警示").sum()), "n_rule_now": int(tab.rule_now.sum()),
                         "max_p_any14": round(float(p_any.max()), 3), "max_p_any14_tfm": round(float(p_any_tfm.max()), 3),
                         "max_p_any14_kde": round(float(np.nanmax(p_any_kde)), 3) if np.isfinite(p_any_kde).any() else None},
              "by_county": [{"county": cty, "confirmed14": int(g.confirmed14.sum()), "pending14": int(g.pending14.sum()), "known7": int(g.known7.sum()),
                             "n_watch": int((g.tier == "注意").sum()), "n_alert": int((g.tier == "警示").sum())} for cty, g in tab.groupby("county")],
              "flagged": flagged.to_dict(orient="records"), "table": tab.to_dict(orient="records"),
              "notes": ["資料：NIDSS 法定傳染病查詢，本土確定病例，依居住地鄉鎮，衛生局收到日 = 通報日；以發病日序列作為起點當日可得快照。",
                        "網站不提供感染地；個案檔中感染地缺漏多且與居住地僅零星不同，故沿用居住地（與回測一致）。",
                        "完整度校正的延遲分布來自個案檔（發病日至 2025-07），假設至今未變；待每日快照累積後改用快照重估。",
                        "機率為 9 個分位數內插的近似值（尾端夾在 0.02–0.98）。"]
              + (["標示依 TimesFM 與鄰近 KDE 的機率平均（2026-10-14 起）：KDE 把各鄉鎮近 60 天的校正後病例放在該鄉鎮歷史病例座標的加權平均點，以 2 km 高斯核估各區的鄰近強度；"
                  "即時資料只有鄉鎮層級，回測顯示與個案座標版的差異可忽略（PLAN_erakde §10.6）。注意 / 警示門檻沿用 TimesFM 版的建議值，集成版的工作量表待重算。"] if method != "tfm" else [])}
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print(f"起點 {origin.date()}（抓取 {meta['fetched_at']}）：{S} 區；近 14 天確定 {int(conf14.sum())}、檢驗中 {int(pend14.sum())}；"
          f"注意 {latest['totals']['n_watch']}、警示 {latest['totals']['n_alert']}、已達閾值 {latest['totals']['n_rule_now']}；最高 p(14 天內突破) {p_any.max():.2f}")
    print(tab.head(10)[["county", "town", "known7", "adj7", "thr", "pending14", "p_h7", "p_any14", "p_any14_tfm", "p_any14_kde", "median_h14", "q90_h14", "tier"]].to_string(index=False))
    print(f"寫入 {out_dir/'dengue_alerts.csv'}、{lp}（{len(new)} 列）、{SITE_DIR/'latest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
