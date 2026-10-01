#!/usr/bin/env python
"""
log_forecast.py — 把每週的即時預測（outputs/latest/latest_forecast*.csv）累積到
outputs/forecast_log/forecast_log.csv，供日後與實際值比對（scripts/evaluate_forecasts.py）。

- 每列一個 (mode, target, origin_yw, target_yw)；同一組合若再次寫入（例如同週重跑）
  以最新一次取代，並記錄 run_at（預測產生時間）。
- 另外加入「門急診合計」nhi_oe_ili（mode = joint_sum）：門診 + 急診兩個聯合預測的
  中位數與分位數同向相加，與 build_site.combine_out_er 的作法一致。
- --from-git：從 git 歷史回填所有曾經 commit 過的 outputs/latest 快照（run_at 取 commit 時間）。
- --project ev_forecast：改處理子專案（ev_forecast/outputs/latest → ev_forecast/outputs/forecast_log）。

用法：
  python scripts/log_forecast.py             # 記錄目前 outputs/latest/
  python scripts/log_forecast.py --from-git  # 回填 git 歷史後再記錄目前的
  python scripts/log_forecast.py --project ev_forecast [--from-git]
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # repo 根目錄（git 指令在此執行）
PROJECT = ""                                   # "" = 流感主線；"ev_forecast" = 子專案（main 會覆寫）
LATEST = ROOT / "outputs" / "latest"
LOG_DIR = ROOT / "outputs" / "forecast_log"
LOG = LOG_DIR / "forecast_log.csv"
FILES = ["latest_forecast_joint.csv", "latest_forecast.csv"]
QCOLS = [f"q{q}" for q in range(10, 100, 10)]
KEY = ["mode", "target", "origin_yw", "target_yw"]
COLS = ["run_at", "mode", "target", "label", "origin_yw", "target_yw", "target_week_start", "h", "median", *QCOLS]
TZ = timezone(timedelta(hours=8))


def _read(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text.lstrip("﻿")), dtype={"origin_yw": str, "target_yw": str})


def add_combined(df: pd.DataFrame) -> pd.DataFrame:
    """門急診合計 = 聯合門診 + 聯合急診（同起點、同目標週）；分位數同向相加。"""
    o = df[(df["mode"] == "joint") & (df["target"] == "nhi_out_ili")]
    e = df[(df["mode"] == "joint") & (df["target"] == "nhi_er_ili")]
    if o.empty or e.empty:
        return df
    m = o.merge(e, on=["origin_yw", "target_yw", "target_week_start", "h"], suffixes=("", "_er"))
    if m.empty:
        return df
    c = m[["origin_yw", "target_yw", "target_week_start", "h"]].copy()
    c["mode"] = "joint_sum"
    c["target"] = "nhi_oe_ili"
    c["label"] = "全國門急診類流感就診人次（門診 + 急診）"
    c["median"] = m["median"] + m["median_er"]
    for q in QCOLS:
        c[q] = m[q] + m[f"{q}_er"]
    return pd.concat([df, c], ignore_index=True)


def snapshot_rows(texts: list[str], run_at: str) -> pd.DataFrame:
    df = pd.concat([_read(t) for t in texts if t and t.strip()], ignore_index=True)
    df = add_combined(df)
    df["run_at"] = run_at
    return df[COLS]


def current_snapshot() -> pd.DataFrame:
    texts = [(LATEST / f).read_text(encoding="utf-8-sig") for f in FILES if (LATEST / f).exists()]
    if not texts:
        raise SystemExit(f"找不到 {LATEST}/latest_forecast*.csv，請先跑 forecast_now.py")
    mtime = max((LATEST / f).stat().st_mtime for f in FILES if (LATEST / f).exists())
    run_at = datetime.fromtimestamp(mtime, TZ).strftime("%Y-%m-%dT%H:%M+08:00")
    return snapshot_rows(texts, run_at)


def git_snapshots() -> list[pd.DataFrame]:
    rel = f"{PROJECT}/outputs/latest" if PROJECT else "outputs/latest"
    out = subprocess.run(["git", "log", "--format=%H %cI", "--", *[f"{rel}/{f}" for f in FILES]],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    frames = []
    for line in reversed([l for l in out if l.strip()]):  # 舊 → 新，後者覆蓋前者
        sha, when = line.split()
        texts = []
        for f in FILES:
            r = subprocess.run(["git", "show", f"{sha}:{rel}/{f}"], cwd=ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                texts.append(r.stdout)
        if texts:
            run_at = datetime.fromisoformat(when).astimezone(TZ).strftime("%Y-%m-%dT%H:%M+08:00")
            frames.append(snapshot_rows(texts, run_at))
            print(f"git {sha[:7]} {run_at}: {len(frames[-1])} 列，起點 {sorted(frames[-1]['origin_yw'].unique())}")
    return frames


def merge_into_log(new_frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, int, int]:
    old = pd.read_csv(LOG, dtype={"origin_yw": str, "target_yw": str}) if LOG.exists() else pd.DataFrame(columns=COLS)
    before = len(old)
    allf = pd.concat([f for f in [old, *new_frames] if not f.empty], ignore_index=True)
    allf = allf.sort_values("run_at", kind="stable").drop_duplicates(subset=KEY, keep="last")
    allf = allf.sort_values(["origin_yw", "mode", "target", "h"], kind="stable").reset_index(drop=True)
    return allf, before, len(allf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-git", action="store_true", help="先從 git 歷史回填所有 outputs/latest 快照")
    ap.add_argument("--project", default="", help="子專案目錄名（如 ev_forecast）；預設為流感主線")
    args = ap.parse_args()
    global PROJECT, LATEST, LOG_DIR, LOG
    PROJECT = args.project.strip("/")
    base = ROOT / PROJECT if PROJECT else ROOT
    LATEST = base / "outputs" / "latest"
    LOG_DIR = base / "outputs" / "forecast_log"
    LOG = LOG_DIR / "forecast_log.csv"
    frames = git_snapshots() if args.from_git else []
    cur = current_snapshot()
    frames.append(cur)
    log, before, after = merge_into_log(frames)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log.to_csv(LOG, index=False, encoding="utf-8-sig")
    origins = sorted(log["origin_yw"].unique())
    print(f"本次快照 run_at {cur['run_at'].iloc[0]}，起點 {sorted(cur['origin_yw'].unique())}，{len(cur)} 列")
    print(f"forecast_log.csv：{before} → {after} 列；起點週 {origins[0]}–{origins[-1]}（共 {len(origins)} 個）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
