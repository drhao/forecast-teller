#!/usr/bin/env python
"""
fetch_resp_lab.py — 從疾管署「傳染病統計資料查詢系統」首頁（op=6）抓取
「全國每週呼吸道病原體分子生物學檢出情形」，更新 data/RESP_LAB.csv。

原理（同 nidss_cdcwnh.py）：頁面沒有 JSON API，Highcharts 資料以
`hcJson.push({...})` 內嵌在 HTML 裡；該圖未來週的值是 JavaScript 的
`undefined`（非合法 JSON），先換成 null 再解析。

輸出格式與 NIDSS 匯出的 CSV 一致：UTF-8-sig、欄位依序為
"Year-Week of Specimen Received", hMPV, Parainfluenza, SARS-CoV-2, RSV,
Rhinovirus, Adeno, Mycoplasma, Influenza, "Positive (%)"；尚無資料的週留空
（`io.read_resp_lab` 會自行丟掉）。

兩個保存機制（網站只顯示近兩年，舊週日後在網站上找不到）：
  1. data/RESP_LAB.csv 是「累積」檔：網站窗口內的週以網站值更新，窗口外的舊週保留。
  2. data/resp_lab_history.csv（長格式、進 git）：每次抓取把當次所有有值的列連同
     fetched_on 一起附加，保留每個時點看到的值與後續回補修正；同一天重抓會取代當天的列。

用法：
  python scripts/fetch_resp_lab.py            # 抓取、比對、更新 RESP_LAB.csv 與 history
  python scripts/fetch_resp_lab.py --dry-run  # 只比對、不寫檔
  python scripts/fetch_resp_lab.py --import old_export.csv 2026-09-17
                                              # 把舊的匯出檔匯入 history（不抓網站、不改 RESP_LAB.csv）
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "RESP_LAB.csv"
HISTORY = ROOT / "data" / "resp_lab_history.csv"
URL = "https://nidss.cdc.gov.tw/Home/Index?op=6"
TITLE = "全國每週呼吸道病原體分子生物學檢出情形"
COLUMNS = ["hMPV", "Parainfluenza", "SARS-CoV-2", "RSV", "Rhinovirus",
           "Adeno", "Mycoplasma", "Influenza", "Positive (%)"]
YW_COL = "Year-Week of Specimen Received"

_HC_RE = re.compile(r"hcJson\.push\((\{.*?\})\);", re.S)


def fetch_chart(url: str = URL) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read().decode("utf-8")
    for raw in _HC_RE.findall(html):
        if TITLE not in raw:
            continue
        return json.loads(raw.replace("undefined", "null"))
    raise RuntimeError(f"頁面裡找不到「{TITLE}」的 hcJson 區塊（頁面改版？）")


def chart_to_rows(chart: dict) -> list[dict]:
    weeks = chart["xAxis_categories"]
    series = {s["name"]: s["data"] for s in chart["series"]}
    missing = [c for c in COLUMNS if c not in series]
    if missing:
        raise RuntimeError(f"缺少系列：{missing}；頁面現有：{list(series)}")
    rows = []
    for i, yw in enumerate(weeks):
        row = {YW_COL: str(yw)}
        for c in COLUMNS:
            v = series[c][i] if i < len(series[c]) else None  # 系列可能比 x 軸短
            if v is None:
                row[c] = ""
            elif c == "Positive (%)":
                row[c] = f"{v:g}"
            else:
                row[c] = str(int(round(v)))
        rows.append(row)
    return rows


def read_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r[YW_COL]: r for r in csv.DictReader(f)}


def last_filled(rows: list[dict]) -> str | None:
    filled = [r[YW_COL] for r in rows if r["Influenza"] != ""]
    return filled[-1] if filled else None


def write_csv(rows: list[dict], path: Path) -> None:
    # 與 NIDSS 匯出格式一致：標題與年週加引號、數值不加引號、LF、無結尾換行
    lines = [",".join(f'"{c}"' for c in [YW_COL, *COLUMNS])]
    for r in rows:
        lines.append(",".join([f'"{r[YW_COL]}"'] + [r[c] for c in COLUMNS]))
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def merge_rows(old: dict[str, dict], new_rows: list[dict]) -> list[dict]:
    """網站窗口內以新值為準；窗口外（更早）的舊週保留。依年週排序。"""
    merged = {yw: {YW_COL: yw, **{c: r.get(c, "") for c in COLUMNS}} for yw, r in old.items()}
    for r in new_rows:
        merged[r[YW_COL]] = r
    return [merged[k] for k in sorted(merged)]


def append_history(rows: list[dict], fetched_on: str, path: Path = HISTORY) -> int:
    """把有值的列加進長格式 history；同一 fetched_on 的舊列先移除。回傳寫入列數。"""
    fields = ["fetched_on", "yw", *COLUMNS]
    kept: list[dict] = []
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["fetched_on"] != fetched_on]
    new = [{"fetched_on": fetched_on, "yw": r[YW_COL], **{c: r[c] for c in COLUMNS}}
           for r in rows if r["Influenza"] != ""]
    allrows = sorted(kept + new, key=lambda r: (r["fetched_on"], r["yw"]))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(allrows)
    return len(new)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--import", dest="import_", nargs=2, metavar=("CSV", "YYYY-MM-DD"),
                    help="把舊的 NIDSS 匯出檔匯入 history（不抓網站、不改 RESP_LAB.csv）")
    args = ap.parse_args()

    if args.import_:
        src, when = args.import_
        n = append_history(list(read_existing(Path(src)).values()), when)
        print(f"已把 {src} 的 {n} 列（fetched_on={when}）匯入 {HISTORY}")
        return 0

    chart = fetch_chart()
    rows = chart_to_rows(chart)
    new_last = last_filled(rows)
    old = read_existing(args.out)
    old_last = last_filled(list(old.values())) if old else None
    print(f"網站資料：{rows[0][YW_COL]}–{rows[-1][YW_COL]}，最後有值週 {new_last}；"
          f"現有檔案最後有值週 {old_last}")

    revised, added = [], []
    for r in rows:
        yw = r[YW_COL]
        o = old.get(yw)
        if o is None or (o["Influenza"] == "" and r["Influenza"] != ""):
            if r["Influenza"] != "":
                added.append(yw)
            continue
        diff = {c: (o[c], r[c]) for c in COLUMNS if o[c] != r[c]}
        if diff:
            revised.append((yw, diff))
    if added:
        print("新增週：", ", ".join(added))
    for yw, diff in revised:
        print(f"修正 {yw}: " + "; ".join(f"{c} {a}→{b}" for c, (a, b) in diff.items()))
    if not added and not revised:
        print("無變動。")

    merged = merge_rows(old, rows)
    kept_old = [r[YW_COL] for r in merged if r[YW_COL] < rows[0][YW_COL]]
    if kept_old:
        print(f"保留網站窗口外的舊週 {kept_old[0]}–{kept_old[-1]}（{len(kept_old)} 週）")
    if args.dry_run:
        return 0
    write_csv(merged, args.out)
    print(f"已寫入 {args.out}（{len(merged)} 列）")
    today = date.today().isoformat()
    n = append_history(rows, today)
    print(f"已把 {n} 列（fetched_on={today}）附加到 {HISTORY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
