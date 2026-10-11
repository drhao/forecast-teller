#!/usr/bin/env python
"""
fetch_nidss.py — 從疾管署「傳染病統計資料查詢系統」法定傳染病頁抓登革熱本土確定病例（鄉鎮 × 日），
作為每日即時預警的 as-of 快照（2026-10-11 起含屏東縣）。

來源：https://nidss.cdc.gov.tw/nndss/disease?id=061 的進階查詢（ASP.NET 表單 POST，需先 GET 取得
__RequestVerificationToken 與 cookie；鄉鎮代碼由 /NidssQuery/GetTowns?city= 取得）。查詢結果的 Highcharts
資料以 hcJson.push({...}) 內嵌在回傳 HTML，三條序列：確定病例數、檢驗中病例數、已排除病例數。

資料定義（2026-10-11 與使用者確認，詳見 PLAN §15）：
  - 網站的「衛生局收到日」= 個案檔的「通報日」（2024 年高雄四區逐區比對完全一致）。
  - 網站的縣市 / 鄉鎮依「居住地」（同上比對確認）。使用者的規則是「感染地優先、缺則居住地」，
    但網站不提供感染地；2024 年個案檔中感染地缺漏 239/284 列、與居住地不同僅 4 列，故即時與回測一律用居住地。
  - 以「發病日」查到的序列 = 今天看得到的「依發病日計的確定病例」，正是回測 asof_adj 模式的起點當日快照；
    「通報日」序列一併保存作紀錄與日後回補研究。

輸出（都是長格式、只存非零列；已排除病例的列占九成，所以依用途裁剪）：
  data/nidss_latest.csv        本次快照：確定 / 檢驗中有值的列（全期間）＋ 已排除只保留最近 90 天
                               欄位 fetched_on, date_type(onset|report), county, town, date, confirmed, pending, excluded
  data/nidss_latest_meta.json  抓取時間、日期範圍、鄉鎮清單、請求數
  data/nidss_history.csv       每次抓取的累積紀錄：只存最近 180 天內確定 / 檢驗中有值的列（這段才會回補變動），
                               同一 fetched_on 重抓會取代，進 git

用法：python scripts/fetch_nidss.py [--start 2023-01-01] [--end 今天] [--pause 0.4]
"""
from __future__ import annotations

import argparse
import datetime as dt
import http.cookiejar
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dengue_ewarn import DATA_DIR  # noqa: E402

BASE = "https://nidss.cdc.gov.tw"
PAGE = f"{BASE}/nndss/disease?id=061"
CITIES = {"台南市": ("4", "05"), "高雄市": ("5", "07"), "屏東縣": ("5", "43")}   # 名稱 → (區域代碼, 縣市代碼)；屏東 2026-10-11 加入
DATE_TYPES = {"onset": "3", "report": "1"}               # 發病日、衛生局收到日（= 通報日）
SERIES = {"確定病例數": "confirmed", "檢驗中病例數": "pending", "已排除病例數": "excluded"}
LATEST, META, HISTORY = DATA_DIR / "nidss_latest.csv", DATA_DIR / "nidss_latest_meta.json", DATA_DIR / "nidss_history.csv"
_HC = re.compile(r"hcJson\.push\((\{.*?\})\);", re.S)
_TOK = re.compile(r'id="SearchForm".*?name="__RequestVerificationToken"[^>]*value="([^"]+)"', re.S)


class Nidss:
    def __init__(self, pause: float = 0.4):
        self.pause = pause
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.op.addheaders = [("User-Agent", "Mozilla/5.0 (research script; forecast-teller dengue_ewarn)")]
        self.token = _TOK.search(self._get(PAGE)).group(1)
        self.n_requests = 1

    def _get(self, url: str, ajax: bool = False) -> str:
        h = {"Referer": PAGE, "X-Requested-With": "XMLHttpRequest"} if ajax else {}
        with self.op.open(urllib.request.Request(url, headers=h), timeout=60) as r:
            return r.read().decode("utf-8")

    def towns(self, city_code: str) -> dict[str, str]:
        self.n_requests += 1
        items = json.loads(self._get(f"{BASE}/NidssQuery/GetTowns?city={city_code}", ajax=True))
        return {x["Text"]: x["Value"] for x in items if x["Value"] and x["Text"] != "其他"}

    def daily(self, area: str, city: str, town_code: str, town: str, date_type: str, start: dt.date, end: dt.date) -> dict[str, list]:
        """One township, one date type, 本土確定病例，年月日粒度 → {"dates": [...], "confirmed": [...], "pending": [...], "excluded": [...]}."""
        form = [("__RequestVerificationToken", self.token), ("pty_Q", "H"), ("pty_disease", "061"),
                ("pty_period_value", "ymd"), ("pty_date_type_value", date_type), ("pty_immigration_value", "0"),
                ("pty_y_s", str(start.year)), ("pty_m_s", str(start.month)), ("pty_d_s", str(start.day)), ("pty_w_s", "1"),
                ("pty_y_e", str(end.year)), ("pty_m_e", str(end.month)), ("pty_d_e", str(end.day)), ("pty_w_e", "52"),
                ("pty_area_value", area), ("pty_area_text", ""), ("pty_city_value", city), ("pty_city_text", ""),
                ("pty_town_value", town_code), ("pty_town_text", town), ("pty_sickclass_value", "determined_cnt"),
                ("StartTime", start.isoformat()), ("EndTime", end.isoformat())]
        for attempt in range(3):
            try:
                time.sleep(self.pause)
                req = urllib.request.Request(PAGE, data=urllib.parse.urlencode(form).encode(), headers={"Referer": PAGE})
                with self.op.open(req, timeout=120) as r:
                    body = r.read().decode("utf-8", "replace")
                self.n_requests += 1
                charts = [json.loads(x.replace("undefined", "null")) for x in _HC.findall(body)]
                chart = next(c for c in charts if c.get("xAxis_categories"))
                out = {"dates": [str(x) for x in chart["xAxis_categories"]]}
                for s in chart["series"]:
                    if s["name"] in SERIES:
                        out[SERIES[s["name"]]] = [int(v or 0) for v in s["data"]]
                missing = [k for k in SERIES.values() if k not in out]
                if missing:
                    raise RuntimeError(f"缺少序列 {missing}")
                return out
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"  重試 {town} {date_type}: {e}", file=sys.stderr)
                time.sleep(2 + 3 * attempt)
                self.token = _TOK.search(self._get(PAGE)).group(1)


def fetch_all(start: dt.date, end: dt.date, pause: float, fetched_on: str) -> tuple[pd.DataFrame, dict]:
    api = Nidss(pause)
    rows, towns_all = [], []
    for county, (area, city) in CITIES.items():
        towns = api.towns(city)
        print(f"{county}: {len(towns)} 區", flush=True)
        for town, code in towns.items():
            towns_all.append(f"{county}|{town}")
            for dkey, dval in DATE_TYPES.items():
                r = api.daily(area, city, code, town, dval, start, end)
                for i, d in enumerate(r["dates"]):
                    c, p, x = r["confirmed"][i], r["pending"][i], r["excluded"][i]
                    if c or p or x:
                        rows.append({"fetched_on": fetched_on, "date_type": dkey, "county": county, "town": town,
                                     "date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "confirmed": c, "pending": p, "excluded": x})
            print(f"  {town} ok", end="\r", flush=True)
        print()
    df = pd.DataFrame(rows, columns=["fetched_on", "date_type", "county", "town", "date", "confirmed", "pending", "excluded"])
    meta = {"fetched_at": dt.datetime.now().astimezone().isoformat(timespec="minutes"), "fetched_on": fetched_on,
            "start": start.isoformat(), "end": end.isoformat(), "towns": towns_all, "n_towns": len(towns_all),
            "n_requests": api.n_requests, "source": PAGE, "case_type": "本土確定病例（居住地；衛生局收到日 = 通報日）",
            "date_types": {"onset": "發病日（as-of 快照，模型輸入）", "report": "衛生局收到日（紀錄用）"}}
    return df, meta


def slim_latest(df: pd.DataFrame, end: dt.date, excluded_days: int = 90) -> pd.DataFrame:
    cut = (end - dt.timedelta(days=excluded_days)).isoformat()
    keep = (df.confirmed > 0) | (df.pending > 0) | (df.date >= cut)
    return df[keep].reset_index(drop=True)


def history_rows(df: pd.DataFrame, end: dt.date, window_days: int = 180) -> pd.DataFrame:
    cut = (end - dt.timedelta(days=window_days)).isoformat()
    return df[((df.confirmed > 0) | (df.pending > 0)) & (df.date >= cut)].reset_index(drop=True)


def append_history(df: pd.DataFrame, fetched_on: str) -> int:
    old = pd.read_csv(HISTORY, dtype=str) if HISTORY.exists() else pd.DataFrame(columns=df.columns)
    old = old[old["fetched_on"] != fetched_on]
    allr = pd.concat([old, df.astype(str)], ignore_index=True)
    allr.to_csv(HISTORY, index=False, encoding="utf-8-sig")
    return len(allr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01", help="序列起日（需 ≥ context 730 天 + 延遲 120 天）")
    ap.add_argument("--end", default=None, help="序列迄日（預設今天）")
    ap.add_argument("--pause", type=float, default=0.4)
    args = ap.parse_args()
    today = dt.date.today()
    start, end = dt.date.fromisoformat(args.start), (dt.date.fromisoformat(args.end) if args.end else today)
    fetched_on = today.isoformat()
    t0 = time.time()
    df, meta = fetch_all(start, end, args.pause, fetched_on)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    slim_latest(df, end).to_csv(LATEST, index=False, encoding="utf-8-sig")
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    n_hist = append_history(history_rows(df, end), fetched_on)
    on = df[df.date_type == "onset"]
    recent = on[on.date >= (end - dt.timedelta(days=13)).isoformat()]
    print(f"抓取完成：{meta['n_towns']} 區 × 2 種日期，{meta['n_requests']} 次請求，{time.time()-t0:.0f} 秒；{start}–{end}")
    print(f"非零列 {len(df)}；近 14 天（發病日）確定 {int(recent.confirmed.sum())}、檢驗中 {int(recent.pending.sum())}、已排除 {int(recent.excluded.sum())}")
    for county, g in on.groupby("county"):
        last = g[g.confirmed > 0].date.max()
        print(f"  {county}: {start.year} 起確定 {int(g.confirmed.sum())}，最近一例發病日 {last}")
    print(f"寫入 {LATEST.name}、{META.name}；{HISTORY.name} 共 {n_hist} 列")
    return 0


if __name__ == "__main__":
    sys.exit(main())
