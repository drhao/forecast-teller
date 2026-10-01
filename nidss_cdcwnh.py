"""
nidss_cdcwnh.py — 程式化查詢疾管署「傳染病統計資料查詢系統」健保門急診就診資料
    https://nidss.cdc.gov.tw/Cdcwnh/CDCWNH01?disease=11

原理：
  該頁面沒有公開 JSON API，查詢是 ASP.NET MVC 表單 POST（帶 __RequestVerificationToken
  防偽 token，需先 GET 一次取得 token + cookie）。查詢結果的 Highcharts 資料以
  `hcJson.push({...})` 內嵌在回傳的 HTML 裡，直接用正規表示式抓出 JSON 即可。

支援的分頁：
  CDCWNH01  地區別趨勢（多地區 x 單一年齡層）
  CDCWNH02  年齡別趨勢（單一地區 x 多年齡層）
  CDCWNH03  地理分佈（區別 / 縣市別，單一年齡層；資料在 jdata1 變數）
  CDCWNH09  同期比較（同一年齡層、多年份疊圖）

用法：
  pip install requests pandas
  from nidss_cdcwnh import NIDSS
  api = NIDSS()
  df = api.area_trend(disease=11, start=(2024, 1), end=(2024, 52),
                      areas=["全國", "中區"], age="65+", case_type="門診 + 急診")
"""
from __future__ import annotations

import json
import re
import time
from typing import Iterable

import pandas as pd
import requests

BASE = "https://nidss.cdc.gov.tw"

# ---- 代碼對照表（皆從網頁 / AjaxAreaDict / AjaxAgeDict 抓下來） -------------------
DISEASE = {          # pty_ICD_CTGRY_value（URL 的 ?disease= 也用同一個代碼）
    "腸病毒": 1, "類流感": 4, "急性上呼吸道感染": 7, "腹瀉": 8,
    "猩紅熱": 9, "水痘": 10, "新冠病毒感染": 11, "紅眼症": 12,
}
CASE_TYPE = {"門診": "[1]", "住院": "[5]", "門診 + 急診": "[1][4]"}
AREA = {
    "全國": "[1][2][3][4][5][6]", "台北區": "[1]", "北區": "[2]", "中區": "[3]",
    "南區": "[4]", "高屏區": "[5]", "東區": "[6]",
}
AGE = {   # 注意：伺服器只接受這幾個既定組合，自訂年齡組合（如 [0][1][2][3][4]）會被退回首頁
    "全部": "[0][1][2][3][4][5][6][7][8][9][10][11][12][13][14][15][16][17][18][19][20]"
            "[25][30][35][40][45][50][55][60][65][70][75][80][85][99]",
    "0-2": "[0][1][2]", "3-6": "[3][4][5][6]", "7-12": "[7][8][9][10][11][12]",
    "13-15": "[13][14][15]", "16-18": "[16][17][18]", "19-24": "[19][20]",
    "25-64": "[25][30][35][40][45][50][55][60]", "65+": "[65][70][75][80][85]",
}

_TOKEN_RE = re.compile(r'name="__RequestVerificationToken"[^>]*value="([^"]+)"')
_HC_RE = re.compile(r"hcJson\.push\((\{.*?\})\);\s*\n")
_JDATA_RE = re.compile(r"var jdata1 = (\[.*?\]);")


class NIDSS:
    def __init__(self, pause: float = 0.5):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0 (research script)"
        self.pause = pause  # 對公家伺服器友善一點

    # ---- 低階：送出查詢、回傳原始 HTML ------------------------------------------
    def _post(self, page: str, disease: int, start, end, case_type: str,
              extra: list[tuple[str, str]]) -> str:
        url = f"{BASE}/Cdcwnh/{page}"
        r = self.s.get(url, params={"disease": disease}, timeout=30)
        r.raise_for_status()
        token = _TOKEN_RE.search(r.text).group(1)

        (ys, ws), (ye, we) = start, end
        dname = {v: k for k, v in DISEASE.items()}[disease]
        # 子疾病別目前每一類都只有一個選項，且代碼與大類相同
        form = [
            ("__RequestVerificationToken", token),
            ("pty_Q", "H"),                       # H = 進階查詢（可自訂起訖）
            ("pty_period_value", "yw"), ("pty_period_text", "年週"),
            ("pty_y_s", str(ys)), ("pty_m_s", ""), ("pty_d_s", ""), ("pty_w_s", f"{ws:02d}"),
            ("pty_y_e", str(ye)), ("pty_m_e", ""), ("pty_d_e", ""), ("pty_w_e", f"{we:02d}"),
            ("pty_case_type_value", CASE_TYPE[case_type]), ("pty_case_type_text", case_type),
            ("pty_ICD_CTGRY_value", str(disease)), ("pty_ICD_CTGRY_text", dname),
            ("pty_ICD_SUBCTGRY_value", str(disease)), ("pty_ICD_SUBCTGRY_text", dname),
            ("pty_Rate_Cnt_value", ""),
        ] + extra
        time.sleep(self.pause)
        r = self.s.post(f"{url}?disease={disease}", data=form,
                        headers={"Referer": f"{url}?disease={disease}"}, timeout=60)
        r.raise_for_status()
        if r.url.endswith("/Home/Index"):
            raise ValueError("伺服器拒絕此參數組合（被導回首頁），請檢查 areas/age 是否為既定選項")
        return r.text

    @staticmethod
    def _charts(html: str) -> list[dict]:
        out = []
        for raw in _HC_RE.findall(html):
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
        return out

    @staticmethod
    def _to_long(charts: list[dict], series_col: str, **const) -> pd.DataFrame:
        """兩張圖（比率 % / 人次）→ 一張長表。"""
        rows = []
        for c in charts:
            metric = "rate_pct" if "比率" in c["yAxis_title_text"] else "visits"
            for se in c["series"]:
                for x, v in zip(c["xAxis_categories"], se["data"]):
                    rows.append({**const, "period": x, series_col: se["name"],
                                 "metric": metric, "value": v})
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return (df.pivot_table(index=[*const.keys(), "period", series_col],
                               columns="metric", values="value", aggfunc="first")
                  .reset_index().rename_axis(None, axis=1))

    # ---- 高階 API ----------------------------------------------------------------
    def area_trend(self, disease: int, start: tuple[int, int], end: tuple[int, int],
                   areas: Iterable[str] = ("全國",), age: str = "全部",
                   case_type: str = "門診 + 急診") -> pd.DataFrame:
        """CDCWNH01 地區別趨勢：多個地區、單一年齡層。回傳 period / area / rate_pct / visits"""
        areas = list(areas)
        extra = [("pty_area_value", AREA[a]) for a in areas]
        extra += [("pty_area_text", ",".join(areas)),
                  ("pty_age_value", AGE[age]), ("pty_age_text", age)]
        html = self._post("CDCWNH01", disease, start, end, case_type, extra)
        return self._to_long(self._charts(html), "area",
                             disease=disease, case_type=case_type, age=age)

    def age_trend(self, disease: int, start: tuple[int, int], end: tuple[int, int],
                  area: str = "全國", ages: Iterable[str] = ("全部",),
                  case_type: str = "門診 + 急診") -> pd.DataFrame:
        """CDCWNH02 年齡別趨勢：單一地區、多個年齡層。"""
        ages = list(ages)
        extra = [("pty_area_value", AREA[area]), ("pty_area_text", area)]
        extra += [("pty_age_value", AGE[a]) for a in ages]
        extra += [("pty_age_text", ",".join(ages))]
        html = self._post("CDCWNH02", disease, start, end, case_type, extra)
        return self._to_long(self._charts(html), "age",
                             disease=disease, case_type=case_type, area=area)

    def geo(self, disease: int, start: tuple[int, int], end: tuple[int, int],
            level: str = "CITY", age: str = "全部",
            case_type: str = "門診 + 急診") -> pd.DataFrame:
        """CDCWNH03 地理分佈：level = 'CITY'（22 縣市）或 'AREA'（6 區）。
        回傳該期間的就診比率(%)；資料嵌在 jdata1 變數裡。"""
        extra = [("pty_Area_City_value", level),
                 ("pty_Area_City_text", "縣市別" if level == "CITY" else "區別"),
                 ("pty_age_value", AGE[age]), ("pty_age_text", age)]
        html = self._post("CDCWNH03", disease, start, end, case_type, extra)
        m = _JDATA_RE.search(html)
        if not m:
            return pd.DataFrame()
        data = json.loads(m.group(1).replace("'", '"'))
        df = pd.DataFrame(data).rename(columns={"code": "region", "value": "rate_pct"})
        df.insert(0, "age", age); df.insert(0, "case_type", case_type)
        df.insert(0, "disease", disease); df.insert(0, "period", f"{start}~{end}")
        return df

    def full_matrix(self, disease: int, start, end, case_type="門診 + 急診",
                    areas=tuple(AREA), ages=tuple(AGE)) -> pd.DataFrame:
        """地區 x 年齡 全矩陣：每個年齡層打一次 CDCWNH01（一次可帶全部地區）。"""
        frames = [self.area_trend(disease, start, end, areas, age, case_type) for age in ages]
        return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    api = NIDSS()
    df = api.area_trend(disease=DISEASE["新冠病毒感染"], start=(2024, 1), end=(2024, 52),
                        areas=["全國", "台北區", "中區"], age="65+")
    print(df.head(10)); print(len(df), "rows")

    df2 = api.age_trend(disease=DISEASE["類流感"], start=(2025, 40), end=(2025, 52),
                        area="全國", ages=["0-2", "3-6", "65+"])
    print(df2.head(10))

    df3 = api.geo(disease=11, start=(2025, 10), end=(2025, 10), level="CITY")
    print(df3.head())
