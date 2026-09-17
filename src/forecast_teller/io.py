"""Readers for the raw surveillance CSVs in data/ (Big5 / cp950 encoded).

Every reader returns a tidy pandas DataFrame with a string 'yw' column and
normalized 'county' / 'age' labels so the sources can be aligned.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA_DIR

ENC = "cp950"

# Pre-2010 county names in NIDDS and full-width variants → current 22 counties.
COUNTY_FIX = {
    "台中市(舊)": "台中市",
    "台南市(舊)": "台南市",
    "高雄縣(舊)": "高雄市",
    "臺北市": "台北市",
    "臺中市": "台中市",
    "臺南市": "台南市",
    "臺東縣": "台東縣",
    "桃園縣": "桃園市",
}
COUNTIES = [
    "台北市", "新北市", "基隆市", "桃園市", "新竹市", "新竹縣", "宜蘭縣", "苗栗縣",
    "台中市", "彰化縣", "南投縣", "雲林縣", "嘉義市", "嘉義縣", "台南市", "高雄市",
    "屏東縣", "台東縣", "花蓮縣", "澎湖縣", "金門縣", "連江縣",
]
AGE_FIX = {"65-": "65+", "99": "unknown", "不詳": "unknown"}
AGE_ORDER = ["00", "01", "02", "03", "04", "05-09", "10-14", "15-19", "20-24", "25-29",
             "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65+", "unknown"]


def _norm_county(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace(COUNTY_FIX)


def _norm_age(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace(AGE_FIX)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")


def read_nhi(path: str | Path) -> pd.DataFrame:
    """健保就診人次 (門診/住院 or 急診). Works for the 48X (類流感) and 487 (流感) files.

    Columns → yw, visit_type, age, county, cases, total
    """
    df = pd.read_csv(path, encoding=ENC, dtype=str)
    cols = list(df.columns)  # 年, 週, 就診類別, 年齡別, 縣市, <cases>, 健保就診總人次
    out = pd.DataFrame(
        {
            "yw": df[cols[1]].astype(str).str.strip(),
            "visit_type": df[cols[2]].astype(str).str.strip(),
            "age": _norm_age(df[cols[3]]),
            "county": _norm_county(df[cols[4]]),
            "cases": _num(df[cols[5]]),
            "total": _num(df[cols[6]]),
        }
    )
    return out


def read_rods(path: str | Path) -> pd.DataFrame:
    """RODS 急診類流感 (hospital level). Columns → yw, age, county, town, hospital, ili, total"""
    df = pd.read_csv(
        path,
        encoding=ENC,
        dtype={"年": str, "年週": str, "年齡別": "category", "就診醫院縣市別": "category",
               "就診醫院鄉鎮別": "category", "就診醫院": "category"},
    )
    out = pd.DataFrame(
        {
            "yw": df["年週"].astype(str).str.strip(),
            "age": _norm_age(df["年齡別"].astype(str)),
            "county": _norm_county(df["就診醫院縣市別"].astype(str)),
            "town": df["就診醫院鄉鎮別"].astype(str),
            "hospital": df["就診醫院"].astype(str),
            "ili": _num(df["類流感急診就診人次"]),
            "total": _num(df["急診就診總人次"]),
        }
    )
    return out


def read_nidds(path: str | Path) -> pd.DataFrame:
    """流感併發重症 confirmed cases by onset week. Columns → yw, county, town, sex, imported, age, cases"""
    df = pd.read_csv(path, encoding=ENC, dtype=str)
    out = pd.DataFrame(
        {
            "yw": df["發病週"].astype(str).str.strip(),
            "county": _norm_county(df["縣市"]),
            "town": df["鄉鎮"].astype(str),
            "sex": df["性別"].astype(str),
            "imported": df["是否為境外移入"].astype(str).eq("是"),
            "age": _norm_age(df["年齡層"]),
            "cases": _num(df["確定病例數"]),
        }
    )
    return out


def read_lab_types(path: str | Path) -> pd.DataFrame:
    """實驗室自動通報系統（LARS） A/B/未分型陽性數 (LARS; contract-lab data to be added later). Columns → yw, county, town, hospital, flu_a, flu_b, flu_u"""
    df = pd.read_csv(path, encoding=ENC, dtype=str)
    out = pd.DataFrame(
        {
            "yw": df["SR_YW"].astype(str).str.strip(),
            "county": _norm_county(df["COUNTY_NAME"]),
            "town": df["TOWN_NAME"].astype(str),
            "hospital": df["HOSPITAL_NAME"].astype(str),
            "flu_a": _num(df["SUM_of_FLU_A"]),
            "flu_b": _num(df["SUM_of_FLU_B"]),
            "flu_u": _num(df["SUM_of_FLU_U"]),
        }
    )
    return out


def read_lab_tests(path: str | Path) -> pd.DataFrame:
    """檢驗件數 (denominator for positivity). Columns → yw, hospital, nhi_code, tests"""
    df = pd.read_csv(path, encoding=ENC, dtype=str)
    out = pd.DataFrame(
        {
            "yw": df["yearweek"].astype(str).str.strip(),
            "hospital": df["HOSPITAL_NAME"].astype(str),
            "nhi_code": df["NHI_CODE"].astype(str),
            "tests": _num(df["SUM_of_INSPECTION_NUM"]),
        }
    )
    return out


RESP_PATHOGENS = {"Influenza": "resp_flu", "RSV": "resp_rsv", "SARS-CoV-2": "resp_covid", "hMPV": "resp_hmpv",
                  "Parainfluenza": "resp_para", "Rhinovirus": "resp_rhino", "Adeno": "resp_adeno", "Mycoplasma": "resp_myco"}


def read_resp_lab(path: str | Path | None = None) -> pd.DataFrame:
    """社區合約實驗室呼吸道病原體分子生物學檢測（RESP_LAB.csv，UTF-8，2025 起，全國週資料）.

    Returns one row per yw with positives per pathogen (resp_*), overall positive %, and
    derived: resp_total_pos, resp_flu_share, resp_tests_est (= total positives / positive %),
    resp_flu_pos_rate (%).
    """
    p = Path(path) if path else DATA_DIR / "RESP_LAB.csv"
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str)
    wk = [c for c in df.columns if c.lower().startswith("year-week")][0]
    out = pd.DataFrame({"yw": df[wk].astype(str).str.strip()})
    for src, dst in RESP_PATHOGENS.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    out["resp_pos_pct"] = pd.to_numeric(df["Positive (%)"], errors="coerce")
    path_cols = list(RESP_PATHOGENS.values())
    out = out[out[path_cols].notna().any(axis=1)]  # the file pre-fills future weeks with empty rows
    out["resp_total_pos"] = out[path_cols].sum(axis=1)
    out["resp_flu_share"] = out["resp_flu"] / out["resp_total_pos"].replace(0, np.nan)
    out["resp_tests_est"] = out["resp_total_pos"] / (out["resp_pos_pct"] / 100).replace(0, np.nan)
    out["resp_flu_pos_rate"] = 100 * out["resp_flu"] / out["resp_tests_est"]
    # 3-week trailing means: the sentinel sample is small (~250 specimens/week), so weekly values are noisy
    out["resp_flu_ma3"] = out["resp_flu"].rolling(3, min_periods=1).mean()
    out["resp_flu_pos_rate_ma3"] = out["resp_flu_pos_rate"].rolling(3, min_periods=1).mean()
    return out.set_index("yw").astype(float)


def read_holidays(path: str | Path | None = None) -> pd.DataFrame:
    """tw_holiday.csv (UTF-8): date, name, isHoliday, holidayCategory, description."""
    p = Path(path) if path else DATA_DIR / "tw_holiday.csv"
    df = pd.read_csv(p, parse_dates=["date"])
    df["isHoliday"] = df["isHoliday"].astype(str).str.lower().eq("true")
    return df


RAW_FILES = {
    "nhi_48x": "NHI_PROCESSED_48X.csv",
    "nhi_48x_er": "NHI_PROCESSED_48X_ER.csv",
    "nhi_487": "NHI_PROCESSED_487.csv",
    "nhi_487_er": "NHI_PROCESSED_487_ER.csv",
    "rods": "RODS_RS.csv",
    "nidds": "NIDDS_487A.csv",
    "lab_types": "INFLUENZA_TYPE_YW.csv",
    "lab_tests": "INFLUENZA_MON_TYPE_YW.csv",
    "resp_lab": "RESP_LAB.csv",
}


def load_all(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    d = Path(data_dir) if data_dir else DATA_DIR
    return {
        "nhi_48x": read_nhi(d / RAW_FILES["nhi_48x"]),
        "nhi_48x_er": read_nhi(d / RAW_FILES["nhi_48x_er"]),
        "nhi_487": read_nhi(d / RAW_FILES["nhi_487"]),
        "nhi_487_er": read_nhi(d / RAW_FILES["nhi_487_er"]),
        "rods": read_rods(d / RAW_FILES["rods"]),
        "nidds": read_nidds(d / RAW_FILES["nidds"]),
        "lab_types": read_lab_types(d / RAW_FILES["lab_types"]),
        "lab_tests": read_lab_tests(d / RAW_FILES["lab_tests"]),
        "resp_lab": read_resp_lab(d / RAW_FILES["resp_lab"]) if (d / RAW_FILES["resp_lab"]).exists() else None,
    }
