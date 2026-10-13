"""Readers for the COVID-19 inputs: CDC open data (UTF-8-sig), NIDSS pulls (fetch_nidss_visits.py) and severe cases (fetch_severe.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA_DIR

NHI_FILE = DATA_DIR / "NHI_COVID-19.csv"
RODS_FILE = DATA_DIR / "RODS_COVID-19.csv"
NIDSS_FILE = DATA_DIR / "nidss_visits.csv"
SEVERE_FILE = DATA_DIR / "severe_weekly.csv"
KIND = {"門診": "out", "住院": "inp", "急診": "er"}

try:  # same county spelling as the influenza panel (臺/台 etc.)
    from forecast_teller.io import COUNTY_FIX
except Exception:  # pragma: no cover
    COUNTY_FIX = {}


def read_csv_auto(path: str | Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp950"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def _yw(df: pd.DataFrame) -> pd.Series:
    if "年週" in df.columns:
        return df["年週"].str.strip()
    w = df["週"].str.strip()
    long = w.str.len() >= 5
    return pd.Series(np.where(long, w, df["年"].str.strip() + w.str.zfill(2)), index=df.index)


def read_nhi_covid(path: str | Path | None = None) -> pd.DataFrame:
    """健保門診及住院 COVID-19 就診人次（開放資料）. Long: yw, kind(out/inp), age, county, cases, total."""
    df = read_csv_auto(path or NHI_FILE)
    cases = next(c for c in df.columns if "COVID" in c.upper() and "人次" in c and "總" not in c)
    total = next(c for c in df.columns if "總人次" in c)
    out = pd.DataFrame({
        "yw": _yw(df),
        "kind": df["就診類別"].str.strip().map(KIND),
        "age": df["年齡別"].str.strip(),
        "county": df["縣市"].str.strip().map(lambda c: COUNTY_FIX.get(c, c)),
        "cases": pd.to_numeric(df[cases], errors="coerce").fillna(0).astype(int),
        "total": pd.to_numeric(df[total], errors="coerce").fillna(0).astype(int),
    })
    if out["kind"].isna().any():
        raise ValueError(f"unknown 就診類別: {df['就診類別'][out['kind'].isna()].unique()}")
    return out


def read_rods_covid(path: str | Path | None = None) -> pd.DataFrame:
    """RODS 急診 COVID-19 就診人次（開放資料，無分母）. Long: yw, age, county, cases."""
    df = read_csv_auto(path or RODS_FILE)
    cases = next(c for c in df.columns if "COVID" in c.upper() and "人次" in c)
    return pd.DataFrame({
        "yw": _yw(df),
        "age": df["年齡別"].str.strip(),
        "county": df["縣市"].str.strip().map(lambda c: COUNTY_FIX.get(c, c)),
        "cases": pd.to_numeric(df[cases], errors="coerce").fillna(0).astype(int),
    })


def read_nidss_visits(path: str | Path | None = None) -> pd.DataFrame:
    """NIDSS 健保門診 / 門診+急診 就診人次（scripts/fetch_nidss_visits.py 的輸出）.
    Columns: yw, area, age, case_type (out | oe), visits, rate_pct. Trailing all-zero weeks are the
    site's placeholder for weeks not yet reported and are dropped here."""
    p = Path(path) if path else NIDSS_FILE
    if not p.exists():
        return pd.DataFrame(columns=["yw", "area", "age", "case_type", "visits", "rate_pct"])
    df = pd.read_csv(p, dtype={"yw": str}, encoding="utf-8-sig")
    nat = df[(df.area == "全國") & (df.age == "全部") & (df.case_type == "oe")].sort_values("yw")
    filled = nat[nat.visits > 0]
    if filled.empty:
        return df.iloc[0:0]
    return df[df.yw <= filled.yw.max()].reset_index(drop=True)


def read_severe(path: str | Path | None = None) -> pd.Series:
    """新冠併發重症 週確定病例數（發病週；scripts/fetch_severe.py 的輸出）. Series indexed by yw."""
    p = Path(path) if path else SEVERE_FILE
    if not p.exists():
        return pd.Series(dtype=float, name="covid_severe")
    df = pd.read_csv(p, dtype={"yw": str}, encoding="utf-8-sig")
    return df.set_index("yw")["severe"].astype(float).rename("covid_severe")
