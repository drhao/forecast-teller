"""Readers for the CDC open-data enterovirus files (UTF-8-sig) and internal cp950 variants."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA_DIR

NHI_FILE = DATA_DIR / "NHI_EnteroviralInfection.csv"
RODS_FILE = DATA_DIR / "RODS_EnteroviralInfection.csv"
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


def read_nhi_ev(path: str | Path | None = None) -> pd.DataFrame:
    """健保門診及住院（及急診，若有）腸病毒就診人次. Long: yw, kind(out/inp/er), age, county, cases, total."""
    df = read_csv_auto(path or NHI_FILE)
    cases = next(c for c in df.columns if "腸病毒" in c and "人次" in c)
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


def read_rods_ev(path: str | Path | None = None) -> pd.DataFrame:
    """RODS 急診腸病毒就診人次 (no denominator in the open-data file). Long: yw, age, county, cases."""
    df = read_csv_auto(path or RODS_FILE)
    cases = next(c for c in df.columns if "腸病毒" in c and "人次" in c)
    return pd.DataFrame({
        "yw": _yw(df),
        "age": df["年齡別"].str.strip(),
        "county": df["縣市"].str.strip().map(lambda c: COUNTY_FIX.get(c, c)),
        "cases": pd.to_numeric(df[cases], errors="coerce").fillna(0).astype(int),
    })
