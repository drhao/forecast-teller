"""Taiwan CDC epidemiological week utilities (from data/date_week_mapping.csv).

A yearweek ("yw") is a 6-character string 'YYYYWW'. From 2009 on, weeks start on
Sunday and some years have 53 weeks (2009, 2014, 2020, 2025, ...). Before 2009 the
mapping anchors week 1 on January 1st, so edge weeks can be shorter than 7 days.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA_DIR


@lru_cache(maxsize=1)
def week_table(path: str | None = None) -> pd.DataFrame:
    """One row per yearweek: yw, year, week, week_start, week_end, n_days, n_weeks_in_year."""
    p = Path(path) if path else DATA_DIR / "date_week_mapping.csv"
    m = pd.read_csv(p, parse_dates=["date"])
    m["yw"] = m["year"].astype(int).astype(str) + m["week"].astype(int).map("{:02d}".format)
    g = (
        m.groupby("yw")
        .agg(
            year=("year", "first"),
            week=("week", "first"),
            week_start=("date", "min"),
            week_end=("date", "max"),
            n_days=("date", "size"),
        )
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    g["n_weeks_in_year"] = g.groupby("year")["week"].transform("max")
    return g


def all_yws() -> list[str]:
    return week_table()["yw"].tolist()


def yw_range(start: str, end: str) -> list[str]:
    """Inclusive list of consecutive yearweeks from start to end."""
    s = all_yws()
    i, j = s.index(start), s.index(end)
    if j < i:
        raise ValueError(f"end {end} before start {start}")
    return s[i : j + 1]


def next_weeks(last_yw: str, n: int) -> list[str]:
    """The n yearweeks following last_yw."""
    s = all_yws()
    i = s.index(last_yw)
    return s[i + 1 : i + 1 + n]


def shift_yw(yw: str, k: int) -> str:
    s = all_yws()
    return s[s.index(yw) + k]


def week_start(yw: str) -> pd.Timestamp:
    t = week_table().set_index("yw")
    return t.loc[yw, "week_start"]


def seasonal_phase(yws: list[str]) -> np.ndarray:
    """Position of each week within its year in [0, 1), robust to 53-week years."""
    t = week_table().set_index("yw")
    w = t.loc[list(yws)]
    return ((w["week"] - 0.5) / w["n_weeks_in_year"]).to_numpy(dtype=np.float64)


def yw_of_date(d) -> str:
    """Yearweek containing a calendar date."""
    t = week_table()
    d = pd.Timestamp(d)
    row = t[(t["week_start"] <= d) & (t["week_end"] >= d)]
    if row.empty:
        raise ValueError(f"date {d.date()} outside mapping table")
    return row["yw"].iloc[0]
