"""Known-future covariates for enterovirus: school calendar, two-harmonic season, plus the shared holiday/CNY flags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_teller.covariates import holiday_weekly, seasonal_features
from forecast_teller.weeks import next_weeks, week_start

# Lunar New Year day (初一); used to place the end of the winter break.
CNY = {2016: "2016-02-08", 2017: "2017-01-28", 2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25",
       2021: "2021-02-12", 2022: "2022-02-01", 2023: "2023-01-22", 2024: "2024-02-10", 2025: "2025-01-29",
       2026: "2026-02-17", 2027: "2027-02-06", 2028: "2028-01-26", 2029: "2029-02-13"}
SCHOOL_COLS = ["summer_break", "winter_break", "reopen"]


def school_calendar(yws: list[str]) -> pd.DataFrame:
    """Approximate K-12 calendar by the week's Wednesday: summer break Jul 1–Aug 29; winter break Jan 21
    to max(Feb 10, CNY + 6 days); `reopen` = first two weeks after either break."""
    rows = []
    for yw in yws:
        d = week_start(yw) + pd.Timedelta(days=3)
        y = d.year
        summer = pd.Timestamp(y, 7, 1) <= d <= pd.Timestamp(y, 8, 29)
        cny = pd.Timestamp(CNY.get(y, f"{y}-02-05"))
        w_end = max(pd.Timestamp(y, 2, 10), cny + pd.Timedelta(days=6))
        winter = pd.Timestamp(y, 1, 21) <= d <= w_end
        rows.append((yw, int(summer), int(winter)))
    df = pd.DataFrame(rows, columns=["yw", "summer_break", "winter_break"]).set_index("yw")
    brk = (df["summer_break"] + df["winter_break"]).clip(upper=1)
    ended = (brk.shift(1, fill_value=0) == 1) & (brk == 0)          # first week after a break
    df["reopen"] = (ended | ended.shift(1, fill_value=False)).astype(int)
    return df


def season2(yws: list[str]) -> np.ndarray:
    """(4, N): sin/cos at 52.18 and 26.09 weeks (bimodal enterovirus season)."""
    wk = np.array([int(w[4:]) - 1 for w in yws], dtype=float)
    a = 2 * np.pi * wk / 52.18
    return np.vstack([np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)]).astype(np.float32)


def future_covariates(context_yws: list[str], horizon: int, kinds: tuple[str, ...] = ("school", "holiday")) -> np.ndarray:
    """(W, L+H) past+future covariate matrix over the context weeks plus `horizon` future weeks.
    kinds: season (flu 2-channel), season2 (4-channel), school (3), cny (1), holiday (1)."""
    yws = list(context_yws) + next_weeks(context_yws[-1], horizon)
    parts = []
    if "season" in kinds:
        parts.append(np.asarray(seasonal_features(yws), dtype=np.float32))
    if "season2" in kinds:
        parts.append(season2(yws))
    if "school" in kinds:
        parts.append(school_calendar(yws)[SCHOOL_COLS].to_numpy(np.float32).T)
    if "cny" in kinds or "holiday" in kinds:
        h = holiday_weekly(yws)
        if "cny" in kinds:
            parts.append(h["cny_week"].to_numpy(np.float32)[None, :])
        if "holiday" in kinds:
            parts.append(h["holiday_days"].to_numpy(np.float32)[None, :])
    if not parts:
        raise ValueError("no covariate kinds selected")
    return np.vstack(parts)
