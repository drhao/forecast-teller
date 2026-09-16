"""Known-future (past+future) covariates: seasonal phase and Lunar New Year / holiday flags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import read_holidays
from .weeks import next_weeks, seasonal_phase, week_table, yw_of_date

# Lunar New Year's Day for years beyond tw_holiday.csv. Verify against the official
# 行政機關辦公日曆表 when the file is extended.
LNY_DATES = {2026: "2026-02-17", 2027: "2027-02-06", 2028: "2028-01-26", 2029: "2029-02-13"}
WEEKEND_CATEGORY = "星期六、星期日"


def holiday_weekly(yws: list[str]) -> pd.DataFrame:
    """Per yearweek: holiday_days (weekday public holidays) and cny_week (0/1)."""
    yws = list(dict.fromkeys(yws))  # de-duplicate, keep order (history and forecast weeks may overlap)
    wt = week_table().set_index("yw")
    h = read_holidays()
    starts = wt["week_start"].sort_values()
    idx = np.searchsorted(starts.values, h["date"].values, side="right") - 1
    h["yw"] = starts.index[idx]
    weekday_hol = h[h["isHoliday"] & (h["holidayCategory"] != WEEKEND_CATEGORY)]
    days = weekday_hol.groupby("yw").size()
    # the file sometimes puts the holiday name in `name`, sometimes only in `holidayCategory`
    text = h["name"].fillna("") + "|" + h["holidayCategory"].fillna("")
    is_cny = text.str.contains("春節|除夕|小年夜") & h["isHoliday"]
    cny_weeks = set(h.loc[is_cny, "yw"])
    file_max = h["date"].max()
    # extend beyond the holiday file with hard-coded Lunar New Year dates
    for yr, d in LNY_DATES.items():
        d = pd.Timestamp(d)
        if d > file_max:
            cny_weeks.add(yw_of_date(d))
            cny_weeks.add(yw_of_date(d - pd.Timedelta(days=1)))
    out = pd.DataFrame(index=pd.Index(yws, name="yw"))
    out["holiday_days"] = days.reindex(yws).fillna(0).astype(float)
    out["cny_week"] = [1.0 if w in cny_weeks else 0.0 for w in yws]
    # Beyond the holiday file the weekday-holiday count is unknown. The CNY break is
    # always at least Mon–Fri of the week containing New Year's Day when the break spans
    # 9 days (2026: Feb 14–22), so assume 5 for CNY weeks and 0 elsewhere. Extend
    # data/tw_holiday.csv with the official 辦公日曆表 to replace this approximation.
    beyond = [w for w in yws if wt.loc[w, "week_start"] > file_max]
    for w in beyond:
        if out.loc[w, "cny_week"] == 1.0:
            out.loc[w, "holiday_days"] = max(out.loc[w, "holiday_days"], 5.0)
    return out


def seasonal_features(yws: list[str]) -> np.ndarray:
    """(2, L) sin/cos of the within-year phase, robust to 53-week years."""
    ph = seasonal_phase(yws)
    return np.stack([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)]).astype(np.float32)


def future_covariates(context_yws: list[str], horizon: int, kinds: tuple[str, ...] = ("season", "cny")) -> np.ndarray:
    """Past+future covariate matrix of shape (W, L + horizon) for the given context weeks."""
    yws = list(context_yws) + next_weeks(context_yws[-1], horizon)
    rows = []
    if "season" in kinds:
        rows.append(seasonal_features(yws))
    if "cny" in kinds or "holiday" in kinds:
        hw = holiday_weekly(yws)
        if "cny" in kinds:
            rows.append(hw["cny_week"].to_numpy(np.float32)[None, :])
        if "holiday" in kinds:
            rows.append(hw["holiday_days"].to_numpy(np.float32)[None, :])
    if not rows:
        raise ValueError("no covariate kinds selected")
    return np.concatenate(rows, axis=0).astype(np.float32)
