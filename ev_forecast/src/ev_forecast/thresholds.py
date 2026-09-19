"""Enterovirus epidemic thresholds (門急診就診人次, set by Taiwan CDC each year) and the 流行期 rule.

Rule (user decision 2026-09-19, provisional): a week at or above the year's threshold enters
the epidemic period; two consecutive weeks below the threshold leave it (the second week is
already outside). Thresholds come from data/thresholds.csv (one row per year, with source).
"""
from __future__ import annotations

import pandas as pd

from . import DATA_DIR

RULE_TEXT = "單週門急診就診人次達流行閾值即進入流行期；連續 2 週低於閾值即脫離（第 2 週起視為非流行期）。閾值由疾管署每年設定，本專案依各年新聞稿記錄於 data/thresholds.csv。"


def load_thresholds() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "thresholds.csv", encoding="utf-8-sig")
    df["year"] = df["year"].astype(int)
    return df.set_index("year").sort_index()


def threshold_for(yw: str, table: pd.DataFrame | None = None) -> float:
    t = table if table is not None else load_thresholds()
    y = int(str(yw)[:4])
    if y in t.index:
        return float(t.loc[y, "threshold"])
    return float(t.loc[t.index.max(), "threshold"])  # carry the latest value forward (and back before first year)


def threshold_series(yws: list[str], table: pd.DataFrame | None = None) -> pd.Series:
    t = table if table is not None else load_thresholds()
    return pd.Series([threshold_for(w, t) for w in yws], index=list(yws), name="ev_thr", dtype=float)


def epidemic_flags(y: pd.Series, thr: pd.Series | None = None) -> pd.DataFrame:
    """Apply the entry/exit rule to an observed weekly series. Returns thr, in_period, entered, exited."""
    y = y.dropna()
    thr = threshold_series(list(y.index)) if thr is None else thr.reindex(y.index)
    in_p, ent, ext = [], [], []
    state, below_run = False, 0
    for w in y.index:
        v, t = float(y[w]), float(thr[w])
        entered = exited = False
        if not state:
            if v >= t:
                state, entered, below_run = True, True, 0
        else:
            if v < t:
                below_run += 1
                if below_run >= 2:
                    state, exited = False, True
            else:
                below_run = 0
        in_p.append(state); ent.append(entered); ext.append(exited)
    return pd.DataFrame({"thr": thr.values, "in_period": in_p, "entered": ent, "exited": ext}, index=y.index)


def periods(flags: pd.DataFrame) -> list[tuple[str, str | None]]:
    """List of (entry week, last in-period week or None if ongoing)."""
    out, start = [], None
    for w, r in flags.iterrows():
        if r.entered:
            start = w
        if r.exited and start is not None:
            prev = flags.index[flags.index.get_loc(w) - 1]
            out.append((start, prev)); start = None
    if start is not None:
        out.append((start, None))
    return out
