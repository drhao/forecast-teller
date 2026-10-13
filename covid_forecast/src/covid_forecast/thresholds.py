"""COVID-19 epidemic-period rule (provisional) and the yearly threshold table.

Taiwan CDC declares 流行期 when visits 「達流行閾值」 but has not published the number. Until it is confirmed
(data/thresholds.csv keeps the status), the pages do not draw a threshold line and use this provisional rule
(user decision 2026-10-13):
  enter : 3 consecutive weekly rises AND the week is at least 2x the trough of the current inter-wave low;
  exit  : after entry, 2 consecutive weekly declines AND the week is below half of the wave's peak.
The entry level (2 x trough) and exit level (peak / 2) are reported as `level` so the pages can show
「達暫行進入水準的機率」 instead of a fixed threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import DATA_DIR

RULE_TEXT = ("暫行規則（待確認）：門急診就診人次連續 3 週上升且達本波低點的 2 倍即進入流行期；進入後連續 2 週下降且低於本波峰值的一半即脫離。"
             "疾管署宣布流行期的「流行閾值」數值尚未取得，確認後改依官方閾值，網頁在此之前不畫閾值線。")
RISES_TO_ENTER, TROUGH_MULT, FALLS_TO_EXIT, PEAK_FRAC = 3, 2.0, 2, 0.5


def load_thresholds() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "thresholds.csv", encoding="utf-8-sig")
    df["year"] = df["year"].astype(int)
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    return df.set_index("year").sort_index()


def threshold_for(yw: str, table: pd.DataFrame | None = None) -> float | None:
    """Official threshold for the week's year, or None while unconfirmed."""
    t = table if table is not None else load_thresholds()
    y = int(str(yw)[:4])
    row = t.loc[y] if y in t.index else t.iloc[-1]
    v = row["threshold"]
    return None if pd.isna(v) else float(v)


def epidemic_flags(y: pd.Series) -> pd.DataFrame:
    """Apply the provisional entry/exit rule. Returns level (entry level 2x trough when outside, exit level
    peak/2 when inside), trough, peak, rises, falls, in_period, entered, exited; indexed like `y`."""
    y = y.dropna().astype(float)
    vals = y.to_numpy()
    state = False
    trough = peak = vals[0] if len(vals) else np.nan
    rises = falls = 0
    rows = []
    for i, v in enumerate(vals):
        if i > 0:
            if v > vals[i - 1]:
                rises, falls = rises + 1, 0
            elif v < vals[i - 1]:
                falls, rises = falls + 1, 0
            else:
                rises = falls = 0
        entered = exited = False
        if not state:
            trough = min(trough, v)
            if rises >= RISES_TO_ENTER and v >= TROUGH_MULT * trough:
                state, entered, peak = True, True, v
        else:
            peak = max(peak, v)
            if falls >= FALLS_TO_EXIT and v < PEAK_FRAC * peak:
                state, exited, trough = False, True, v
        level = PEAK_FRAC * peak if state else TROUGH_MULT * trough
        rows.append({"level": level, "trough": trough, "peak": peak, "rises": rises, "falls": falls,
                     "in_period": state, "entered": entered, "exited": exited})
    return pd.DataFrame(rows, index=y.index)


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
