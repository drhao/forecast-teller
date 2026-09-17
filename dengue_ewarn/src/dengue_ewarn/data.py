"""Dengue daily line list → township × day panel, with report-delay reconstruction (as-of snapshots).

Source: 疾管署「登革熱1998年起每日確定病例統計」(Dengue_Daily.csv; the portal copy is delisted, the
project uses the Internet Archive snapshot of the official file, onset dates through 2025-07).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA_DIR, PROCESSED_DIR

CITIES = ("台南市", "高雄市")
DMAX = 120  # cap on report delay (days) tracked for as-of reconstruction


def load_line_list(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else DATA_DIR / "Dengue_Daily.csv"
    df = pd.read_csv(p, encoding="utf-8-sig", dtype=str, low_memory=False)
    out = pd.DataFrame({
        "onset": pd.to_datetime(df["發病日"], format="%Y/%m/%d", errors="coerce"),
        "report": pd.to_datetime(df["通報日"], format="%Y/%m/%d", errors="coerce"),
        "confirm": pd.to_datetime(df["個案研判日"], format="%Y/%m/%d", errors="coerce"),
        "county": df["居住縣市"].str.replace("臺", "台", regex=False),
        "township": df["居住鄉鎮"],
        "imported": df["是否境外移入"].eq("是"),
        "n": pd.to_numeric(df["確定病例數"], errors="coerce").fillna(1).astype(int),
        "serotype": df["血清型"],
    })
    out = out.dropna(subset=["onset"])
    out["report"] = out["report"].fillna(out["onset"])
    out["delay"] = (out["report"] - out["onset"]).dt.days.clip(lower=0, upper=DMAX)
    return out


def township_panel(ll: pd.DataFrame, cities=CITIES, start="2012-01-01", end: str | None = None):
    """Returns (counts DataFrame [date × series], hist ndarray [series, day, delay]) for local cases.

    counts are final onset-date counts; hist supports as-of reconstruction:
    cases with onset day t known at day d = sum_{k <= d - t} hist[s, t, k].
    """
    loc = ll[(~ll.imported) & ll.county.isin(cities)].copy()
    end = pd.Timestamp(end) if end else loc.onset.max()
    days = pd.date_range(start, end, freq="D")
    loc = loc[(loc.onset >= days[0]) & (loc.onset <= days[-1])]
    series = sorted(set(zip(loc.county, loc.township)))
    idx = {s: i for i, s in enumerate(series)}
    S, T = len(series), len(days)
    hist = np.zeros((S, T, DMAX + 1), dtype=np.int32)
    t_idx = ((loc.onset - days[0]).dt.days).to_numpy()
    s_idx = np.array([idx[(c, t)] for c, t in zip(loc.county, loc.township)])
    np.add.at(hist, (s_idx, t_idx, loc.delay.to_numpy()), loc.n.to_numpy())
    counts = pd.DataFrame(hist.sum(axis=2).T, index=days, columns=[f"{c}|{t}" for c, t in series])
    return counts, hist


def asof_counts(hist: np.ndarray, d: int) -> np.ndarray:
    """Counts by onset day known at day index d: shape (S, d + 1)."""
    S, T, K = hist.shape
    cum = np.cumsum(hist[:, : d + 1, :], axis=2)              # (S, d+1, K)
    k = np.minimum(d - np.arange(d + 1), K - 1)                # allowed delay per onset day
    return cum[:, np.arange(d + 1), k]                         # (S, d+1)


def completeness(hist: np.ndarray, upto: int, lookback: int = 730) -> np.ndarray:
    """Fraction of cases reported within k days (k = 0..DMAX), from onset days in [upto-lookback, upto-DMAX]."""
    lo, hi = max(0, upto - lookback - DMAX), max(1, upto - DMAX)
    h = hist[:, lo:hi, :].sum(axis=(0, 1)).astype(float)
    if h.sum() == 0:
        return np.ones(hist.shape[2])
    c = np.cumsum(h) / h.sum()
    return np.maximum(c, 0.05)


def rolling7(x: np.ndarray) -> np.ndarray:
    """Trailing 7-day sum along the last axis (partial windows at the start use available days)."""
    c = np.cumsum(x, axis=-1)
    out = c.copy()
    out[..., 7:] = c[..., 7:] - c[..., :-7]
    return out


def ewarn_threshold(daily: np.ndarray, d: int, weeks: int = 3, factor: float = 2.0, floor: float = 3.0) -> np.ndarray:
    """EWARN-style threshold at origin d: factor × mean weekly count over the previous `weeks` weeks, with a floor.

    daily: (S, >= d+1) counts known at d. Weeks are the 7-day blocks ending at d.
    """
    blocks = [daily[:, max(0, d - 7 * (w + 1) + 1): d - 7 * w + 1].sum(axis=1) for w in range(weeks)]
    return np.maximum(factor * np.mean(blocks, axis=0), floor)


def build(out_dir: str | Path | None = None) -> dict:
    out = Path(out_dir) if out_dir else PROCESSED_DIR
    ll = load_line_list()
    counts, hist = township_panel(ll, end=str(ll.onset.max().date()))  # keep quiet 2025 weeks for false-alarm evaluation
    counts.to_parquet(out / "dengue_township_daily.parquet")
    np.savez_compressed(out / "dengue_delay_hist.npz", hist=hist, days=counts.index.values.astype("datetime64[D]"), series=np.array(counts.columns))
    yearly = counts.groupby(counts.index.year).sum().sum(axis=1)
    return {"counts": counts, "hist": hist, "yearly": yearly, "line_list": ll}
