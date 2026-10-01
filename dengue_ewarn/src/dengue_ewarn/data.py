"""Dengue daily line list → township × day panel, with report-delay reconstruction (as-of snapshots).

Source: 疾管署「登革熱1998年起每日確定病例統計」(Dengue_Daily.csv; the portal copy is delisted, the
project uses the Internet Archive snapshot of the official file, onset dates through 2025-07).

Scopes (2026-10-01): "tn_kh" = 台南市 + 高雄市 only (panel tag "dengue", versions 1–2);
"all" = every township in Taiwan with a local case since PANEL_START (panel tag "dengue_all", version 3).
Panel files are data_processed/{tag}_township_daily.parquet and {tag}_delay_hist.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA_DIR, PROCESSED_DIR

CITIES = ("台南市", "高雄市")
SCOPES = {"tn_kh": CITIES, "all": None}   # None = every county
DMAX = 120  # cap on report delay (days) tracked for as-of reconstruction
PANEL_START = "2008-01-01"   # 2026-10-01: moved back from 2012 so that 2010–2012 (≈1,200–1,500 local cases/yr) become test years
EPIDEMIC_MIN_CASES = 1000    # a year with >= this many local cases in the panel counts as 流行年 in the reports


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


def township_panel(ll: pd.DataFrame, cities=CITIES, start=PANEL_START, end: str | None = None):
    """Returns (counts DataFrame [date × series], hist ndarray [series, day, delay]) for local cases.

    The line list already uses post-2010 administrative names (台南市/高雄市 and 區) for earlier years,
    so no recoding is needed when the panel starts before the 2010-12-25 county–city mergers.

    counts are final onset-date counts; hist supports as-of reconstruction:
    cases with onset day t known at day d = sum_{k <= d - t} hist[s, t, k].
    """
    loc = ll[~ll.imported]
    if cities:
        loc = loc[loc.county.isin(cities)]
    loc = loc.dropna(subset=["county", "township"]).copy()
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


def region_of(series_name: str) -> str:
    """Report stratum of a series ("county|township"): 台南高雄 or 其他縣市."""
    return "台南高雄" if series_name.split("|")[0] in CITIES else "其他縣市"


def epidemic_years(counts: pd.DataFrame, min_cases: int = EPIDEMIC_MIN_CASES) -> set[int]:
    """流行年 = years whose 台南 + 高雄 local cases in the panel total >= min_cases.

    Other counties are ignored on purpose so the split is the same for every scope; local outbreaks
    elsewhere (e.g. 2013 屏東, 2018 台中/新北) are evaluated through the region × event-type strata instead.
    """
    cols = [c for c in counts.columns if region_of(c) == "台南高雄"] or list(counts.columns)
    yearly = counts[cols].groupby(counts.index.year).sum().sum(axis=1)
    return {int(y) for y, v in yearly.items() if v >= min_cases}


def panel_paths(tag: str = "dengue", out_dir: str | Path | None = None) -> tuple[Path, Path]:
    out = Path(out_dir) if out_dir else PROCESSED_DIR
    return out / f"{tag}_township_daily.parquet", out / f"{tag}_delay_hist.npz"


def load_panel(tag: str = "dengue") -> tuple[pd.DataFrame, np.ndarray]:
    """(counts [date × series], hist [series, day, delay]) for a panel tag."""
    cp, hp = panel_paths(tag)
    counts = pd.read_parquet(cp)
    hist = np.load(hp, allow_pickle=True)["hist"]
    return counts, hist


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


def build(out_dir: str | Path | None = None, scope: str = "tn_kh", tag: str = "dengue") -> dict:
    ll = load_line_list()
    counts, hist = township_panel(ll, cities=SCOPES[scope], end=str(ll.onset.max().date()))  # keep quiet 2025 weeks for false-alarm evaluation
    cp, hp = panel_paths(tag, out_dir)
    counts.to_parquet(cp)
    np.savez_compressed(hp, hist=hist, days=counts.index.values.astype("datetime64[D]"), series=np.array(counts.columns))
    yearly = counts.groupby(counts.index.year).sum().sum(axis=1)
    return {"counts": counts, "hist": hist, "yearly": yearly, "line_list": ll}
