"""Point and probabilistic forecast metrics. Quantile arrays are (H, 9) for levels 0.1…0.9."""
from __future__ import annotations

import numpy as np

Q_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MEDIAN_IDX = 4
# (lower idx, upper idx, alpha) for the four central intervals available from deciles
INTERVALS = [(0, 8, 0.2), (1, 7, 0.4), (2, 6, 0.6), (3, 5, 0.8)]


def wis(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Weighted interval score per horizon step (FluSight definition, K=4 intervals)."""
    y = np.asarray(y, dtype=float)
    total = 0.5 * np.abs(y - q[:, MEDIAN_IDX])
    for lo, hi, a in INTERVALS:
        l, u = q[:, lo], q[:, hi]
        interval_score = (u - l) + (2 / a) * (l - y) * (y < l) + (2 / a) * (y - u) * (y > u)
        total += (a / 2) * interval_score
    return total / (len(INTERVALS) + 0.5)


def covered(y: np.ndarray, q: np.ndarray, level: int = 80) -> np.ndarray:
    """Boolean per step: is y inside the central `level`% interval (80 → q10–q90, 60 → q20–q80)."""
    lo, hi = {80: (0, 8), 60: (1, 7), 40: (2, 6), 20: (3, 5)}[level]
    return (y >= q[:, lo]) & (y <= q[:, hi])


def abs_err(y: np.ndarray, median: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y, dtype=float) - median)


def mase_scale(context: np.ndarray, season: int = 52) -> float:
    """In-sample MAE of the seasonal naive forecast over the context (denominator of MASE)."""
    c = np.asarray(context, dtype=float)
    if len(c) <= season:
        return np.nan
    return float(np.mean(np.abs(c[season:] - c[:-season])))


# ----------------------------------------------------------------------------- hit rates
def directional_hit(y_origin: float, y: np.ndarray, median: np.ndarray) -> np.ndarray:
    """2-class directional hit: forecast says 'up' (median > last observed) iff actual is up."""
    y = np.asarray(y, dtype=float); median = np.asarray(median, dtype=float)
    return (y > y_origin) == (median > y_origin)


def _direction_class(v: np.ndarray, y_origin: float, band: float) -> np.ndarray:
    change = (np.asarray(v, dtype=float) - y_origin) / max(abs(y_origin), 1e-9)
    return np.where(change > band, 1, np.where(change < -band, -1, 0))


def directional_hit3(y_origin: float, y: np.ndarray, median: np.ndarray, band: float = 0.05) -> np.ndarray:
    """3-class directional hit (up / stable / down, stable = within ±band of the last observed value)."""
    return _direction_class(y, y_origin, band) == _direction_class(median, y_origin, band)


def tolerance_hit(y: np.ndarray, median: np.ndarray, tol: float = 0.10) -> np.ndarray:
    """Median within ±tol (relative) of the actual value."""
    y = np.asarray(y, dtype=float)
    return np.abs(np.asarray(median, dtype=float) - y) <= tol * np.abs(y)


def threshold_flags(y: np.ndarray, median: np.ndarray, thr: float) -> dict[str, np.ndarray]:
    """Event = actual >= thr, alarm = median >= thr. Returns boolean tp/fp/fn/tn arrays."""
    event = np.asarray(y, dtype=float) >= thr
    alarm = np.asarray(median, dtype=float) >= thr
    return {"tp": event & alarm, "fp": ~event & alarm, "fn": event & ~alarm, "tn": ~event & ~alarm}


def threshold_rates(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    """Hit rate (sensitivity), false-alarm rate, accuracy from confusion counts."""
    return {
        "thr_hit_rate": tp / (tp + fn) if (tp + fn) else np.nan,
        "thr_false_alarm": fp / (fp + tn) if (fp + tn) else np.nan,
        "thr_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }
