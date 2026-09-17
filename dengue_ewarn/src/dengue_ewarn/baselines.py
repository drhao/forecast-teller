"""Probabilistic baseline forecasters returning (median (H,), quantiles (H, 9))."""
from __future__ import annotations

import numpy as np

Q_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def _from_errors(point: np.ndarray, errors: np.ndarray, positive: bool = True, center: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Quantiles = point + empirical error quantiles (errors shape (n,) or (n, H)).

    With center=True the error quantiles are shifted so their median is zero: the median
    forecast is then exactly the baseline rule (last value, moving average, ...) and the
    empirical errors only provide the width/asymmetry of the prediction intervals.
    """
    errors = np.asarray(errors, dtype=float)
    if errors.ndim == 1:
        eq = np.quantile(errors, Q_LEVELS)  # (9,)
        if center:
            eq = eq - eq[4]
        q = point[:, None] + eq[None, :]
    else:
        eq = np.quantile(errors, Q_LEVELS, axis=0).T  # (H, 9)
        if center:
            eq = eq - eq[:, 4:5]
        q = point[:, None] + eq
    if positive:
        q = np.maximum(q, 0.0)
    return q[:, 4].copy(), q


def seasonal_naive(context: np.ndarray, horizon: int, season: int = 52) -> tuple[np.ndarray, np.ndarray]:
    """Same week last year, with an empirical error distribution from the context."""
    c = np.asarray(context, dtype=float)
    if len(c) < season + 8:
        raise ValueError("context too short for seasonal naive")
    point = np.array([c[len(c) - season + h] if h < season else c[len(c) - 2 * season + h] for h in range(horizon)])
    errors = c[season:] - c[:-season]
    return _from_errors(point, errors)


def naive_last(context: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Last observed value, with horizon-specific empirical change distributions."""
    c = np.asarray(context, dtype=float)
    point = np.repeat(c[-1], horizon)
    errors = np.stack([c[h:] [: len(c) - horizon] - c[: len(c) - h][: len(c) - horizon] for h in range(1, horizon + 1)], axis=1)
    return _from_errors(point, errors)


def moving_average(context: np.ndarray, horizon: int, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """MA(window) baseline: the forecast for a week is the mean of the previous `window` weeks.

    h = 1 uses the last `window` observed weeks; for h >= 2 the intermediate weeks are not yet
    observed, so their forecasts are substituted recursively. Quantiles come from the empirical
    distribution of the same recursive h-step errors inside the context.
    """
    c = np.asarray(context, dtype=float)
    if len(c) < window + horizon + 8:
        raise ValueError("context too short for moving-average baseline")

    def rec(hist: np.ndarray) -> np.ndarray:
        buf = list(hist[-window:]); out = []
        for _ in range(horizon):
            f = float(np.mean(buf[-window:])); out.append(f); buf.append(f)
        return np.array(out)

    point = rec(c)
    errs = np.empty((len(c) - horizon - window + 1, horizon))
    for i, k in enumerate(range(window, len(c) - horizon + 1)):
        errs[i] = c[k:k + horizon] - rec(c[:k])
    return _from_errors(point, errs)
