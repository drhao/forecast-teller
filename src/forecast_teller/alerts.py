"""Threshold-crossing alerts from quantile forecasts, static/statistical detectors, lead times and alert metrics."""
from __future__ import annotations

import numpy as np

Q_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def prob_ge(q: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """P(Y >= thr) from decile forecasts q (..., 9), linear interpolation of the CDF; tails clipped to [0.02, 0.98]."""
    q = np.asarray(q, dtype=float); thr = np.asarray(thr, dtype=float)[..., None]
    below = (q <= thr).sum(axis=-1)                      # number of quantiles <= thr, 0..9
    p = np.empty(below.shape, dtype=float)
    lo_mask, hi_mask = below == 0, below == 9
    mid = ~(lo_mask | hi_mask)
    i = np.clip(below - 1, 0, 7)
    q_lo = np.take_along_axis(q, i[..., None], axis=-1)[..., 0]
    q_hi = np.take_along_axis(q, (i + 1)[..., None], axis=-1)[..., 0]
    frac = np.where(q_hi > q_lo, (thr[..., 0] - q_lo) / np.where(q_hi > q_lo, q_hi - q_lo, 1), 0.0)
    F = Q_LEVELS[i] + frac * 0.1
    p[mid] = 1 - F[mid]
    p[lo_mask] = 0.98; p[hi_mask] = 0.02
    return p


def ewma_alarm(x: np.ndarray, lam: float = 0.3, k: float = 3.0, baseline: int = 56, min_sd: float = 0.5) -> np.ndarray:
    """Daily EWMA control chart; alarm when EWMA exceeds trailing-baseline mean + k·sd (baseline excludes last 7 days)."""
    x = np.asarray(x, dtype=float); T = x.shape[-1]
    z = np.zeros_like(x); z[..., 0] = x[..., 0]
    for t in range(1, T):
        z[..., t] = lam * x[..., t] + (1 - lam) * z[..., t - 1]
    alarm = np.zeros(x.shape, dtype=bool)
    for t in range(baseline + 7, T):
        b = x[..., t - baseline - 7: t - 7]
        mu, sd = b.mean(axis=-1), np.maximum(b.std(axis=-1), min_sd)
        alarm[..., t] = z[..., t] > mu + k * sd
    return alarm


def cusum_alarm(x: np.ndarray, k: float = 0.5, h: float = 4.0, baseline: int = 56, min_sd: float = 0.5) -> np.ndarray:
    """One-sided CUSUM on standardized daily counts (trailing baseline, excluding last 7 days)."""
    x = np.asarray(x, dtype=float); T = x.shape[-1]
    s = np.zeros(x.shape[:-1]); alarm = np.zeros(x.shape, dtype=bool)
    for t in range(baseline + 7, T):
        b = x[..., t - baseline - 7: t - 7]
        mu, sd = b.mean(axis=-1), np.maximum(b.std(axis=-1), min_sd)
        s = np.maximum(0, s + (x[..., t] - mu) / sd - k)
        alarm[..., t] = s > h
        s = np.where(alarm[..., t], 0, s)  # reset after alarm
    return alarm


def episodes(S: np.ndarray, thr: np.ndarray, min_gap: int = 14) -> list[int]:
    """Indices where the 7-day sum first reaches the threshold after >= min_gap days below it."""
    above = S >= thr
    out, last_above = [], -10**9
    for t in range(len(S)):
        if above[t]:
            if t - last_above > min_gap:
                out.append(t)
            last_above = t
    return out


def alert_metrics(event: np.ndarray, prob: np.ndarray, p_star: float) -> dict:
    """Sensitivity, false-alarm rate (per 100 non-event origins), PPV for alerts prob >= p_star."""
    a = prob >= p_star
    tp, fp, fn, tn = int((a & event).sum()), int((a & ~event).sum()), int((~a & event).sum()), int((~a & ~event).sum())
    return {"p_star": p_star, "alerts": tp + fp, "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "false_alarm_per100": 100 * fp / (fp + tn) if fp + tn else np.nan, "ppv": tp / (tp + fp) if tp + fp else np.nan, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auc(event: np.ndarray, prob: np.ndarray) -> float:
    """ROC AUC by rank statistic."""
    pos, neg = prob[event], prob[~event]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
