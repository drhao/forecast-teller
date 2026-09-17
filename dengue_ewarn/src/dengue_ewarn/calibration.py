"""Probability calibration (isotonic, leave-one-year-out) and two-tier alert workload analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- isotonic (PAV)
class Isotonic:
    """Non-decreasing calibration map fitted by pool-adjacent-violators; linear interpolation between block centres."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Isotonic":
        order = np.argsort(x, kind="mergesort"); x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
        # blocks: (sum_y, n, sum_x)
        blocks = [[yi, 1.0, xi] for xi, yi in zip(x, y)]
        merged: list[list[float]] = []
        for b in blocks:
            merged.append(b)
            while len(merged) > 1 and merged[-2][0] / merged[-2][1] > merged[-1][0] / merged[-1][1]:
                b2 = merged.pop(); merged[-1] = [merged[-1][0] + b2[0], merged[-1][1] + b2[1], merged[-1][2] + b2[2]]
        self.x_ = np.array([b[2] / b[1] for b in merged]); self.y_ = np.array([b[0] / b[1] for b in merged])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if len(self.x_) == 1:
            return np.full(x.shape, self.y_[0])
        return np.clip(np.interp(x, self.x_, self.y_), 0.0, 1.0)


class Platt:
    """Logistic recalibration p' = sigmoid(a·logit(p) + b), fitted by Newton's method (monotone → ranking preserved)."""

    def _z(self, p): p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3); return np.log(p / (1 - p))

    def fit(self, x, y) -> "Platt":
        z, y = self._z(x), np.asarray(y, float); a, b = 1.0, 0.0
        for _ in range(50):
            q = 1 / (1 + np.exp(-(a * z + b))); w = q * (1 - q) + 1e-9
            g = np.array([np.sum((q - y) * z), np.sum(q - y)]); H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) + 1e-6 * np.eye(2)
            step = np.linalg.solve(H, g); a, b = a - step[0], b - step[1]
            if np.abs(step).max() < 1e-8:
                break
        self.a_, self.b_ = a, b; return self

    def predict(self, x): return 1 / (1 + np.exp(-(self.a_ * self._z(x) + self.b_)))


def loyo_calibrate(df: pd.DataFrame, prob_col="prob", event_col="event", year_col="year", method="isotonic") -> np.ndarray:
    """Leave-one-year-out recalibration (isotonic or Platt); returns out-of-sample calibrated probabilities."""
    out = np.full(len(df), np.nan)
    for yr in sorted(df[year_col].unique()):
        tr, te = df[year_col] != yr, df[year_col] == yr
        if tr.sum() == 0 or te.sum() == 0:
            continue
        model = (Isotonic() if method == "isotonic" else Platt()).fit(df.loc[tr, prob_col].to_numpy(), df.loc[tr, event_col].to_numpy(float))
        out[te.to_numpy()] = model.predict(df.loc[te, prob_col].to_numpy())
    return out


# ----------------------------------------------------------------------------- scores
def brier(p, y): p, y = np.asarray(p, float), np.asarray(y, float); return float(np.mean((p - y) ** 2))
def logloss(p, y, eps=1e-3):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); y = np.asarray(y, float); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float); edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))
def reliability(p, y, bins=10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float); edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = [{"bin": f"{edges[b]:.1f}–{edges[b+1]:.1f}", "n": int((idx == b).sum()), "mean_pred": float(p[idx == b].mean()) if (idx == b).any() else np.nan,
             "observed": float(y[idx == b].mean()) if (idx == b).any() else np.nan} for b in range(bins)]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- weekly alert statistics
def weekly_table(rows: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """Township-week aggregation: max probability and any event within the week."""
    r = rows.copy(); r["week"] = r["origin"].dt.to_period("W")
    return r.groupby(["series", "week"]).agg(prob=(prob_col, "max"), event=("event", "max")).reset_index()


def sweep(wk: pd.DataFrame, grid) -> pd.DataFrame:
    ev, pr = wk.event.to_numpy(bool), wk.prob.to_numpy(float); out = []
    for p in grid:
        a = pr >= p; tp, fp, fn, tn = (a & ev).sum(), (a & ~ev).sum(), (~a & ev).sum(), (~a & ~ev).sum()
        out.append({"p_star": p, "alerts": int(tp + fp), "alert_rate_per_site_week": float((tp + fp) / len(wk)), "sensitivity": tp / max(tp + fn, 1),
                    "false_alarm_per100": 100 * fp / max(fp + tn, 1), "ppv": tp / max(tp + fp, 1) if tp + fp else np.nan, "n_weeks": len(wk), "event_weeks": int(ev.sum())})
    return pd.DataFrame(out)


def episode_leads(ep: list, prob_lookup: dict, series: list, dates, p_star: float, window: int = 14) -> np.ndarray:
    """Lead (days) from the earliest alerting origin in the window to each crossing; NaN if never alerted."""
    leads = []
    for s_i, t in ep:
        ks = [k for k in range(1, window + 1) if prob_lookup.get((series[s_i], dates[t - k]), -1.0) >= p_star]
        leads.append(max(ks) if ks else np.nan)
    return np.array(leads, float)


def two_tier_table(wk: pd.DataFrame, ep: list, prob_lookup: dict, series: list, dates, pairs, n_sites: int = 20, window: int = 14) -> pd.DataFrame:
    """For each (watch, alert) pair: weekly workload per tier, sensitivity, lead, and escalation behaviour."""
    ev, pr = wk.event.to_numpy(bool), wk.prob.to_numpy(float); rows = []
    for pw, pa in pairs:
        w, a = pr >= pw, pr >= pa
        lw, la = episode_leads(ep, prob_lookup, series, dates, pw, window), episode_leads(ep, prob_lookup, series, dates, pa, window)
        both = ~np.isnan(lw) & ~np.isnan(la)
        rows.append({"watch_p": pw, "alert_p": pa,
                     "watch_flags_per_site_week": float(w.mean()), "alert_orders_per_site_week": float(a.mean()),
                     "watch_only_per_site_week": float((w & ~a).mean()),
                     f"watch_flags_week_{n_sites}": float(w.mean() * n_sites), f"alert_orders_week_{n_sites}": float(a.mean() * n_sites),
                     f"false_orders_week_{n_sites}": float((a & ~ev).mean() * n_sites),
                     "watch_sensitivity": float((w & ev).sum() / max(ev.sum(), 1)), "alert_sensitivity": float((a & ev).sum() / max(ev.sum(), 1)),
                     "alert_ppv": float((a & ev).sum() / max(a.sum(), 1)), "alert_false_per100": float(100 * (a & ~ev).sum() / max((~ev).sum(), 1)),
                     "watch_false_per100": float(100 * (w & ~ev).sum() / max((~ev).sum(), 1)),
                     "events_detected_watch": float(np.mean(~np.isnan(lw))), "events_detected_alert": float(np.mean(~np.isnan(la))),
                     "lead_median_watch": float(np.nanmedian(lw)) if np.any(~np.isnan(lw)) else np.nan, "lead_median_alert": float(np.nanmedian(la)) if np.any(~np.isnan(la)) else np.nan,
                     "watch_earlier_than_alert": float(np.mean(lw[both] > la[both])) if both.any() else np.nan,   # share of alerted events where the watch fired at an earlier origin
                     "days_watch_to_alert_mean": float(np.mean((lw - la)[both])) if both.any() else np.nan,
                     "days_watch_to_alert_median_if_earlier": float(np.median((lw - la)[both][(lw - la)[both] > 0])) if (both.any() and ((lw - la)[both] > 0).any()) else np.nan,
                     "watch_only_events_detected": float(np.mean(~np.isnan(lw) & np.isnan(la)))})
    return pd.DataFrame(rows)
