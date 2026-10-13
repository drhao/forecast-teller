"""Space–time kernel density forecaster for the dengue township backtest (PLAN_erakde.md §4, first version 2026-10-13).

Points  = local cases (onset day, report day, 最小統計區 centroid lon/lat → km), from the line list; the as-of selection
          (report day <= origin) is done per origin directly on the points, so the village / statistical-area resolution
          costs no memory. Weights: case count × 1/completeness(delay) in mode asof_adj (floor 0.2, as the panel does)
          × recency decay 0.5^(age / half_life); optional seasonal points from the previous years (circular time, γ).
Kernel  = 2-D Gaussian; bandwidth fixed (σ km) or per point by the RVKDE rule (Oyang et al. 2005, reconstructed from the
          paper, to be checked against the original): σ_i = β · R̄(s_i) · √π / (k_s · Γ(m/2 + 1))^(1/m), m = 2,
          R̄ = mean distance to the k_s nearest neighbours among the selected points; clipped to [σ_min, σ_max].
Eval    = village centroids (mean of the statistical-area centroids per 居住村里代碼) tagged with their township;
          the township share π_s = Σ_{v in s} λ(v) / Σ_v λ(v) approximates the integral of the density over the township,
          weighted by villages (≈ population) rather than by area.
Counts  = μ_{s,h} = (1 − α) · π_s · N̂_h + α · own7_s · g^(h/7), N̂_h = regional 7-day total × g^(h/7), g = trend factor
          (last 7 days over the previous 7, shrunk with a pseudo-count, clipped to [0.5, 2]); α = 0 is the pure spatial
          redistribution, α = 0.5 blends it with the township's own persistence.
Dist.   = negative binomial with mean μ and dispersion φ → the 9 deciles the harness expects (prob_ge, WIS, AUC...).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import nbinom

from .alerts import Q_LEVELS
from .data import DMAX, completeness

LAT0, LON0 = 23.0, 120.5
KM_LAT = 111.0
KM_LON = 111.32 * np.cos(np.radians(LAT0))


def to_km(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(lon, float) - LON0) * KM_LON, (np.asarray(lat, float) - LAT0) * KM_LAT


@dataclass(frozen=True)
class KDEParams:
    name: str = "kde_rv"
    bandwidth: str = "rvkde"      # "fixed" | "rvkde"
    sigma_km: float = 2.0         # fixed bandwidth
    k_s: int = 10                 # RVKDE neighbours
    beta: float = 1.0             # RVKDE relaxation
    sigma_min: float = 0.5
    sigma_max: float = 10.0
    half_life: float = 14.0       # days, recency decay
    window: int = 60              # days of history used for the density
    alpha: float = 0.0            # blend with own-township persistence (0 = pure spatial)
    season_gamma: float = 0.0     # weight of the same-season points from previous years (0 = off)
    season_days: int = 21         # ± days around the same calendar date
    season_years: int = 2
    trend: bool = True
    phi: float = 2.0              # NB dispersion (var = μ + μ² / φ)
    pseudo: float = 5.0           # pseudo-count in the trend ratio
    c_floor: float = 0.2


VARIANTS = {
    "kde_fix": KDEParams(name="kde_fix", bandwidth="fixed"),
    "kde_rv": KDEParams(name="kde_rv"),
    "kde_mix": KDEParams(name="kde_mix", alpha=0.5),
    "kde_rv_season": KDEParams(name="kde_rv_season", season_gamma=0.2),
}


def load_points(ll: pd.DataFrame, series: list[str], dates: pd.DatetimeIndex) -> dict:
    """Local cases of the panel's townships with coordinates, as day indices relative to dates[0]."""
    counties = sorted({s.split("|")[0] for s in series})
    loc = ll[(~ll.imported) & ll.county.isin(counties)].copy()
    loc["key"] = loc.county + "|" + loc.township
    idx = {s: i for i, s in enumerate(series)}
    loc = loc[loc.key.isin(idx)]
    loc = loc[(loc.onset >= dates[0]) & (loc.onset <= dates[-1])]
    loc = loc.dropna(subset=["lon", "lat"])
    x, y = to_km(loc.lon.to_numpy(), loc.lat.to_numpy())
    t_on = ((loc.onset - dates[0]).dt.days).to_numpy()
    t_rep = ((loc.report - dates[0]).dt.days).to_numpy()
    s_idx = loc.key.map(idx).to_numpy()
    n = loc.n.to_numpy(float)
    # evaluation points: village centroids (mean of statistical-area centroids), township fallback
    v = loc.dropna(subset=["village_code"]).groupby("village_code").agg(x=("lon", "mean"), y=("lat", "mean"), key=("key", "first"))
    ex, ey = to_km(v.x.to_numpy(), v.y.to_numpy()); es = v.key.map(idx).to_numpy()
    missing = [i for i in range(len(series)) if i not in set(es.tolist())]
    for i in missing:
        m = s_idx == i
        if m.any():
            ex = np.append(ex, x[m].mean()); ey = np.append(ey, y[m].mean()); es = np.append(es, i)
    return {"x": x, "y": y, "t_on": t_on, "t_rep": t_rep, "s_idx": s_idx, "n": n, "ex": ex, "ey": ey, "es": es.astype(int),
            "n_villages": int(len(v)), "n_series_missing": len(missing)}


def line_list_with_coords(ll_raw: pd.DataFrame, ll: pd.DataFrame) -> pd.DataFrame:
    """Attach lon / lat (最小統計區中心點) and village code to data.load_line_list()'s frame (same row order / filter)."""
    raw = ll_raw.loc[ll.index]
    out = ll.copy()
    out["lon"] = pd.to_numeric(raw["最小統計區中心點X"], errors="coerce")
    out["lat"] = pd.to_numeric(raw["最小統計區中心點Y"], errors="coerce")
    out["village_code"] = raw["居住村里代碼"]
    return out


class KDEForecaster:
    def __init__(self, pts: dict, hist: np.ndarray, n_series: int, params: KDEParams):
        self.p = pts; self.hist = hist; self.S = n_series; self.prm = params
        self._c_cache: dict[int, np.ndarray] = {}

    # ---- helpers
    def _completeness(self, d: int) -> np.ndarray:
        if d not in self._c_cache:
            self._c_cache[d] = np.maximum(completeness(self.hist, d), self.prm.c_floor)
        return self._c_cache[d]

    def _select(self, d: int, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, y, w) of the points that feed the density at origin d."""
        p, prm = self.p, self.prm
        age = d - p["t_on"]
        m = (age >= 0) & (age < prm.window)
        if mode != "final":
            m &= p["t_rep"] <= d
        w = p["n"][m] * 0.5 ** (age[m] / prm.half_life)
        if mode == "asof_adj":
            c = self._completeness(d)
            w = w / c[np.minimum(age[m], DMAX)]
        xs, ys = p["x"][m], p["y"][m]
        if prm.season_gamma > 0:
            for yb in range(1, prm.season_years + 1):
                centre = d - 365 * yb + 7           # same calendar date, centred on the middle of the next 2 weeks
                ms = (np.abs(p["t_on"] - centre) <= prm.season_days)
                if mode != "final":
                    ms &= p["t_rep"] <= d
                xs = np.concatenate([xs, p["x"][ms]]); ys = np.concatenate([ys, p["y"][ms]])
                w = np.concatenate([w, prm.season_gamma * p["n"][ms]])
        return xs, ys, w

    def _bandwidth(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        prm = self.prm
        if prm.bandwidth == "fixed" or len(xs) <= prm.k_s:
            return np.full(len(xs), prm.sigma_km if prm.bandwidth == "fixed" else prm.sigma_max)
        xy = np.column_stack([xs, ys])
        dist, _ = cKDTree(xy).query(xy, k=prm.k_s + 1)
        rbar = dist[:, 1:].mean(axis=1)                       # mean distance to the k_s nearest neighbours (km)
        sigma = prm.beta * rbar * np.sqrt(np.pi) / np.sqrt(prm.k_s)   # RVKDE, m = 2 (Γ(2) = 1)
        return np.clip(sigma, prm.sigma_min, prm.sigma_max)

    def _intensity(self, xs, ys, w, sigma) -> np.ndarray:
        """Σ_i w_i K_{σ_i}(v − s_i) at the evaluation points, chunked over points to bound memory."""
        ex, ey = self.p["ex"], self.p["ey"]
        lam = np.zeros(len(ex))
        inv = 1.0 / (2.0 * sigma ** 2); norm = w * inv / np.pi      # w / (2π σ²)
        for i in range(0, len(xs), 4000):
            dx = ex[None, :] - xs[i:i + 4000, None]; dy = ey[None, :] - ys[i:i + 4000, None]
            lam += (norm[i:i + 4000, None] * np.exp(-(dx * dx + dy * dy) * inv[i:i + 4000, None])).sum(axis=0)
        return lam

    # ---- forecast
    def forecast(self, d: int, mode: str, daily: np.ndarray, H: int) -> np.ndarray:
        """Decile quantiles (S, H, 9) of the 7-day sum ending at d + h, h = 1..H, from the data known at origin d.

        daily: (S, d+1) daily counts as the harness sees them in this mode (final / as-of / completeness-corrected)."""
        prm = self.prm; S = self.S
        own7 = daily[:, -7:].sum(axis=1); tot7 = float(own7.sum())
        prev7 = float(daily[:, -14:-7].sum()) if daily.shape[1] >= 14 else tot7
        g = float(np.clip((tot7 + prm.pseudo) / (prev7 + prm.pseudo), 0.5, 2.0)) if prm.trend else 1.0
        xs, ys, w = self._select(d, mode)
        if len(xs) and w.sum() > 0:
            lam = self._intensity(xs, ys, w, self._bandwidth(xs, ys))
            share = np.bincount(self.p["es"], weights=lam, minlength=S)
            share = share / share.sum() if share.sum() > 0 else np.full(S, 1.0 / S)
        else:
            share = own7 / tot7 if tot7 > 0 else np.full(S, 1.0 / S)
        q = np.zeros((S, H, 9))
        for h in range(1, H + 1):
            gh = g ** (h / 7.0)
            mu = (1 - prm.alpha) * share * tot7 * gh + prm.alpha * own7 * gh
            r = prm.phi; pnb = r / (r + np.maximum(mu, 1e-9))
            qh = nbinom.ppf(Q_LEVELS[None, :], r, pnb[:, None])
            qh[mu <= 1e-9] = 0.0
            q[:, h - 1, :] = qh
        return q
