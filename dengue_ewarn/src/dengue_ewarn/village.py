"""Village-level hotspot backtest (PLAN_erakde §11, 2026-10-14).

Target   : for origin day d and village v, Y_v = local confirmed cases with onset in (d, d + 14]; event = Y_v >= 2 (secondary >= 3).
           Cases on or before d do not count towards the threshold (user's definition, 2026-10-14).
Universe : every named village in the NLSC boundary file (data_processed/villages.parquet); cases are assigned to villages by
           point-in-polygon on the 最小統計區 centroid (scripts/build_villages.py).
Strata   : the village's own reported cases with onset in (d - 14, d] as known at the origin: 0 / 1 / >= 2 (a village with 0
           is a potential new cluster; with 1 the question is whether a second case follows); region 台南高雄 / 其他縣市.
Modes    : final (every case with onset <= d), asof_adj (report <= d, weight 1 / completeness(delay), floor 0.2).
Models   : own    — persistence: μ_v = own14_adj (expected next-14-day count = last-14-day count) → P(N >= 2), NB(φ)
           town   — township count spread uniformly over its villages (what NIDSS township data alone would allow)
           kde_s  — Gaussian KDE (σ = s km, window 60 d, half-life 14 d) at village centroids → share × projected total
           kde_rv — RVKDE adaptive bandwidth (k = 10, β = 1, σ ∈ [0.2, 5] km)
           kde_s_mix — 0.5 × kde_s + 0.5 × own (μ blend)
           Scores are P(N >= 2) under NB(mean μ, dispersion φ = 2); AUC / top-k only need the ranking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import nbinom

from .data import DMAX, completeness

H = 14


@dataclass(frozen=True)
class VParams:
    name: str
    kind: str                 # own | town | kde
    sigma_km: float = 1.0
    adaptive: bool = False
    k_s: int = 10
    beta: float = 1.0
    sigma_min: float = 0.2
    sigma_max: float = 5.0
    mix: float = 0.0          # weight of own persistence in the μ blend
    window: int = 60
    half_life: float = 14.0
    phi: float = 2.0
    pseudo: float = 5.0


MODELS = {
    "own": VParams("own", "own"),
    "town": VParams("town", "town"),
    "kde_05": VParams("kde_05", "kde", sigma_km=0.5),
    "kde_1": VParams("kde_1", "kde", sigma_km=1.0),
    "kde_2": VParams("kde_2", "kde", sigma_km=2.0),
    "kde_rv": VParams("kde_rv", "kde", adaptive=True),
    "kde_05_mix": VParams("kde_05_mix", "kde", sigma_km=0.5, mix=0.5),
    "kde_1_mix": VParams("kde_1_mix", "kde", sigma_km=1.0, mix=0.5),
    "kde_rv_mix": VParams("kde_rv_mix", "kde", adaptive=True, mix=0.5),
}


class VillageData:
    def __init__(self, villages: pd.DataFrame, cases: pd.DataFrame, dates: pd.DatetimeIndex, hist: np.ndarray):
        vil = villages[villages.village != ""].reset_index(drop=True)
        self.vil = vil; self.V = len(vil); self.dates = dates; self.hist = hist
        self.vidx = {c: i for i, c in enumerate(vil.villcode)}
        self.ex, self.ey = vil.x_km.to_numpy(), vil.y_km.to_numpy()
        town_key = vil.county + "|" + vil.town
        self.town_codes, self.town_of = np.unique(town_key.to_numpy(), return_inverse=True)
        self.town_size = np.bincount(self.town_of)
        self.region = np.where(vil.county.isin(["台南市", "高雄市"]), "台南高雄", "其他縣市")
        c = cases[cases.villcode.isin(self.vidx)].copy()
        self.v = c.villcode.map(self.vidx).to_numpy()
        self.t_on = ((c.onset - dates[0]).dt.days).to_numpy(); self.t_rep = ((c.report - dates[0]).dt.days).to_numpy()
        self.n = c.n.to_numpy(float)
        self._c_cache: dict[int, np.ndarray] = {}
        # final daily counts per village as a sparse-ish table for the truth: dict day -> (v, n) via sorting
        order = np.argsort(self.t_on); self._on_sorted = self.t_on[order]; self._v_sorted = self.v[order]; self._n_sorted = self.n[order]

    def truth(self, d: int) -> np.ndarray:
        lo = np.searchsorted(self._on_sorted, d + 1); hi = np.searchsorted(self._on_sorted, d + H + 1)
        return np.bincount(self._v_sorted[lo:hi], weights=self._n_sorted[lo:hi], minlength=self.V)

    def _completeness(self, d: int) -> np.ndarray:
        if d not in self._c_cache:
            self._c_cache[d] = np.maximum(completeness(self.hist, d), 0.2)
        return self._c_cache[d]

    def known(self, d: int, mode: str):
        """Cases known at origin d: index mask, weights (completeness-adjusted in asof_adj), raw-count flag."""
        age = d - self.t_on
        m = (age >= 0) & (age < 400)
        if mode != "final":
            m &= self.t_rep <= d
        w = self.n[m].copy()
        if mode == "asof_adj":
            w = w / self._completeness(d)[np.minimum(age[m], DMAX)]
        return m, w, age[m]


class VillageForecaster:
    def __init__(self, data: VillageData, prm: VParams):
        self.D = data; self.p = prm

    def _intensity(self, xs, ys, w, sigma):
        D = self.D; lam = np.zeros(D.V)
        inv = 1.0 / (2.0 * sigma ** 2); norm = w * inv / np.pi
        for i in range(0, len(xs), 2000):
            dx = D.ex[None, :] - xs[i:i + 2000, None]; dy = D.ey[None, :] - ys[i:i + 2000, None]
            lam += (norm[i:i + 2000, None] * np.exp(-(dx * dx + dy * dy) * inv[i:i + 2000, None])).sum(axis=0)
        return lam

    def _bandwidth(self, xs, ys):
        p = self.p
        if not p.adaptive or len(xs) <= p.k_s:
            return np.full(len(xs), p.sigma_km if not p.adaptive else p.sigma_max)
        xy = np.column_stack([xs, ys]); dist, _ = cKDTree(xy).query(xy, k=p.k_s + 1)
        return np.clip(p.beta * dist[:, 1:].mean(axis=1) * np.sqrt(np.pi) / np.sqrt(p.k_s), p.sigma_min, p.sigma_max)

    def mu(self, d: int, mode: str, known) -> np.ndarray:
        """Expected number of cases per village with onset in (d, d + 14]."""
        D, p = self.D, self.p; m, w, age = known
        v = D.v[m]
        own14 = np.bincount(v[age < 14], weights=w[age < 14], minlength=D.V)
        prev14 = np.bincount(v[(age >= 14) & (age < 28)], weights=w[(age >= 14) & (age < 28)], minlength=D.V)
        tot14, tot_prev = own14.sum(), prev14.sum()
        g = float(np.clip((tot14 + p.pseudo) / (tot_prev + p.pseudo), 0.5, 2.0))
        if p.kind == "own":
            return own14
        if p.kind == "town":
            t14 = np.bincount(D.town_of, weights=own14, minlength=len(D.town_codes))
            return (t14 / D.town_size)[D.town_of] * g
        sel = age < p.window
        if sel.sum() == 0 or tot14 == 0:
            spatial = np.zeros(D.V)
        else:
            vv = v[sel]; ww = w[sel] * 0.5 ** (age[sel] / p.half_life)
            # points = village centroids of the cases' villages (the 統計區 centroid itself would be finer; the village
            # centroid is what the real-time degradation would also have, and keeps the point set small)
            xs, ys = D.ex[vv], D.ey[vv]
            lam = self._intensity(xs, ys, ww, self._bandwidth(xs, ys))
            share = lam / lam.sum() if lam.sum() > 0 else np.zeros(D.V)
            spatial = share * tot14 * g
        return (1 - p.mix) * spatial + p.mix * own14

    def score(self, mu: np.ndarray, k: int = 2) -> np.ndarray:
        r = self.p.phi; pnb = r / (r + np.maximum(mu, 1e-12))
        return 1.0 - nbinom.cdf(k - 1, r, pnb)
