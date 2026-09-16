"""Tables and figures for backtest results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import OUTPUT_DIR
from .weeks import week_start

plt.rcParams["font.family"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def leaderboard(summary: pd.DataFrame, segment: str = "post_covid", hs=(1, 2, 3, 4), target: str | None = None) -> pd.DataFrame:
    s = summary[(summary.segment == segment) & (summary.h.isin(hs))]
    if target is not None and "target" in s.columns:
        s = s[s.target == target]
    tab = s.groupby("config").agg(wis=("wis", "mean"), mae=("mae", "mean"), mape=("mape", "mean"), mase=("mase", "mean"),
                                  cov80=("cov80", "mean"), cov60=("cov60", "mean"),
                                  rel_wis=("rel_wis_vs_snaive", "mean")).sort_values("wis")
    return tab


def wis_by_horizon_table(summary: pd.DataFrame, segment: str, target: str | None = None) -> pd.DataFrame:
    s = summary[summary.segment == segment]
    if target is not None and "target" in s.columns:
        s = s[s.target == target]
    return s.pivot(index="config", columns="h", values="wis").sort_values(1)


def plot_wis_by_horizon(summary: pd.DataFrame, segment: str, path: Path, title: str = "", target: str | None = None) -> None:
    s = summary[summary.segment == segment]
    if target is not None and "target" in s.columns:
        s = s[s.target == target]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cfg, g in s.groupby("config"):
        ax.plot(g["h"], g["wis"], marker="o", label=cfg)
    ax.set_xlabel("horizon (weeks)"); ax.set_ylabel("mean WIS"); ax.set_title(title or f"WIS by horizon ({segment})")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_fan_over_time(results: pd.DataFrame, config: str, h: int, path: Path, start_year: int = 2023, title: str = "") -> None:
    """Truth vs h-step-ahead median with 80% band over time."""
    r = results[(results.config == config) & (results.h == h) & (results.target_year >= start_year)].copy()
    r["date"] = [week_start(w) for w in r["target_yw"]]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(r["date"], r["q10"], r["q90"], alpha=.25, label="80% PI")
    ax.plot(r["date"], r["median"], label=f"median (h={h})")
    ax.plot(r["date"], r["y"], color="black", lw=1.2, label="actual")
    ax.set_title(title or f"{config}: {h}-week-ahead forecasts"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_latest_forecast(history: pd.Series, med: np.ndarray, q: np.ndarray, target_yws: list[str], path: Path,
                         title: str = "", n_hist: int = 104) -> None:
    hist = history.iloc[-n_hist:]
    hd = [week_start(w) for w in hist.index]
    fd = [week_start(w) for w in target_yws]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hd, hist.values, color="black", lw=1.2, label="observed")
    ax.fill_between(fd, q[:, 0], q[:, 8], alpha=.25, label="80% PI")
    ax.fill_between(fd, q[:, 1], q[:, 7], alpha=.25, label="60% PI")
    ax.plot(fd, med, marker="o", label="median")
    ax.set_title(title); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
