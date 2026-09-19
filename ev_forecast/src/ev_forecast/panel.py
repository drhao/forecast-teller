"""Weekly enterovirus panel: national indicators, age-group and county long tables, coverage JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_teller.panel import detect_edges
from forecast_teller.weeks import yw_range

from . import FLU_PROCESSED_DIR, PROCESSED_DIR
from .io import read_nhi_ev, read_rods_ev
from .thresholds import epidemic_flags, threshold_series

NHI_COLS = ["ev_out", "ev_out_total", "ev_inp", "ev_inp_total", "ev_er", "ev_er_total"]
RODS_COLS = ["ev_rods"]
LABELS = {
    "ev_oe": "全國腸病毒門急診就診人次（健保門診 + RODS 急診）",
    "ev_out": "全國腸病毒健保門診就診人次",
    "ev_rods": "RODS 急診腸病毒就診人次",
    "ev_rods_pct": "RODS 急診腸病毒就診百分比（%）",
    "ev_inp": "全國腸病毒健保住院人次",
}


def _national(nhi: pd.DataFrame, rods: pd.DataFrame, rods_total: pd.Series | None) -> tuple[pd.DataFrame, dict]:
    g = nhi.groupby(["yw", "kind"])[["cases", "total"]].sum().unstack("kind")
    nat = pd.DataFrame(index=yw_range(min(g.index.min(), rods["yw"].min()), max(g.index.max(), rods["yw"].max())))
    for kind in g["cases"].columns:
        nat[f"ev_{kind}"] = g["cases"][kind]
        nat[f"ev_{kind}_total"] = g["total"][kind]
    nat["ev_rods"] = rods.groupby("yw")["cases"].sum()
    if rods_total is not None:
        nat["rods_total"] = rods_total.reindex(nat.index)
    cov = {}
    # NHI edges from the outpatient denominator; RODS edges from the count itself (ratio 0.5) and the denominator
    s = nat["ev_out_total"].dropna(); f, l = detect_edges(s, ratio=0.6)
    cov["nhi"] = {"raw_first": str(s.index[0]), "raw_last": str(s.index[-1]), "first_complete": f, "last_complete": l}
    r = nat["ev_rods"].dropna(); rf, rl = detect_edges(r, ratio=0.5)
    cov["rods"] = {"raw_first": str(r.index[0]), "raw_last": str(r.index[-1]), "first_complete": rf, "last_complete": rl}
    if rods_total is not None:
        d = nat["rods_total"].dropna()
        cov["rods_total"] = {"first": str(d.index[0]), "last": str(d.index[-1]), "source": "influenza panel data_processed/national_weekly.parquet (RODS_RS.csv)"}
    yws = list(nat.index)

    def mask(cols, first, last):
        keep = (np.array(yws) >= first) & (np.array(yws) <= last)
        for c in cols:
            if c in nat.columns:
                nat.loc[~keep, c] = np.nan
    mask([c for c in NHI_COLS if c in nat.columns], cov["nhi"]["first_complete"], cov["nhi"]["last_complete"])
    mask(RODS_COLS, cov["rods"]["first_complete"], cov["rods"]["last_complete"])
    # derived indicators
    nat["ev_oe"] = nat["ev_out"] + nat["ev_rods"]                       # user decision: ER component = RODS counts
    if "rods_total" in nat.columns:
        nat["ev_rods_pct"] = 100 * nat["ev_rods"] / nat["rods_total"]
    nat["ev_out_per_1k"] = 1000 * nat["ev_out"] / nat["ev_out_total"]
    nat["ev_thr"] = threshold_series(yws)
    fl = epidemic_flags(nat["ev_oe"])
    nat["ev_in_period"] = fl["in_period"].reindex(nat.index)
    nat.index.name = "yw"
    return nat, cov


def _long(df: pd.DataFrame, key: str, cols: dict[str, str], yws: list[str], first: str, last: str) -> pd.DataFrame:
    g = df.groupby(["yw", key]).sum(numeric_only=True)
    idx = pd.MultiIndex.from_product([yws, sorted(df[key].unique())], names=["yw", key])
    g = g.reindex(idx).fillna(0.0).reset_index().rename(columns=cols)
    keep = (g["yw"] >= first) & (g["yw"] <= last)
    return g[keep].reset_index(drop=True)


def build_all(nhi_path=None, rods_path=None, out: Path | None = None) -> pd.DataFrame:
    out = out or PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    nhi, rods = read_nhi_ev(nhi_path), read_rods_ev(rods_path)
    flu = FLU_PROCESSED_DIR / "national_weekly.parquet"
    rods_total = pd.read_parquet(flu)["rods_total"] if flu.exists() else None
    if rods_total is not None:
        rods_total.index = rods_total.index.astype(str)
    nat, cov = _national(nhi, rods, rods_total)
    nat.to_parquet(out / "national_weekly.parquet"); nat.to_csv(out / "national_weekly.csv", encoding="utf-8-sig")
    yws = list(nat.index)
    nf, nl = cov["nhi"]["first_complete"], cov["nhi"]["last_complete"]
    rf, rl = cov["rods"]["first_complete"], cov["rods"]["last_complete"]
    o = nhi[nhi.kind == "out"]
    age_nhi = _long(o, "age", {"cases": "ev_out", "total": "ev_out_total"}, yws, nf, nl)
    age_rods = _long(rods, "age", {"cases": "ev_rods"}, yws, rf, rl)
    cty = _long(o, "county", {"cases": "ev_out", "total": "ev_out_total"}, yws, nf, nl)
    cty_r = _long(rods, "county", {"cases": "ev_rods"}, yws, rf, rl)
    cty = cty.merge(cty_r, on=["yw", "county"], how="left")
    age_nhi.to_parquet(out / "age_nhi_weekly.parquet"); age_rods.to_parquet(out / "age_rods_weekly.parquet")
    cty.to_parquet(out / "county_weekly.parquet")
    cov["n_weeks"] = len(yws); cov["age_groups_nhi"] = sorted(o["age"].unique()); cov["age_groups_rods"] = sorted(rods["age"].unique())
    cov["counties"] = int(cty["county"].nunique())
    (out / "coverage.json").write_text(json.dumps(cov, ensure_ascii=False, indent=2), encoding="utf-8")
    return nat


def load_national(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(path) if path else PROCESSED_DIR / "national_weekly.parquet")
