"""Weekly COVID-19 panel: national indicators, region / age / county long tables, coverage JSON.

Sources merged into one yw-indexed table:
  open data NHI  → covid_out, covid_out_total, covid_inp, covid_inp_total (2021w27–)
  open data RODS → covid_rods (2022w20–); denominator rods_total from the influenza panel
  NIDSS pull     → covid_oe (門診 + 急診, the CDC press-release indicator), covid_out_nidss, covid_er_est
  NIDSS severe   → covid_severe (發病週, 新冠併發重症 2024w34–; lag 3 weeks)
  influenza panel→ resp_covid, resp_pos_pct, resp_total_pos (RESP_LAB SARS-CoV-2, 2025–)
Derived: covid_oe_alt = covid_out + covid_rods (press-release-like composite), covid_rods_pct, covid_out_per_1k,
provisional epidemic flags (covid_in_period, covid_level).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast_teller.panel import detect_edges
from forecast_teller.weeks import yw_of_date, yw_range

from . import FLU_PROCESSED_DIR, PROCESSED_DIR
from .io import read_nhi_covid, read_nidss_visits, read_rods_covid, read_severe
from .thresholds import epidemic_flags

LABELS = {
    "covid_oe": "全國新冠門急診就診人次（健保門診 + 急診，NIDSS）",
    "covid_out": "全國新冠健保門診就診人次（開放資料）",
    "covid_rods": "RODS 急診新冠就診人次",
    "covid_rods_pct": "RODS 急診新冠就診百分比（%）",
    "covid_inp": "全國新冠健保住院人次",
    "covid_oe_alt": "全國新冠門急診合成指標（開放資料門診 + RODS 急診）",
    "covid_severe": "新冠併發重症週病例數（發病週）",
}
NHI_COLS = ["covid_out", "covid_out_total", "covid_inp", "covid_inp_total"]
NIDSS_COLS = ["covid_oe", "covid_out_nidss"]
LAG = {"severe": 3}          # onset-week back-fill: drop 3 more weeks beyond the ratio test
AGE3 = {"25~64": "25_64", "65+": "65p"}   # other groups → "other"
REGIONS = ["台北區", "北區", "中區", "南區", "高屏區", "東區"]


def _head_complete(full: pd.Series, k: int = 8, ratio: float = 0.3) -> str:
    """First week from which the series is gap-free for k+1 weeks and not far below the following weeks
    (drops sparse starts such as RODS COVID-19 2022w20–w34, which has a handful of rows a week with gaps)."""
    for i in range(len(full) - k):
        win = full.iloc[i:i + k + 1]
        if win.notna().all() and win.iloc[0] >= ratio * win.iloc[1:].median():
            return str(full.index[i])
    return str(full.dropna().index[0])


def _edges(s: pd.Series, ratio: float | None, asof=None) -> tuple[str, str, str, str]:
    """(raw_first, raw_last, first_complete, last_complete). `s` is the full yw-indexed column (NaN = no rows).
    Calendar cut first (a week that has not ended is never complete); then either the ratio-based partial-week
    test from the influenza panel (ratio given) or, for count series gated by a denominator (ratio=None), the
    head-only sparse-start test with the tail taken as the last observed week."""
    nn = s.dropna()
    raw_first, raw_last = str(nn.index[0]), str(nn.index[-1])
    cur = yw_of_date(pd.Timestamp(asof) if asof is not None else pd.Timestamp.today().normalize())
    s = s[(s.index >= raw_first) & (s.index < cur)]
    if ratio is None:
        return raw_first, raw_last, _head_complete(s), str(s.dropna().index[-1])
    f, l = detect_edges(s.dropna(), ratio=ratio)
    return raw_first, raw_last, f, l


def _national(nhi, rods, nid, sev, flu: pd.DataFrame | None, asof=None) -> tuple[pd.DataFrame, dict]:
    g = nhi.groupby(["yw", "kind"])[["cases", "total"]].sum().unstack("kind")
    starts = [g.index.min(), rods["yw"].min()] + ([nid.yw.min()] if len(nid) else []) + ([sev.index.min()] if len(sev) else [])
    ends = [g.index.max(), rods["yw"].max()] + ([nid.yw.max()] if len(nid) else []) + ([sev.index.max()] if len(sev) else [])
    nat = pd.DataFrame(index=yw_range(min(starts), max(ends)))
    for kind in g["cases"].columns:
        nat[f"covid_{kind}"] = g["cases"][kind]
        nat[f"covid_{kind}_total"] = g["total"][kind]
    nat["covid_rods"] = rods.groupby("yw")["cases"].sum()
    if len(nid):
        n = nid[(nid.area == "全國") & (nid.age == "全部")]
        nat["covid_oe"] = n[n.case_type == "oe"].set_index("yw")["visits"].astype(float)
        nat["covid_out_nidss"] = n[n.case_type == "out"].set_index("yw")["visits"].astype(float)
        nat.loc[nat["covid_oe"] <= 0, ["covid_oe", "covid_out_nidss"]] = np.nan   # 2021w01–w26 and unreported weeks
    if len(sev):
        nat["covid_severe"] = sev.reindex(nat.index)
    if flu is not None:
        for c in ["rods_total", "resp_covid", "resp_pos_pct", "resp_total_pos"]:
            if c in flu.columns:
                nat[c] = flu[c].reindex(nat.index)
    cov = {}
    yws = np.array(nat.index)

    def mask(cols, first, last):
        keep = (yws >= first) & (yws <= last)
        for c in cols:
            if c in nat.columns:
                nat.loc[~keep, c] = np.nan

    rf, rl, f, l = _edges(nat["covid_out_total"], 0.6, asof)
    cov["nhi"] = {"raw_first": rf, "raw_last": rl, "first_complete": f, "last_complete": l, "source": "開放資料 NHI_COVID-19.csv（門診、住院）"}
    mask(NHI_COLS, f, l)
    # Counts fall by 20–40% a week in the tail of a wave, so the ratio test on the count itself would mark complete
    # weeks as partial. Gate the count sources by the completeness of their denominators instead: RODS by the
    # influenza panel's rods_total (same RODS feed), NIDSS by the NHI open-data total visits (same NHI claims).
    rf, rl, f, l = _edges(nat["covid_rods"], None, asof)
    if "rods_total" in nat.columns:
        l = min(l, str(nat["rods_total"].dropna().index[-1]))
    cov["rods"] = {"raw_first": rf, "raw_last": rl, "first_complete": f, "last_complete": l, "gate": "rods_total", "source": "開放資料 RODS_COVID-19.csv"}
    mask(["covid_rods"], f, l)
    if "covid_oe" in nat.columns and nat["covid_oe"].notna().any():
        rf, rl, f, l = _edges(nat["covid_oe"], None, asof)
        l = min(l, cov["nhi"]["last_complete"])
        cov["nidss"] = {"raw_first": rf, "raw_last": rl, "first_complete": f, "last_complete": l, "gate": "nhi", "source": "NIDSS 傳染病統計資料查詢系統 健保門急診（disease=11）"}
        mask(NIDSS_COLS, f, l)
    if "covid_severe" in nat.columns and nat["covid_severe"].notna().any():
        sv = nat["covid_severe"]
        first_nz = sv[sv > 0].index.min()          # the new case definition starts 2024/9/1; earlier zeros are not observations
        nat.loc[nat.index < first_nz, "covid_severe"] = np.nan
        rf, rl, f, l = _edges(nat["covid_severe"], 0.5, asof)
        li = list(nat.index).index(l) - LAG["severe"]
        l = str(nat.index[li])
        cov["severe"] = {"raw_first": rf, "raw_last": rl, "first_complete": f, "last_complete": l, "lag_weeks": LAG["severe"], "source": "NIDSS 新冠併發重症（發病週；2024/9/1 新病例定義）"}
        mask(["covid_severe"], f, l)
    if "rods_total" in nat.columns:
        d = nat["rods_total"].dropna()
        cov["rods_total"] = {"first": str(d.index[0]), "last": str(d.index[-1]), "source": "influenza panel data_processed/national_weekly.parquet (RODS_RS.csv)"}
    if "resp_covid" in nat.columns and nat["resp_covid"].notna().any():
        d = nat["resp_covid"].dropna()
        cov["resp"] = {"first": str(d.index[0]), "last": str(d.index[-1]), "source": "influenza panel (RESP_LAB.csv, SARS-CoV-2 positives)"}
    # derived
    nat["covid_oe_alt"] = nat["covid_out"] + nat["covid_rods"]
    if "covid_oe" in nat.columns:
        nat["covid_er_est"] = nat["covid_oe"] - nat["covid_out_nidss"]
    if "rods_total" in nat.columns:
        nat["covid_rods_pct"] = 100 * nat["covid_rods"] / nat["rods_total"]
    nat["covid_out_per_1k"] = 1000 * nat["covid_out"] / nat["covid_out_total"]
    primary = "covid_oe" if "covid_oe" in nat.columns and nat["covid_oe"].notna().sum() > 10 else "covid_oe_alt"
    fl = epidemic_flags(nat[primary])
    nat["covid_in_period"] = fl["in_period"].reindex(nat.index)
    nat["covid_level"] = fl["level"].reindex(nat.index)
    cov["primary"] = primary
    nat.index.name = "yw"
    return nat, cov


def _long(df: pd.DataFrame, key: str, cols: dict[str, str], yws: list[str], first: str, last: str) -> pd.DataFrame:
    g = df.groupby(["yw", key]).sum(numeric_only=True)
    idx = pd.MultiIndex.from_product([yws, sorted(df[key].unique())], names=["yw", key])
    g = g.reindex(idx).fillna(0.0).reset_index().rename(columns=cols)
    keep = (g["yw"] >= first) & (g["yw"] <= last)
    return g[keep].reset_index(drop=True)


def build_all(out: Path | None = None, asof=None) -> pd.DataFrame:
    out = out or PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    nhi, rods, nid, sev = read_nhi_covid(), read_rods_covid(), read_nidss_visits(), read_severe()
    flu_p = FLU_PROCESSED_DIR / "national_weekly.parquet"
    flu = pd.read_parquet(flu_p) if flu_p.exists() else None
    if flu is not None:
        flu.index = flu.index.astype(str)
    nat, cov = _national(nhi, rods, nid, sev, flu, asof)
    nat.to_parquet(out / "national_weekly.parquet"); nat.to_csv(out / "national_weekly.csv", encoding="utf-8-sig")
    yws = list(nat.index)
    nf, nl = cov["nhi"]["first_complete"], cov["nhi"]["last_complete"]
    rf, rl = cov["rods"]["first_complete"], cov["rods"]["last_complete"]
    o = nhi[nhi.kind == "out"]
    age_nhi = _long(o, "age", {"cases": "covid_out", "total": "covid_out_total"}, yws, nf, nl)
    age_nhi.to_parquet(out / "age_nhi_weekly.parquet")
    # three-group wide table for joint age forecasts (user decision 2026-10-13)
    a = age_nhi.assign(grp=age_nhi["age"].map(lambda g: AGE3.get(g, "other")))
    age3 = a.pivot_table(index="yw", columns="grp", values="covid_out", aggfunc="sum").reindex(yws)
    age3.columns = [f"covid_out_{c}" for c in age3.columns]
    age3.to_parquet(out / "age3_weekly.parquet")
    age_rods = _long(rods, "age", {"cases": "covid_rods"}, yws, rf, rl); age_rods.to_parquet(out / "age_rods_weekly.parquet")
    cty = _long(o, "county", {"cases": "covid_out", "total": "covid_out_total"}, yws, nf, nl)
    cty = cty.merge(_long(rods, "county", {"cases": "covid_rods"}, yws, rf, rl), on=["yw", "county"], how="left")
    cty.to_parquet(out / "county_weekly.parquet")
    if "nidss" in cov:
        xf, xl = cov["nidss"]["first_complete"], cov["nidss"]["last_complete"]
        reg = nid[(nid.age == "全部") & (nid.area.isin(REGIONS)) & (nid.yw >= xf) & (nid.yw <= xl)]
        reg = reg.pivot_table(index=["yw", "area"], columns="case_type", values="visits", aggfunc="first").reset_index().rename(columns={"oe": "covid_oe", "out": "covid_out"})
        reg.to_parquet(out / "region_weekly.parquet")
        agn = nid[(nid.area == "全國") & (nid.age != "全部") & (nid.yw >= xf) & (nid.yw <= xl)]
        agn = agn.pivot_table(index=["yw", "age"], columns="case_type", values="visits", aggfunc="first").reset_index().rename(columns={"oe": "covid_oe", "out": "covid_out"})
        agn.to_parquet(out / "age_nidss_weekly.parquet")
        cov["regions"] = REGIONS
    cov["n_weeks"] = len(yws); cov["age_groups_nhi"] = sorted(o["age"].unique()); cov["age_groups_rods"] = sorted(rods["age"].unique())
    cov["counties"] = int(cty["county"].nunique())
    (out / "coverage.json").write_text(json.dumps(cov, ensure_ascii=False, indent=2), encoding="utf-8")
    return nat


def load_national(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(Path(path) if path else PROCESSED_DIR / "national_weekly.parquet")
