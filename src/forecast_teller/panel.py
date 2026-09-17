"""Build aligned weekly panels (national / county / age) from the raw sources.

Outputs (data_processed/):
  national_weekly.parquet|csv  one row per yearweek, one column per indicator
  county_weekly.parquet        long: yw, county, indicator, value
  age_weekly.parquet           long: yw, age, indicator, value
  coverage.json                first/last complete week per source
  qa_report.md                 what was trimmed and why
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import PROCESSED_DIR
from .io import AGE_ORDER, COUNTIES, load_all
from .weeks import week_table, yw_range

# Extra weeks trimmed from the tail beyond automatic partial-week detection
# (reporting lag: NIDDS is by onset week and keeps being back-filled for weeks).
DEFAULT_LAG = {"nhi": 0, "nhi_er": 0, "rods": 0, "nidds": 3, "lab": 0, "resp": 1}  # resp: specimen-received week, allow 1 week for results

# Which national column is the "denominator" used to detect partial edge weeks.
SOURCE_DENOM = {
    "nhi": "nhi_out_total",
    "nhi_er": "nhi_er_total",
    "rods": "rods_total",
    "nidds": "nidds_severe",
    "lab": "lab_tests",
    "resp": "resp_total_pos",
}
RESP_COLS = ["resp_flu", "resp_rsv", "resp_covid", "resp_hmpv", "resp_para", "resp_rhino", "resp_adeno", "resp_myco",
             "resp_pos_pct", "resp_total_pos", "resp_flu_share", "resp_tests_est", "resp_flu_pos_rate", "resp_flu_ma3", "resp_flu_pos_rate_ma3"]
SOURCE_COLUMNS = {
    "nhi": ["nhi_out_ili", "nhi_out_total", "nhi_inp_ili", "nhi_inp_total", "nhi_out_flu", "nhi_inp_flu"],
    "nhi_er": ["nhi_er_ili", "nhi_er_total", "nhi_er_flu"],
    "rods": ["rods_ili", "rods_total"],
    "nidds": ["nidds_severe"],
    "lab": ["lab_flu_a", "lab_flu_b", "lab_flu_u", "lab_tests"],
    "resp": RESP_COLS,
}
ZERO_FILL_SOURCES = {"nidds"}  # case lists: a week with no rows is a true zero


def detect_edges(s: pd.Series, ratio: float = 0.6, k: int = 8) -> tuple[str, str]:
    """(first_complete, last_complete) of a weekly total series.

    An edge week is 'partial' when it is below `ratio` × the median of its k
    neighbouring weeks. Repeats until the edge looks normal.
    """
    s = s.dropna().sort_index()
    while len(s) > k + 1 and s.iloc[-1] < ratio * s.iloc[-(k + 1) : -1].median():
        s = s.iloc[:-1]
    while len(s) > k + 1 and s.iloc[0] < ratio * s.iloc[1 : k + 1].median():
        s = s.iloc[1:]
    return str(s.index[0]), str(s.index[-1])


def _national_parts(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts: dict[str, pd.Series] = {}
    nhi = raw["nhi_48x"]
    for vt, key in [("門診", "out"), ("住院", "inp")]:
        g = nhi[nhi["visit_type"] == vt].groupby("yw")[["cases", "total"]].sum()
        parts[f"nhi_{key}_ili"] = g["cases"]
        parts[f"nhi_{key}_total"] = g["total"]
    er = raw["nhi_48x_er"].groupby("yw")[["cases", "total"]].sum()
    parts["nhi_er_ili"] = er["cases"]
    parts["nhi_er_total"] = er["total"]
    flu = raw["nhi_487"]
    for vt, key in [("門診", "out"), ("住院", "inp")]:
        parts[f"nhi_{key}_flu"] = flu[flu["visit_type"] == vt].groupby("yw")["cases"].sum()
    parts["nhi_er_flu"] = raw["nhi_487_er"].groupby("yw")["cases"].sum()
    r = raw["rods"].groupby("yw")[["ili", "total"]].sum()
    parts["rods_ili"] = r["ili"]
    parts["rods_total"] = r["total"]
    parts["nidds_severe"] = raw["nidds"].groupby("yw")["cases"].sum()
    lt = raw["lab_types"].groupby("yw")[["flu_a", "flu_b", "flu_u"]].sum()
    for c in ["flu_a", "flu_b", "flu_u"]:
        parts[f"lab_{c}"] = lt[c]
    parts["lab_tests"] = raw["lab_tests"].groupby("yw")["tests"].sum()
    if raw.get("resp_lab") is not None:
        for c in RESP_COLS:
            parts[c] = raw["resp_lab"][c]
    df = pd.DataFrame(parts)
    df.index = df.index.astype(str)
    valid = [i for i in df.index if len(i) == 6 and i.isdigit()]
    df = df.loc[valid]
    return df.reindex(yw_range(min(df.index), max(df.index)))


def coverage_from_national(nat: pd.DataFrame, lag: dict[str, int] | None = None) -> dict[str, dict]:
    lag = {**DEFAULT_LAG, **(lag or {})}
    cov = {}
    yws = list(nat.index)
    for src, denom in SOURCE_DENOM.items():
        if denom not in nat.columns or nat[denom].dropna().empty:
            continue
        s = nat[denom].dropna()
        raw_first, raw_last = str(s.index[0]), str(s.index[-1])
        first, last = detect_edges(s, ratio=0.5 if src == "nidds" else 0.6)
        last_i = yws.index(last) - lag[src]
        cov[src] = {
            "raw_first": raw_first,
            "raw_last": raw_last,
            "first_complete": first,
            "last_complete": yws[last_i],
            "dropped_head": yws[yws.index(raw_first) : yws.index(first)],
            "dropped_tail": yws[last_i + 1 : yws.index(raw_last) + 1],
        }
    return cov


def _apply_coverage(df: pd.DataFrame, cov: dict, cols_by_source: dict[str, list[str]], yws: list[str]) -> pd.DataFrame:
    df = df.copy()
    for src, cols in cols_by_source.items():
        cols = [c for c in cols if c in df.columns]
        if not cols or src not in cov:
            continue
        lo, hi = yws.index(cov[src]["first_complete"]), yws.index(cov[src]["last_complete"])
        inside = np.zeros(len(df), dtype=bool)
        pos = df.index.get_level_values("yw") if isinstance(df.index, pd.MultiIndex) else df.index
        pos_i = pd.Index(yws).get_indexer(pos)
        inside = (pos_i >= lo) & (pos_i <= hi)
        df.loc[~inside, cols] = np.nan
        if src in ZERO_FILL_SOURCES:
            df.loc[inside, cols] = df.loc[inside, cols].fillna(0)
    return df


def add_derived(nat: pd.DataFrame) -> pd.DataFrame:
    nat = nat.copy()
    nat["nhi_out_ili_rate"] = 100 * nat["nhi_out_ili"] / nat["nhi_out_total"]
    nat["nhi_er_ili_rate"] = 100 * nat["nhi_er_ili"] / nat["nhi_er_total"]
    nat["rods_ili_pct"] = 100 * nat["rods_ili"] / nat["rods_total"]
    pos = nat["lab_flu_a"] + nat["lab_flu_b"] + nat["lab_flu_u"]
    nat["lab_pos_rate"] = 100 * pos / nat["lab_tests"]
    nat["lab_a_share"] = nat["lab_flu_a"] / pos.replace(0, np.nan)
    wt = week_table().set_index("yw")
    nat.insert(0, "week_start", wt.loc[nat.index, "week_start"].dt.date.values)
    return nat


def _long_panel(raw: dict[str, pd.DataFrame], by: str, cov: dict, yws_all: list[str]) -> pd.DataFrame:
    """by='county' or 'age'. Long format yw, <by>, indicator, value with coverage trimming."""
    frames = []

    def agg(df, val, name, src):
        g = df.groupby(["yw", by])[val].sum().rename("value").reset_index()
        g["indicator"] = name
        g["source"] = src
        frames.append(g)

    nhi = raw["nhi_48x"]
    out = nhi[nhi["visit_type"] == "門診"]
    agg(out, "cases", "nhi_out_ili", "nhi")
    agg(out, "total", "nhi_out_total", "nhi")
    agg(raw["nhi_48x_er"], "cases", "nhi_er_ili", "nhi_er")
    agg(raw["nhi_48x_er"], "total", "nhi_er_total", "nhi_er")
    agg(raw["rods"], "ili", "rods_ili", "rods")
    agg(raw["rods"], "total", "rods_total", "rods")
    agg(raw["nidds"], "cases", "nidds_severe", "nidds")
    if by == "county":
        agg(raw["lab_types"], "flu_a", "lab_flu_a", "lab")
        agg(raw["lab_types"], "flu_b", "lab_flu_b", "lab")
    long = pd.concat(frames, ignore_index=True)
    keep = COUNTIES if by == "county" else [a for a in AGE_ORDER if a != "unknown"]
    long = long[long[by].isin(keep)]
    # complete grid per (indicator) over its source coverage, so zero weeks exist
    pieces = []
    for (ind, src), g in long.groupby(["indicator", "source"]):
        lo, hi = cov[src]["first_complete"], cov[src]["last_complete"]
        yws = yw_range(lo, hi)
        grid = pd.MultiIndex.from_product([yws, keep], names=["yw", by])
        w = g.set_index(["yw", by])["value"].reindex(grid)
        # aggregated tables omit empty (week, group) cells → a missing cell inside coverage is a zero
        w = w.fillna(0)
        w = w.rename("value").reset_index()
        w["indicator"] = ind
        pieces.append(w)
    res = pd.concat(pieces, ignore_index=True)
    return res[["yw", by, "indicator", "value"]].sort_values(["indicator", by, "yw"]).reset_index(drop=True)


def yearly_summary(nat: pd.DataFrame) -> pd.DataFrame:
    y = nat.copy()
    y["year"] = y.index.str[:4]
    cols = ["nhi_out_ili", "nhi_er_ili", "rods_ili", "nidds_severe", "lab_flu_a", "lab_flu_b"]
    tot = y.groupby("year")[cols].sum(min_count=1)
    rate = y.groupby("year")[["nhi_out_ili_rate", "rods_ili_pct", "lab_pos_rate"]].mean()
    return pd.concat([tot, rate.round(2)], axis=1)


def build_all(out_dir: str | Path | None = None, lag: dict[str, int] | None = None) -> dict:
    out = Path(out_dir) if out_dir else PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)
    raw = load_all()
    nat = _national_parts(raw)
    yws = list(nat.index)
    cov = coverage_from_national(nat, lag)
    nat = _apply_coverage(nat, cov, SOURCE_COLUMNS, yws)
    nat = add_derived(nat)
    nat.index.name = "yw"
    nat.to_parquet(out / "national_weekly.parquet")
    nat.to_csv(out / "national_weekly.csv", encoding="utf-8-sig")

    county = _long_panel(raw, "county", cov, yws)
    county.to_parquet(out / "county_weekly.parquet")
    age = _long_panel(raw, "age", cov, yws)
    age.to_parquet(out / "age_weekly.parquet")

    (out / "coverage.json").write_text(json.dumps(cov, ensure_ascii=False, indent=2), encoding="utf-8")

    ys = yearly_summary(nat)
    lines = ["# Data QA report", "", "## Coverage (complete weeks per source)", "",
             "| source | raw range | complete range | dropped head | dropped tail |", "|---|---|---|---|---|"]
    for src, c in cov.items():
        lines.append(f"| {src} | {c['raw_first']}–{c['raw_last']} | {c['first_complete']}–{c['last_complete']} | "
                     f"{', '.join(c['dropped_head']) or '-'} | {', '.join(c['dropped_tail']) or '-'} |")
    lines += ["", "## Yearly totals / mean rates (national)", "", ys.to_markdown(), "",
              f"county rows: {len(county):,}  age rows: {len(age):,}", ""]
    (out / "qa_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"national": nat, "county": county, "age": age, "coverage": cov}


def load_national(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else PROCESSED_DIR / "national_weekly.parquet"
    return pd.read_parquet(p)


def load_long(kind: str, path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else PROCESSED_DIR / f"{kind}_weekly.parquet"
    return pd.read_parquet(p)
