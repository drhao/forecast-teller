#!/usr/bin/env python
"""Build the COVID-19 weekly panel from data/ → data_processed/."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_forecast import PROCESSED_DIR  # noqa: E402
from covid_forecast.panel import build_all  # noqa: E402

nat = build_all()
cov = json.loads((PROCESSED_DIR / "coverage.json").read_text(encoding="utf-8"))
print("coverage:", {k: (v.get("last_complete") or v.get("last")) for k, v in cov.items() if isinstance(v, dict)})
cols = [c for c in ["covid_oe", "covid_out", "covid_rods", "covid_oe_alt", "covid_rods_pct", "covid_inp", "covid_severe", "covid_in_period"] if c in nat.columns]
print(nat[cols].dropna(subset=["covid_out"]).tail(6).round(2).to_string())
