#!/usr/bin/env python
"""Build the enterovirus weekly panel from data/*.csv → data_processed/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import json  # noqa: E402

from ev_forecast import PROCESSED_DIR  # noqa: E402
from ev_forecast.panel import build_all  # noqa: E402

nat = build_all()
cov = json.loads((PROCESSED_DIR / "coverage.json").read_text(encoding="utf-8"))
print("coverage:", {k: v for k, v in cov.items() if k in ("nhi", "rods", "rods_total")})
print(nat[["ev_out", "ev_rods", "ev_oe", "ev_rods_pct", "ev_inp", "ev_thr", "ev_in_period"]].dropna(subset=["ev_out"]).tail(6).round(2).to_string())
