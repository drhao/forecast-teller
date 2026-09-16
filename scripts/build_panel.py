#!/usr/bin/env python
"""Build data_processed/ panels from data/ raw CSVs."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller.panel import build_all  # noqa: E402

t0 = time.time()
res = build_all()
nat = res["national"]
print(f"built in {time.time()-t0:.0f}s | national {nat.shape} {nat.index[0]}..{nat.index[-1]}")
print(open(Path(__file__).resolve().parents[1] / "data_processed" / "qa_report.md", encoding="utf-8").read())
