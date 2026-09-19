"""ev_forecast: Taiwan enterovirus (腸病毒) forecasting sub-project built on the forecast-teller core.

Layout: ev_forecast/{data,data_processed,outputs,scripts,src}; site output in <repo>/docs/ev/.
The shared engine (TimesFM wrapper, metrics, baselines, backtest, weeks, holiday covariates)
is imported from <repo>/src/forecast_teller.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # ev_forecast/
REPO = ROOT.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "data_processed"
OUTPUT_DIR = ROOT / "outputs"
DOCS_DIR = REPO / "docs" / "ev"
FLU_PROCESSED_DIR = REPO / "data_processed"     # influenza panel: RODS denominators live here

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
