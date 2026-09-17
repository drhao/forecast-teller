"""dengue_ewarn: township-level dengue early-warning validation with TimesFM 3.0 (CHG scenario 1).

Independent of the influenza module: own data/, data_processed/, outputs/ under dengue_ewarn/.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../dengue_ewarn
REPO = ROOT.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "data_processed"
OUTPUT_DIR = ROOT / "outputs"
SITE_DIR = REPO / "docs" / "dengue"                 # published with the main GitHub Pages site
for _d in (DATA_DIR, PROCESSED_DIR, OUTPUT_DIR / "backtest", OUTPUT_DIR / "figures"):
    _d.mkdir(parents=True, exist_ok=True)
