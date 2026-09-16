"""forecast-teller: Taiwan influenza forecasting with TimesFM 3.0."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "data_processed"
OUTPUT_DIR = ROOT / "outputs"
