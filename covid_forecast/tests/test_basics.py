"""Quick sanity checks for the covid_forecast sub-project (run: ../.venv/bin/python tests/test_basics.py). No pytest dependency."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_forecast.thresholds import epidemic_flags, periods, threshold_for, load_thresholds  # noqa: E402
from covid_forecast.panel import _head_complete, _edges  # noqa: E402
from forecast_teller.weeks import yw_range  # noqa: E402

# provisional epidemic rule: 3 rises and >= 2x trough enters; 2 falls and < peak/2 exits
yws = yw_range("202601", "202616")
y = pd.Series([100, 90, 80, 85, 120, 170, 260, 300, 280, 250, 200, 120, 110, 100, 90, 95], index=yws, dtype=float)
fl = epidemic_flags(y)
assert not fl.loc["202605", "in_period"]           # 2 rises only (85 → 120), not yet
assert fl.loc["202606", "entered"] and fl.loc["202606", "in_period"]   # 3rd rise, 170 >= 2 x 80
assert fl.loc["202610", "in_period"]               # 250 falls 2 weeks but >= 300/2
assert fl.loc["202612", "exited"] and not fl.loc["202612", "in_period"]   # 120 < 150 after 2+ falls
assert periods(fl) == [("202606", "202611")]
assert fl.loc["202604", "level"] == 2 * 80 and fl.loc["202609", "level"] == 150

# thresholds table: unconfirmed → None
assert threshold_for("202640") is None and load_thresholds()["threshold"].isna().all()

# sparse head detection (RODS 2022w20–w34 pattern): a few rows then real counts
full = pd.Series([3, 1, np.nan, 3, np.nan, 1, 4, 2, 7560, 20514, 22694, 20426, 21755, 23832, 24404, 19508, 16585, 13163],
                 index=yw_range("202220", "202237"), dtype=float)
assert _head_complete(full) == "202228"
rf, rl, f, l = _edges(full, None, asof="2022-09-20")   # 2022-09-20 is in 202239 → 202237 complete
assert (rf, rl, f, l) == ("202220", "202237", "202228", "202237")
rf, rl, f, l = _edges(full, None, asof="2022-09-14")   # inside 202237 → that week is not complete
assert l == "202236"
print("all covid_forecast checks passed")
