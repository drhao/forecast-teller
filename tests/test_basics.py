"""Quick sanity checks (run: python tests/test_basics.py). No pytest dependency."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller.baselines import naive_last, seasonal_naive  # noqa: E402
from forecast_teller.covariates import future_covariates, holiday_weekly, seasonal_features  # noqa: E402
from forecast_teller.metrics import covered, wis  # noqa: E402
from forecast_teller.weeks import next_weeks, seasonal_phase, week_start, yw_of_date, yw_range  # noqa: E402

# weeks
yws = yw_range("201601", "202553")
assert len(yws) == 522, len(yws)
assert yws[0] == "201601" and yws[-1] == "202553"
assert next_weeks("202553", 2) == ["202601", "202602"]
assert str(week_start("202505").date()) == "2025-01-26"
assert yw_of_date("2026-02-17") == "202607"
ph = seasonal_phase(yws)
assert ph.min() >= 0 and ph.max() < 1
assert abs(seasonal_phase(["202053"])[0] - (52.5 / 53)) < 1e-9  # 53-week year handled

# covariates
hw = holiday_weekly(yws + next_weeks("202553", 10))
assert hw.loc["202505", "cny_week"] == 1 and hw.loc["202505", "holiday_days"] == 5
assert hw.loc["202407", "cny_week"] == 1 and hw.loc["202607", "cny_week"] == 1
assert hw.loc["202510", "cny_week"] == 0
pf = future_covariates(yws, 4, kinds=("season", "cny", "holiday"))
assert pf.shape == (4, 526) and pf.dtype == np.float32
assert np.allclose(seasonal_features(yws)[0] ** 2 + seasonal_features(yws)[1] ** 2, 1, atol=1e-5)

# metrics: perfect forecast with degenerate quantiles scores 0; symmetric miss scores > 0
y = np.array([10.0, 20.0])
q = np.repeat(y[:, None], 9, axis=1)
assert np.allclose(wis(y, q), 0)
q2 = q + np.linspace(-5, 5, 9)[None, :]
assert (wis(y, q2) > 0).all() and covered(y, q2, 80).all()
assert not covered(np.array([100.0, 100.0]), q2, 80).any()

# baselines
ctx = np.abs(np.sin(np.arange(200) / 8.0)) * 100 + 5
m, qq = seasonal_naive(ctx, 4)
assert m.shape == (4,) and qq.shape == (4, 9) and (np.diff(qq, axis=1) >= 0).all()
m2, qq2 = naive_last(ctx, 4)
assert np.allclose(m2, ctx[-1], atol=50) and (qq2 >= 0).all()
print("all basic checks passed")
