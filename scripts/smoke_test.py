#!/usr/bin/env python
"""Load TimesFM 3.0 and forecast the national ILI outpatient series 4 weeks ahead."""
import sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_teller.covariates import future_covariates  # noqa: E402
from forecast_teller.model_timesfm3 import TimesFM3Model  # noqa: E402
from forecast_teller.panel import load_national  # noqa: E402
from forecast_teller.weeks import next_weeks  # noqa: E402

nat = load_national()
s = nat["nhi_out_ili"].loc["201601":].dropna()
yws = list(s.index)
y = s.to_numpy(np.float32)
print(f"series nhi_out_ili: {yws[0]}..{yws[-1]} n={len(y)} last values={y[-4:].astype(int).tolist()}")

m = TimesFM3Model(backend="mlx", batch_size=8)
print(f"model loaded in {m.load_seconds:.1f}s (backend={m.backend})")

H = 4
t0 = time.time()
(base_med, base_q), = m.forecast([y], H)
t1 = time.time()
pf = future_covariates(yws, H, kinds=("season", "cny"))
(cov_med, cov_q), = m.forecast([y], H, past_future=[pf])
t2 = time.time()
print(f"predict: plain {t1-t0:.2f}s, with covariates {t2-t1:.2f}s ; pf shape {pf.shape}")
print("target weeks:", next_weeks(yws[-1], H))
print("plain      median:", base_med.round(0).astype(int).tolist(), " q10:", base_q[:, 0].round(0).astype(int).tolist(), " q90:", base_q[:, 8].round(0).astype(int).tolist())
print("covariates median:", cov_med.round(0).astype(int).tolist(), " q10:", cov_q[:, 0].round(0).astype(int).tolist(), " q90:", cov_q[:, 8].round(0).astype(int).tolist())
# timing for a batch of variable-length contexts (backtest-like)
ctxs = [y[:t] for t in range(len(y) - 60, len(y) - H)]
t3 = time.time(); outs = m.forecast(ctxs, H); t4 = time.time()
print(f"batch of {len(ctxs)} contexts (no covariates): {t4-t3:.2f}s")
pfs = [pf[:, :t + H] for t in range(len(y) - 60, len(y) - H)]
t5 = time.time(); outs = m.forecast(ctxs, H, past_future=pfs); t6 = time.time()
print(f"batch of {len(ctxs)} contexts (with covariates): {t6-t5:.2f}s")
