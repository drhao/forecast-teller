"""Thin wrapper around TimesFM 3.0 (MLX or PyTorch backend) for count-like weekly series."""
from __future__ import annotations

import time

import numpy as np

CHECKPOINT = "google/timesfm-3.0-pytorch"


class TimesFM3Model:
    def __init__(self, backend: str = "mlx", batch_size: int = 8, checkpoint: str = CHECKPOINT,
                 device: str | None = None, compile: bool = True):
        t0 = time.time()
        self.backend = backend
        if backend == "mlx":
            from timesfm3.mlx import TimesFM3Forecaster

            self.fc = TimesFM3Forecaster.from_pretrained(checkpoint, per_core_batch_size=batch_size, compile=compile)
        elif backend == "torch":
            from timesfm3 import TimesFM3Forecaster

            self.fc = TimesFM3Forecaster.from_pretrained(checkpoint, device=device or "cpu", per_core_batch_size=batch_size)
        else:
            raise ValueError(backend)
        self.load_seconds = time.time() - t0

    def forecast(self, contexts: list[np.ndarray], horizon: int,
                 past_future: list[np.ndarray | None] | None = None,
                 past_only: list[np.ndarray | None] | None = None,
                 log1p: bool = False, symmetric: bool = False, positive: bool = True,
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns per context (median (H,), quantiles (H, 9)) in the original scale.

        contexts may be 1-D (L,) or 2-D (V, L); for 2-D the outputs are (V, H) and (V, H, 9).
        """
        ctxs = [np.asarray(c, dtype=np.float32) for c in contexts]
        if log1p:
            ctxs = [np.log1p(np.maximum(c, 0.0)) for c in ctxs]
        kwargs = dict(horizon=horizon, return_quantiles=True, use_symmetric_averaging=symmetric,
                      make_positive=positive and not log1p)
        if past_future is not None:
            kwargs["past_future_covariates"] = [None if p is None else np.asarray(p, dtype=np.float32) for p in past_future]
        if past_only is not None:
            kwargs["past_only_covariates"] = [None if p is None else np.asarray(p, dtype=np.float32) for p in past_only]
        outs = list(self.fc.predict_batch(ctxs, **kwargs))
        res = []
        for o in outs:
            q = np.asarray(o.quantiles, dtype=np.float64)
            if log1p:
                q = np.expm1(q)
            q = np.sort(q, axis=-1)
            if positive:
                q = np.maximum(q, 0.0)
            med = q[..., 4].copy()
            res.append((med, q))
        return res
