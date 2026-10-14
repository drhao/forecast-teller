"""Threshold-crossing probabilities over a 1–4 week forecast path, derived from per-week quantiles.

The model returns marginal quantiles per horizon, not joint sample paths. P(any week in 1..H crosses the
threshold) therefore needs a dependence assumption:
  * comonotonic (errors move together across horizons) → P(any) = max_h p_h   (used as the headline value;
    forecast errors of one model across adjacent horizons are strongly positively correlated)
  * independent → P(any) = 1 − Π(1 − p_h)                                     (upper bound, reported as a range)
When the indicator is already above the threshold the event of interest is "drops below within H weeks":
comonotonic P = 1 − min_h p_h, independent P = 1 − Π p_h.
"""
from __future__ import annotations

import numpy as np

Q_KEYS = [f"q{k}" for k in range(10, 100, 10)]
Q_LEVELS = [k / 100 for k in range(10, 100, 10)]


def prob_ge(f: dict, thr: float) -> float:
    """P(value >= thr) from the 9 quantiles q10..q90 (linear interpolation; capped at 0.05–0.95)."""
    qs = [float(f[k]) for k in Q_KEYS]
    if thr <= qs[0]:
        return 0.95
    if thr >= qs[-1]:
        return 0.05
    for i in range(8):
        if qs[i] <= thr <= qs[i + 1]:
            frac = (thr - qs[i]) / (qs[i + 1] - qs[i]) if qs[i + 1] > qs[i] else 0.0
            return 1 - (Q_LEVELS[i] + frac * (Q_LEVELS[i + 1] - Q_LEVELS[i]))
    return 0.5


def pct5(p: float) -> int:
    """Round a probability to the nearest 5%."""
    return int(round(p * 20) * 5)


def threshold_outlook(forecast: list[dict], thr: float, last_observed: float, name: str, thr_label: str,
                      unit: str = "", decimals: int = 0, provisional: bool = False) -> dict:
    """Summary dict for the pages: event, headline probability (comonotonic), independence bound, per-week
    probabilities, first week whose own probability reaches 50%, and a ready-made sentence."""
    by_h = [prob_ge(f, thr) for f in forecast]
    above = float(last_observed) >= float(thr)
    H = len(forecast)
    if not above:
        p, p_ind = max(by_h), 1 - float(np.prod([1 - x for x in by_h]))
        event, verb = "exceed", f"超過{thr_label}"
        weekly = "各週超過的機率 " + " / ".join(f"{pct5(x)}%" for x in by_h)
        first = next((f["yw"] for f, x in zip(forecast, by_h) if x >= 0.5), None)
    else:
        p, p_ind = 1 - min(by_h), 1 - float(np.prod(by_h))
        event, verb = "drop", f"降到{thr_label}以下"
        weekly = "各週維持其上的機率 " + " / ".join(f"{pct5(x)}%" for x in by_h)
        first = next((f["yw"] for f, x in zip(forecast, by_h) if x < 0.5), None)
    label = f"未來 {H} 週內{name}{verb}的機率"
    text = (f"{label}約 {pct5(p)}%（{weekly}；以各週機率的極值近似，假設各週誤差同向；若各週獨立則約 {pct5(p_ind)}%）"
            + (f"。以中位數判斷最快在 {first} 當週{verb}。" if first else "。"))
    if provisional:
        text += "閾值為暫行水準，待疾管署正式閾值確認。"
    return {"indicator": name, "threshold": float(thr), "threshold_label": thr_label, "unit": unit, "decimals": decimals,
            "above_now": above, "event": event, "label": label, "p": round(p, 3), "p_indep": round(p_ind, 3),
            "by_h": [round(x, 3) for x in by_h], "first_week": first, "provisional": provisional, "text": text,
            "method": "各週機率取極值（comonotonic）為主值，各週獨立為上界；機率由 9 個分位數線性內插，上下限 5%–95%"}
