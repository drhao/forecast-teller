"""Shape of a 4-week median path, for the rule-generated narratives (flu, EV, COVID).

The trend is the log-linear slope through the origin-week observation and the h = 1..4 medians, reported as an
average weekly % change, so a small wiggle at week 4 does not flip "falling" to "flat". A clear interior peak
(or trough) is reported as such before the slope is consulted.
"""
import numpy as np

FLAT_PCT = 2.0   # |average weekly change| below this → flat
FAST_PCT = 5.0   # at or above this → rising / falling; in between → slowly
TURN_PCT = 5.0   # an interior peak (trough) must sit this far above (below) both the origin and week 4


def weekly_slope_pct(last: float, meds) -> float:
    """Average weekly % change: OLS slope of log(value) on week index 0..4 (origin + 4 medians)."""
    v = np.r_[last, meds].astype(float)
    y = np.log(np.maximum(v, max(v.max() * 1e-3, 1e-9)))   # floor so a zero median (sparse COVID weeks) stays finite
    x = np.arange(len(y))
    b = np.polyfit(x, y, 1)[0]
    return 100 * (np.exp(b) - 1)


def path_shape(last: float, meds) -> dict:
    """Classify the median path. kind ∈ peak, trough, up, slow_up, flat, slow_down, down."""
    meds = np.asarray(meds, dtype=float)
    slope = weekly_slope_pct(last, meds)
    ch = 100 * (meds / last - 1)
    imax, imin = int(np.argmax(meds)), int(np.argmin(meds))
    if 0 < imax < 3 and meds[imax] >= max(last, meds[-1]) * (1 + TURN_PCT / 100):
        kind, word = "peak", "先升後降"
    elif 0 < imin < 3 and meds[imin] <= min(last, meds[-1]) * (1 - TURN_PCT / 100):
        kind, word = "trough", "先降後回升"
    elif slope >= FAST_PCT:
        kind, word = "up", "上升"
    elif slope >= FLAT_PCT:
        kind, word = "slow_up", "緩升"
    elif slope > -FLAT_PCT:
        kind, word = "flat", "大致持平"
    elif slope > -FAST_PCT:
        kind, word = "slow_down", "緩降"
    else:
        kind, word = "down", "下降"
    return {"kind": kind, "word": word, "slope_pct": float(slope), "ch": [float(c) for c in ch],
            "imax": imax, "imin": imin}


def headline_outlook(sh: dict, fc) -> str:
    """Headline clause, e.g. 模型預期未來 4 週緩降（平均每週 -2%）."""
    if sh["kind"] == "peak":
        i = sh["imax"]
        return f"模型預期第 {i + 1} 週（{fc[i]['date']} 起）前後觸頂，之後緩降"
    if sh["kind"] == "trough":
        i = sh["imin"]
        return f"模型預期第 {i + 1} 週（{fc[i]['date']} 起）前後落底，之後回升"
    return f"模型預期未來 4 週{sh['word']}（平均每週 {sh['slope_pct']:+.1f}%）"


def shape_text(sh: dict, fc, meds, fmt, unit: str) -> str:
    """Detail sentence for the outlook list, e.g. 未來 4 週緩降（平均每週 -2%），第 4 週中位數 77,507 人次（較目前 -9%）."""
    ch = sh["ch"]
    if sh["kind"] == "peak":
        i = sh["imax"]
        return (f"先升後降：中位數在第 {i + 1} 週（{fc[i]['yw']}，{fc[i]['date']} 起）達高點 {fmt(meds[i])} {unit}"
                f"（較目前 {ch[i]:+.0f}%），第 4 週回到 {fmt(meds[3])} {unit}")
    if sh["kind"] == "trough":
        i = sh["imin"]
        return (f"先降後回升：中位數在第 {i + 1} 週（{fc[i]['yw']}，{fc[i]['date']} 起）達低點 {fmt(meds[i])} {unit}"
                f"（較目前 {ch[i]:+.0f}%），第 4 週回到 {fmt(meds[3])} {unit}")
    return (f"未來 4 週{sh['word']}（平均每週 {sh['slope_pct']:+.1f}%），"
            f"第 4 週中位數 {fmt(meds[3])} {unit}（較目前 {ch[3]:+.0f}%）")


METHOD_NOTE = ("<b>未來 4 週的走勢</b>：以起點週觀測值與第 1–4 週中位數取對數做線性迴歸，換算成平均每週變化率；"
               f"每週變化在 ±{FLAT_PCT:g}% 內稱「大致持平」，{FLAT_PCT:g}–{FAST_PCT:g}% 稱「緩升／緩降」，{FAST_PCT:g}% 以上稱「上升／下降」。"
               f"若中位數在第 2–3 週出現比起點與第 4 週都高（低）{TURN_PCT:g}% 以上的高（低）點，改稱觸頂（落底）。")
