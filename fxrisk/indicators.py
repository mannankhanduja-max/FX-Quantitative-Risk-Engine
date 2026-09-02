"""
VWAP and its EMA, on daily bars with real volume.

This module exists because the universe moved to ETFs. On FX spot
Yahoo reports Volume = 0, so a volume-weighted average price is
not merely noisy - it is undefined, and any code that computes one
is either dividing by zero or silently substituting equal weights.
`fxrisk/data/histdata.py` documents the same problem for tick
files and refuses to fabricate a weight. ETFs report actual share
volume, so everything below is real.

WHAT VWAP MEANS ON A DAILY BAR
-------------------------------
Textbook VWAP is intraday: every trade weighted by its size,
reset each session. On daily bars the best available analogue is a
rolling VWAP - typical price (H+L+C)/3, weighted by that day's
volume, over a trailing window:

    VWAP_t = sum(TP_i * V_i) / sum(V_i)   for i in the last N days

That is a genuine volume weighting, but it is not the intraday
benchmark a trader means by "VWAP", and it should not be presented
as an execution benchmark. It is a volume-weighted trend line.

THE 9-PERIOD EMA
----------------
    EMA_t = alpha * VWAP_t + (1 - alpha) * EMA_{t-1},  alpha = 2/(span+1)

Span 9 gives alpha = 0.2 and a centre of mass of 4 days, so the
EMA leads a 9-day simple average and reacts roughly twice as fast
as the 20-day VWAP underneath it. The pair is used as a
trend/momentum read: EMA above VWAP is short-term strength against
the volume-weighted base.

NO LOOK-AHEAD. Every series here is causal - `pandas.ewm` with
`adjust=False` and a trailing `rolling` window use only current
and past bars. `signal()` additionally shifts by one bar so a
signal for day t is knowable at the close of t-1, which is the
form a backtest can use without cheating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def typical_price(bars: pd.DataFrame) -> pd.Series:
    """(High + Low + Close) / 3."""
    for col in ("High", "Low", "Close"):
        if col not in bars.columns:
            raise ValueError(f"bars is missing the '{col}' column")
    return (bars["High"] + bars["Low"] + bars["Close"]) / 3.0


def rolling_vwap(
    bars: pd.DataFrame, window: int = 20, volume_col: str = "Volume"
) -> pd.Series:
    """
    Volume-weighted average price over a trailing window.

    Raises if volume is absent or entirely non-positive rather than
    falling back to an equal-weighted average. A TWAP dressed as a
    VWAP is the specific failure this codebase refuses.
    """
    if volume_col not in bars.columns:
        raise ValueError(f"no '{volume_col}' column - VWAP needs volume")

    vol = pd.to_numeric(bars[volume_col], errors="coerce").fillna(0.0)
    if (vol <= 0).all():
        raise ValueError(
            "volume is zero on every bar, so VWAP is undefined. This is the "
            "case for FX spot on Yahoo and for HistData files. Use an "
            "instrument that reports real volume (the ETF universe does), "
            "or use a TWAP and call it one."
        )

    tp = typical_price(bars)
    num = (tp * vol).rolling(window, min_periods=window).sum()
    den = vol.rolling(window, min_periods=window).sum().replace(0, np.nan)
    return (num / den).rename(f"vwap_{window}")


def ema(series: pd.Series, span: int = 9) -> pd.Series:
    """
    Exponential moving average, causal.

    `adjust=False` gives the recursive form above, which is what a
    charting package computes. `adjust=True` would reweight the
    early history and is not what "9 EMA" means to a trader.
    """
    return (
        pd.Series(series, dtype="float64")
        .ewm(span=span, adjust=False)
        .mean()
        .rename(f"ema_{span}")
    )


def vwap_ema(
    bars: pd.DataFrame,
    window: int = 20,
    span: int = 9,
    volume_col: str = "Volume",
) -> pd.DataFrame:
    """
    Rolling VWAP, its EMA, and the gap between them.

    Columns
    -------
    vwap        volume-weighted average price over `window` bars
    ema         `span`-period EMA of that VWAP
    deviation   (close / vwap - 1), how far price sits from the base
    ema_gap     (ema / vwap - 1), the trend read
    """
    vwap = rolling_vwap(bars, window=window, volume_col=volume_col)
    e = ema(vwap, span=span)

    return pd.DataFrame(
        {
            "vwap": vwap,
            "ema": e,
            "deviation": bars["Close"] / vwap - 1.0,
            "ema_gap": e / vwap - 1.0,
        }
    )


def signal(
    bars: pd.DataFrame,
    window: int = 20,
    span: int = 9,
    volume_col: str = "Volume",
) -> pd.Series:
    """
    +1 when the EMA sits above the VWAP, -1 below, 0 while either
    is still forming.

    SHIFTED BY ONE BAR. The value at date t is computed from data
    through t-1, so it is knowable at the close of t-1 and can be
    applied to the return on t. Without the shift a backtest would
    be trading on a close it has not yet seen - the most common way
    a strategy result becomes fictional.
    """
    frame = vwap_ema(bars, window=window, span=span, volume_col=volume_col)
    raw = np.sign(frame["ema_gap"]).fillna(0.0)
    return raw.shift(1).fillna(0.0).rename("vwap_ema_signal")


def summary(frame: pd.DataFrame) -> str:
    """One-paragraph read of a vwap_ema frame."""
    valid = frame.dropna()
    if valid.empty:
        return "  no valid observations - window longer than the sample?"

    above = float((valid["ema_gap"] > 0).mean())
    return "\n".join(
        [
            f"  observations          {len(valid)}",
            f"  EMA above VWAP        {above:.1%} of days",
            f"  mean |deviation|      {valid['deviation'].abs().mean():.3%}",
            f"  max deviation         {valid['deviation'].max():+.2%}",
            f"  min deviation         {valid['deviation'].min():+.2%}",
        ]
    )
