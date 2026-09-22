"""
Liquidity sweeps: price takes out a level, then refuses to hold it.

THE IDEA, STATED SO IT CAN BE TESTED
-------------------------------------
Resting stop orders cluster just beyond obvious levels - the high of
the last N bars, the low of the session. A sweep is the bar that
reaches through such a level, triggering those stops, and then
CLOSES back inside. The reading is that the move through was not
demand arriving but supply being served: someone filled an order
against the stops and the level held.

Written as a rule, a bullish sweep of the lows is:

    low  <  lowest low of the previous `lookback` bars     (taken out)
    close >  that same level                               (reclaimed)
    penetration >= `min_pierce` sigma                      (not a graze)
    close in the upper `close_frac` of the bar's range      (rejected)

and the bearish case mirrors it. The direction of the SIGNAL is
opposite to the direction of the pierce: sweeping the lows is a
bullish event.

WHY THIS IS WORTH A TEST HERE SPECIFICALLY
--------------------------------------------
Every momentum-shaped rule in this repository has failed, and the
signal study found the opposite: over 1-24 hours these instruments
mean-revert, with rank ICs confirming out of sample. A sweep is a
mean-reversion event by construction - a failed extension - so it
sits on the side of the evidence rather than against it.

It is also the exact complement of the breakout rule already
tested. A breakout is a bar that closes BEYOND the prior extreme; a
sweep is a bar that pierces it and closes back INSIDE. The two
partition the same set of bars that touch a level, which is why
running both answers a cleaner question than either alone: is there
information in touching the level at all, and if so, which side of
it pays?

NOTHING HERE IS SHIFTED AFTER THE FACT. A sweep is identified at
the close of the bar that completes it, and any trade taken on it
fills at that close.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SweepConfig:
    """What counts as a sweep rather than an ordinary bar."""

    lookback: int = 20          # bars whose extreme defines the level
    min_pierce: float = 0.25    # how far through, in EWMA sigmas
    close_frac: float = 0.5     # close must be in this fraction of the range
    lam: float = 0.94

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if self.min_pierce < 0:
            raise ValueError("min_pierce cannot be negative")
        if not 0.0 < self.close_frac <= 1.0:
            raise ValueError("close_frac must be in (0, 1]")


def _sigma(close: pd.Series, lam: float) -> pd.Series:
    r = np.log(close).diff()
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()).shift(1)


def detect(bars: pd.DataFrame, cfg: SweepConfig | None = None) -> pd.DataFrame:
    """
    One row per bar: was this a sweep, and in which direction.

    Columns
    -------
    sweep       +1 bullish (lows swept), -1 bearish (highs swept), 0 none
    level       the level that was pierced
    pierce      how far through it price went, in sigmas
    """
    cfg = cfg or SweepConfig()
    for col in ("High", "Low", "Close"):
        if col not in bars.columns:
            raise ValueError(f"bars is missing the '{col}' column")

    high, low, close = bars["High"], bars["Low"], bars["Close"]
    # Previous extremes EXCLUDE the current bar - without the shift
    # every bar trivially touches its own extreme.
    hi_prev = high.rolling(cfg.lookback).max().shift(1)
    lo_prev = low.rolling(cfg.lookback).min().shift(1)
    sig = _sigma(close, cfg.lam)

    rng = (high - low).replace(0.0, np.nan)
    pos_in_range = (close - low) / rng          # 1 = closed on the high

    pierce_dn = (lo_prev - low) / (sig * close)
    pierce_up = (high - hi_prev) / (sig * close)

    bull = (
        (low < lo_prev)
        & (close > lo_prev)
        & (pierce_dn >= cfg.min_pierce)
        & (pos_in_range >= 1.0 - cfg.close_frac)
    )
    bear = (
        (high > hi_prev)
        & (close < hi_prev)
        & (pierce_up >= cfg.min_pierce)
        & (pos_in_range <= cfg.close_frac)
    )

    out = pd.DataFrame(index=bars.index)
    out["sweep"] = np.where(bull.fillna(False), 1.0,
                            np.where(bear.fillna(False), -1.0, 0.0))
    out["level"] = np.where(bull.fillna(False), lo_prev,
                            np.where(bear.fillna(False), hi_prev, np.nan))
    out["pierce"] = np.where(bull.fillna(False), pierce_dn,
                             np.where(bear.fillna(False), pierce_up, np.nan))
    out["sigma"] = sig
    return out


def reversal_entries(bars: pd.DataFrame, cfg: SweepConfig | None = None,
                     stop_sigma: float = 1.0,
                     beyond_extreme: bool = True) -> pd.DataFrame:
    """
    Trade the sweep itself: enter at its close, against the pierce.

    The stop goes beyond the sweep bar's own extreme - the price the
    market just rejected - with a floor of `stop_sigma` sigmas so a
    narrow sweep bar cannot produce a stop one tick wide. That floor
    matters more than it looks: a tight stop raises the cost in units
    of risk, which is what sank the early breakout variants.
    """
    cfg = cfg or SweepConfig()
    sw = detect(bars, cfg)
    close = bars["Close"].to_numpy(dtype="float64")
    high = bars["High"].to_numpy(dtype="float64")
    low = bars["Low"].to_numpy(dtype="float64")
    sig = sw["sigma"].to_numpy(dtype="float64")
    side_all = sw["sweep"].to_numpy(dtype="float64")

    rows = []
    for i in range(cfg.lookback + 1, len(bars) - 1):
        side = side_all[i]
        if side == 0.0 or not np.isfinite(sig[i]) or sig[i] <= 0:
            continue
        entry = close[i]
        structural = low[i] if side > 0 else high[i]
        d = max(abs(entry - structural), stop_sigma * sig[i] * entry)
        if not beyond_extreme:
            d = stop_sigma * sig[i] * entry
        rows.append({"bar": i, "side": side, "entry": entry,
                     "stop": entry - side * d})
    return pd.DataFrame(rows)


def recent_sweep(bars: pd.DataFrame, within: int = 12,
                 cfg: SweepConfig | None = None) -> pd.Series:
    """
    Direction of the most recent sweep, if one happened within
    `within` bars - for use as a gate on some other rule's entries.

    Carried forward, not shifted: a sweep identified at bar i is
    known from bar i's close, so it may gate an entry at bar i or
    later. Gating bar i-1 with it would be look-ahead.
    """
    s = detect(bars, cfg)["sweep"].replace(0.0, np.nan)
    return s.ffill(limit=within).fillna(0.0).rename("recent_sweep")
