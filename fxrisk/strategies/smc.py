"""
Fair value gaps and order blocks, as retracement CONFIRMATION.

WHAT THESE ARE
---------------
Both come from the "smart money concepts" vocabulary, and both
have a defensible mechanical reading underneath the branding.

FAIR VALUE GAP (FVG), also imbalance. A three-bar pattern where
the middle bar moves far enough that bar 1 and bar 3 do not
overlap:

    bullish   high[i-1] < low[i+1]     the gap is (high[i-1], low[i+1])
    bearish   low[i-1]  > high[i+1]    the gap is (high[i+1], low[i-1])

The claim is that price traded through that range so fast nobody
filled there, leaving unmatched orders, and that price tends to
return to it. Whether or not the order-flow story is true, the
OBJECT is well defined: a range no trade printed through in the
normal way. That is testable, which is all this module needs.

ORDER BLOCK (OB). The last opposite-direction candle before the
move that broke structure. For a bullish block: the last DOWN
close before a run that takes out a prior swing high. The zone is
that candle's own range.

WHY THEY GO ON THE RETRACEMENT AND NOTHING ELSE
------------------------------------------------
The retracement rule as written accepts any pullback of 33-100%
of the breakout leg. That is a depth condition and nothing more -
it says how far price came back, not where it came back TO. Every
other measured version of this strategy failed for want of
information, and a depth band carries almost none.

An FVG or an order block is a LOCATION condition: a specific
price range, fixed before the pullback started, that the
retracement either reaches or does not. It is a strictly stronger
requirement than depth, so it can only reduce the trade count -
and the only question worth asking is whether what it removes is
worse than what it keeps.

Note what it CANNOT fix. The measured problem is that this whole
family produces a 6-9bp gross edge against a spread that costs
more. A confirmation filter cannot raise the edge per trade
unless the trades it removes were systematically worse. It can,
however, cut the trade count hard, and that alone moves the
cost arithmetic - which is the honest reason to test it.

CAUSALITY
----------
An FVG at bars (i-1, i, i+1) is only knowable when bar i+1
CLOSES. An order block is only knowable when the break of
structure it precedes has happened. Both are returned indexed by
the bar that can first see them, never by the bar they formed on,
and `tests/test_smc.py` perturbs the future to prove it. This is
the single most common way these setups are backtested into
spectacular and imaginary results.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SMCConfig:
    """Geometry for both zone types."""

    min_gap_sigma: float = 0.25   # FVG width floor, in EWMA sigmas
    ob_lookback: int = 20         # bars defining "structure" for a break
    ob_max_scan: int = 10         # bars back to search for the block candle
    zone_max_age: int = 24        # bars a zone stays live
    lam: float = 0.94

    def __post_init__(self) -> None:
        if self.min_gap_sigma < 0:
            raise ValueError("min_gap_sigma cannot be negative")
        if self.ob_lookback < 3:
            raise ValueError("ob_lookback must be at least 3")
        if self.ob_max_scan < 1:
            raise ValueError("ob_max_scan must be at least 1")
        if self.zone_max_age < 1:
            raise ValueError("zone_max_age must be at least 1")


def _sigma(close: pd.Series, lam: float) -> np.ndarray:
    r = np.log(close).diff()
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()).shift(1).to_numpy()


def fair_value_gaps(bars: pd.DataFrame,
                    cfg: SMCConfig | None = None) -> pd.DataFrame:
    """
    Every FVG, stamped with the bar that can first see it.

    Columns: `known_bar` (integer position of bar i+1, the bar at
    whose CLOSE the gap is confirmed), `side` (+1 bullish / -1
    bearish), `lo`, `hi` (the gap's range), `formed_bar`.
    """
    cfg = cfg or SMCConfig()
    high = bars["High"].to_numpy(dtype="float64")
    low = bars["Low"].to_numpy(dtype="float64")
    sig = _sigma(bars["Close"], cfg.lam)
    n = len(bars)

    rows = []
    for i in range(1, n - 1):
        k = i + 1                      # the bar that completes the pattern
        if not np.isfinite(sig[k]) or sig[k] <= 0:
            continue
        floor = cfg.min_gap_sigma * sig[k] * low[i]

        if high[i - 1] < low[k] and (low[k] - high[i - 1]) >= floor:
            rows.append({"known_bar": k, "side": 1.0,
                         "lo": high[i - 1], "hi": low[k], "formed_bar": i})
        elif low[i - 1] > high[k] and (low[i - 1] - high[k]) >= floor:
            rows.append({"known_bar": k, "side": -1.0,
                         "lo": high[k], "hi": low[i - 1], "formed_bar": i})

    return pd.DataFrame(rows, columns=["known_bar", "side", "lo", "hi",
                                       "formed_bar"])


def order_blocks(bars: pd.DataFrame,
                 cfg: SMCConfig | None = None) -> pd.DataFrame:
    """
    Order blocks, stamped with the bar that can first see them.

    The block is the last opposite-close candle before the bar
    that broke the prior `ob_lookback`-bar extreme. It is knowable
    only at that breaking bar's close, which is `known_bar` - NOT
    at the block candle itself, which is `formed_bar` and sits
    several bars earlier. Indexing by `formed_bar` is the
    look-ahead that makes these backtest beautifully.
    """
    cfg = cfg or SMCConfig()
    high = bars["High"].to_numpy(dtype="float64")
    low = bars["Low"].to_numpy(dtype="float64")
    close = bars["Close"].to_numpy(dtype="float64")
    open_ = bars["Open"].to_numpy(dtype="float64")
    n = len(bars)

    hi_prev = bars["High"].rolling(cfg.ob_lookback).max().shift(1).to_numpy()
    lo_prev = bars["Low"].rolling(cfg.ob_lookback).min().shift(1).to_numpy()

    rows = []
    for k in range(cfg.ob_lookback + 1, n):
        if np.isfinite(hi_prev[k]) and close[k] > hi_prev[k]:
            # Bullish break: find the last DOWN candle before it.
            for j in range(k - 1, max(k - 1 - cfg.ob_max_scan, 0), -1):
                if close[j] < open_[j]:
                    rows.append({"known_bar": k, "side": 1.0,
                                 "lo": low[j], "hi": high[j],
                                 "formed_bar": j})
                    break
        elif np.isfinite(lo_prev[k]) and close[k] < lo_prev[k]:
            for j in range(k - 1, max(k - 1 - cfg.ob_max_scan, 0), -1):
                if close[j] > open_[j]:
                    rows.append({"known_bar": k, "side": -1.0,
                                 "lo": low[j], "hi": high[j],
                                 "formed_bar": j})
                    break

    return pd.DataFrame(rows, columns=["known_bar", "side", "lo", "hi",
                                       "formed_bar"])


def zones(bars: pd.DataFrame, cfg: SMCConfig | None = None,
          use_fvg: bool = True, use_ob: bool = True) -> pd.DataFrame:
    """Both zone types in one frame, sorted by when they are known."""
    cfg = cfg or SMCConfig()
    parts = []
    if use_fvg:
        f = fair_value_gaps(bars, cfg)
        f["kind"] = "fvg"
        parts.append(f)
    if use_ob:
        o = order_blocks(bars, cfg)
        o["kind"] = "ob"
        parts.append(o)
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["known_bar", "side", "lo", "hi",
                                     "formed_bar", "kind"])
    return (pd.concat(parts, ignore_index=True)
            .sort_values("known_bar").reset_index(drop=True))


def in_zone(zone_frame: pd.DataFrame, bar: int, side: float,
            low: float, high: float,
            cfg: SMCConfig | None = None) -> bool:
    """
    Did this bar's range touch a live zone on the given side?

    A zone counts only if it was KNOWN strictly before `bar` and
    has not aged out. Allowing `known_bar == bar` would let a gap
    confirmed by this very bar's close justify an entry decided
    from the same bar - a subtle one-bar leak.
    """
    cfg = cfg or SMCConfig()
    if zone_frame.empty:
        return False
    kb = zone_frame["known_bar"].to_numpy()
    live = (kb < bar) & (kb >= bar - cfg.zone_max_age)
    live &= zone_frame["side"].to_numpy() == side
    if not live.any():
        return False
    lo = zone_frame["lo"].to_numpy()[live]
    hi = zone_frame["hi"].to_numpy()[live]
    # Overlap between [low, high] and [lo, hi].
    return bool(np.any((low <= hi) & (high >= lo)))
