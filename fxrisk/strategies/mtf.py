"""
Multi-timeframe cascade: 1h bias, 30m setup, 5m trigger, 30m exit.

THE CASCADE
------------
    1h    direction    9-period EMA of the session VWAP vs the VWAP.
                       Up = longs only, down = shorts only.
    30m   setup        one of: breakout, fakeout (liquidity sweep),
                       or retracement after a breakout.
    5m    trigger      the entry bar. Fill at its close.
    30m   management   stop, target and time limit walked on 30m bars.

THE ONE MISTAKE THAT MAKES THIS WORK ON PAPER
-----------------------------------------------
A 1-hour bar that spans 10:00-11:00 is only COMPLETE at 11:00. Its
EMA, its VWAP, its high and low are unknown at 10:05. A 5-minute
entry at 10:05 that consults "the current 1h bar" is reading eleven
minutes into its own future, and on a trending day that single lookup
is worth more than any strategy in this repository.

So every higher-timeframe value here is taken from the last bar that
had CLOSED at or before the trigger bar's own close. `align_to`
does that with a merge_asof on the higher frame shifted by one bar,
and `tests/test_mtf.py` proves it by perturbing a 1h bar and checking
that no 5m decision inside that hour moves.

The same applies to the 30m setup: a setup identified on the
10:00-10:30 bar is actionable from 10:30 onward, never from 10:05.

WHY THIS COMPOSITION IS WORTH TESTING AT ALL
----------------------------------------------
Every single-timeframe version of these ideas has now been measured
and sits at or below the Black-Scholes coin flip. The honest prior is
that stacking them changes little: three filters that each know
nothing still know nothing together.

What a cascade CAN do, and the reason it is not merely another
variant, is change the trade population rather than the entry logic -
a 5-minute trigger inside a 30-minute setup gives a much tighter
stop for the same structural level, and cost in units of risk is
what has sunk most versions here. Whether that helps is exactly what
the benchmark measures: a tighter stop lowers the dollar loss per
trade AND raises the hurdle, so the two effects fight, and only the
data settles it.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxrisk.data.intraday import session_id
from fxrisk.strategies import liquidity as lq
from fxrisk.strategies import sweeps as sw


@dataclass
class MTFConfig:
    """Each timeframe's job, and the geometry it uses."""

    bias_ema: int = 9            # EMA span on the 1h session VWAP
    setup_lookback: int = 20     # 30m bars defining the level
    retrace_min: float = 0.33
    retrace_max: float = 1.00
    retrace_window: int = 6      # 30m bars allowed for the pullback
    trigger_window: int = 6      # 5m bars allowed to trigger after a setup
    stop_sigma: float = 1.0      # floor on the stop, in 5m EWMA sigmas
    max_bars_30m: int = 20       # time limit, in 30m bars
    lam: float = 0.94

    def __post_init__(self) -> None:
        if self.bias_ema < 1:
            raise ValueError("bias_ema must be at least 1")
        if self.setup_lookback < 3:
            raise ValueError("setup_lookback must be at least 3")
        if not 0.0 < self.retrace_min < self.retrace_max:
            raise ValueError("need 0 < retrace_min < retrace_max")
        if self.trigger_window < 1 or self.retrace_window < 1:
            raise ValueError("windows must be at least 1")
        if self.stop_sigma <= 0:
            raise ValueError("stop_sigma must be positive")


def align_to(lower_index: pd.DatetimeIndex, higher: pd.Series) -> pd.Series:
    """
    Value of the last COMPLETED higher-timeframe bar, per lower bar.

    A bar stamped 10:00 on a 1h frame covers 10:00-11:00 and closes
    at 11:00, so it may inform a 5m bar only from 11:00 onward. The
    shift below is that rule, and removing it is the multi-timeframe
    look-ahead this module exists to avoid.
    """
    h = pd.Series(higher).dropna()
    if h.empty:
        return pd.Series(np.nan, index=lower_index)

    # Stamp each higher bar with its CLOSE time, then take the most
    # recent one at or before each lower bar's own timestamp.
    step = h.index.to_series().diff().median()
    closes = h.index + step
    frame = pd.DataFrame({"t": closes, "v": h.to_numpy()}).sort_values("t")
    low = pd.DataFrame({"t": lower_index}).sort_values("t")
    out = pd.merge_asof(low, frame, on="t", direction="backward")
    return pd.Series(out["v"].to_numpy(), index=lower_index)


def hourly_bias(bars_1h: pd.DataFrame, cfg: MTFConfig | None = None) -> pd.Series:
    """+1 / -1 / 0 from the 9 EMA of the session VWAP, on 1h bars."""
    cfg = cfg or MTFConfig()
    tp = (bars_1h["High"] + bars_1h["Low"] + bars_1h["Close"]) / 3.0
    vol = pd.to_numeric(bars_1h["Volume"], errors="coerce").fillna(0.0)
    if (vol <= 0).all():
        raise ValueError("volume is zero on every bar, so VWAP is undefined")

    sess = pd.Series(session_id(bars_1h.index).to_numpy(), index=bars_1h.index)
    vwap = (tp * vol).groupby(sess).cumsum() / vol.groupby(sess).cumsum().replace(0, np.nan)
    ema = vwap.ewm(span=cfg.bias_ema, adjust=False).mean()
    return np.sign(ema - vwap).fillna(0.0).rename("bias")


def setups_30m(bars_30m: pd.DataFrame, cfg: MTFConfig | None = None,
               kinds=("breakout", "fakeout", "retrace")) -> pd.DataFrame:
    """
    Setups on the 30m frame, each stamped with the time it is KNOWN.

    Returns one row per setup: `known_at` (the 30m bar's close),
    `side`, `level`, and `kind`. Nothing here decides an entry - the
    5m frame does that.
    """
    cfg = cfg or MTFConfig()
    high = bars_30m["High"].to_numpy(dtype="float64")
    low = bars_30m["Low"].to_numpy(dtype="float64")
    close = bars_30m["Close"].to_numpy(dtype="float64")
    idx = bars_30m.index
    step = idx.to_series().diff().median()

    hi_prev = bars_30m["High"].rolling(cfg.setup_lookback).max().shift(1).to_numpy()
    lo_prev = bars_30m["Low"].rolling(cfg.setup_lookback).min().shift(1).to_numpy()
    swept = sw.detect(bars_30m, sw.SweepConfig(lookback=cfg.setup_lookback))["sweep"].to_numpy()

    rows = []
    n = len(bars_30m)
    for i in range(cfg.setup_lookback + 1, n):
        if "breakout" in kinds:
            if np.isfinite(hi_prev[i]) and close[i] > hi_prev[i]:
                rows.append((idx[i] + step, 1.0, hi_prev[i], "breakout"))
            elif np.isfinite(lo_prev[i]) and close[i] < lo_prev[i]:
                rows.append((idx[i] + step, -1.0, lo_prev[i], "breakout"))

        if "fakeout" in kinds and swept[i] != 0.0:
            rows.append((idx[i] + step, float(swept[i]), np.nan, "fakeout"))

        if "retrace" in kinds:
            # A pullback into the band after a breakout within the window.
            for j in range(max(i - cfg.retrace_window, cfg.setup_lookback + 1), i):
                if np.isfinite(hi_prev[j]) and close[j] > hi_prev[j]:
                    imp = close[j] - hi_prev[j]
                    if imp > 0:
                        near = hi_prev[j] + imp * (1 - cfg.retrace_min)
                        far = hi_prev[j] + imp * (1 - cfg.retrace_max)
                        if far <= low[i] <= near:
                            rows.append((idx[i] + step, 1.0, hi_prev[j], "retrace"))
                            break
                if np.isfinite(lo_prev[j]) and close[j] < lo_prev[j]:
                    imp = lo_prev[j] - close[j]
                    if imp > 0:
                        near = lo_prev[j] - imp * (1 - cfg.retrace_min)
                        far = lo_prev[j] - imp * (1 - cfg.retrace_max)
                        if near <= high[i] <= far:
                            rows.append((idx[i] + step, -1.0, lo_prev[j], "retrace"))
                            break

    return pd.DataFrame(rows, columns=["known_at", "side", "level", "kind"])


def entries_5m(bars_5m: pd.DataFrame, setups: pd.DataFrame,
               bias_1h: pd.Series, cfg: MTFConfig | None = None) -> pd.DataFrame:
    """
    The 5m trigger: first bar closing in the setup's direction, with
    the completed 1h bias agreeing.

    Stop goes beyond the trigger bar's own extreme, floored at
    `stop_sigma` 5m sigmas - the tight stop a fine trigger timeframe
    is supposed to buy.
    """
    cfg = cfg or MTFConfig()
    if setups.empty:
        return pd.DataFrame()

    close = bars_5m["Close"].to_numpy(dtype="float64")
    high = bars_5m["High"].to_numpy(dtype="float64")
    low = bars_5m["Low"].to_numpy(dtype="float64")
    r = np.log(bars_5m["Close"]).diff()
    sig = np.sqrt(r.pow(2).ewm(alpha=1 - cfg.lam, adjust=False).mean()).shift(1).to_numpy()

    bias = align_to(bars_5m.index, bias_1h).to_numpy()
    idx = bars_5m.index

    rows = []
    last_bar = -1
    for s in setups.sort_values("known_at").itertuples(index=False):
        start = int(idx.searchsorted(s.known_at, side="left"))
        for k in range(start, min(start + cfg.trigger_window, len(idx) - 1)):
            if k <= last_bar or not np.isfinite(sig[k]) or sig[k] <= 0:
                continue
            if bias[k] != s.side:
                continue
            moved = close[k] > close[k - 1] if s.side > 0 else close[k] < close[k - 1]
            if not moved:
                continue
            entry = close[k]
            structural = low[k] if s.side > 0 else high[k]
            d = max(abs(entry - structural), cfg.stop_sigma * sig[k] * entry)
            rows.append({"time": idx[k], "side": s.side, "entry": entry,
                         "stop": entry - s.side * d, "kind": s.kind})
            last_bar = k
            break

    return pd.DataFrame(rows)


def to_30m_positions(entries: pd.DataFrame, bars_30m: pd.DataFrame) -> pd.DataFrame:
    """
    Map 5m entries onto the 30m frame the exit is walked on.

    The entry PRICE stays the 5m fill - that is what was paid. Only
    the bar index moves, to the first 30m bar that closes strictly
    after the trigger, so the exit walk never re-uses the half-hour
    the entry happened in.
    """
    if entries.empty:
        return pd.DataFrame()
    idx = bars_30m.index
    step = idx.to_series().diff().median()
    closes = idx + step

    rows = []
    for e in entries.itertuples(index=False):
        # Search on the tz-aware DatetimeIndex directly. Going via
        # np.datetime64 silently drops the timezone and then compares
        # a naive stamp against aware ones, which raises on some
        # pandas versions and - worse - would quietly compare wrong
        # instants on others.
        j = int(closes.searchsorted(pd.Timestamp(e.time), side="right"))
        if j >= len(idx) - 1:
            continue
        rows.append({"bar": j, "side": e.side, "entry": e.entry,
                     "stop": e.stop, "kind": e.kind, "time": e.time})
    return pd.DataFrame(rows)
