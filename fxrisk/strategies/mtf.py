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

CHOOSING THE REWARD RATIO, AND THE TIME LIMIT WITH IT
------------------------------------------------------
These two are one decision, not two, and getting that wrong
produced the worst error in this study.

An 8:1 target with a 60-bar limit looked like the best
configuration measured here. It was not. At that geometry most
winners never reach the target inside the limit, so they close at
market - and those closes averaged +2.8 to +3.6R while the trades
that actually RESOLVED AT A BARRIER lost money. The headline mean
was roughly sixty lucky time-outs. The Black-Scholes benchmark
describes barrier resolutions only, so comparing it to a mean that
time exits dominate compares two different things.

So the reward ratio is chosen on the BARRIER population, with the
time limit set long enough that time exits are negligible and the
benchmark comparison is therefore valid. Measured across the four
instruments, ATR(14) stop, real spread plus 0.35bp commission:

    rr   bars    time%    win      BM       z     barrier R
   1.5     80     0.1%  39.94%  40.00%   -0.11     -0.2117
   2.0     80     0.1%  35.53%  33.33%   +4.52     -0.1448
   2.5     80     0.1%  31.80%  28.57%   +6.67     -0.0965
   3.0     80     0.1%  28.19%  25.00%   +6.68     -0.0811
   4.0     80     0.3%  22.89%  20.00%   +6.19     -0.0645
   5.0     80     0.6%  18.79%  16.67%   +4.67     -0.0816
   6.0     80     0.7%  16.58%  14.29%   +5.19     -0.0472

3:1 with an 80-bar limit is the default because it sits at the
peak of the directional evidence (z +6.68, tied with 2.5:1 and the
strongest in this study) with time exits at 0.1%, so nothing in
the number is an artefact of where the limit happened to fall.
Below 2:1 the edge disappears entirely - at 1.5:1 the realised win
rate is BELOW the benchmark. Above 4:1 the z falls and time exits
start to contaminate again.

6:1 loses slightly less money per trade (-0.047 against -0.081),
and is not the default: it buys that on weaker evidence and a
thinner barrier population, which is the same trade that made 8:1
look good. When nothing is profitable, the configuration worth
keeping is the one whose measurement is most trustworthy, not the
one that loses least.

WHAT IT WOULD TAKE. At 3:1 the gross edge is +0.128R against a
cost of 0.209R. Breakeven needs the round trip down to 62% of what
it currently is. That is the whole gap, stated as one number.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxrisk.data.intraday import session_id
from fxrisk.strategies import liquidity as lq
from fxrisk.strategies import smc
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
    stop_sigma: float = 1.0      # floor on the stop, in 5m volatility units
    stop_mode: str = "atr"       # "atr" (true range) or "sigma" (close-to-close)
    atr_period: int = 14         # Wilder's period, when stop_mode="atr"
    max_bars_30m: int = 80       # time limit, in 30m bars. Long on purpose:
                                 # see CHOOSING THE REWARD RATIO below.
    lam: float = 0.94
    vwap_filter: str = "none"    # "none" | "revert" | "trend"

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
        if self.vwap_filter not in ("none", "revert", "trend"):
            raise ValueError("vwap_filter must be 'none', 'revert' or 'trend'")
        if self.stop_mode not in ("sigma", "atr"):
            raise ValueError("stop_mode must be 'sigma' or 'atr'")
        if self.atr_period < 2:
            raise ValueError("atr_period must be at least 2")


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
               kinds=("breakout", "fakeout", "retrace"),
               zone_frame: pd.DataFrame | None = None,
               smc_cfg: "smc.SMCConfig | None" = None) -> pd.DataFrame:
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

    # Retracement CONFIRMATION. The depth band says how far price
    # came back; a fair value gap or order block says where it came
    # back TO. Passing zone_frame requires both - a strictly
    # stronger condition, so it can only remove trades.
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
                            if zone_frame is not None and not smc.in_zone(
                                    zone_frame, i, 1.0, low[i], high[i], smc_cfg):
                                continue
                            rows.append((idx[i] + step, 1.0, hi_prev[j], "retrace"))
                            break
                if np.isfinite(lo_prev[j]) and close[j] < lo_prev[j]:
                    imp = lo_prev[j] - close[j]
                    if imp > 0:
                        near = lo_prev[j] - imp * (1 - cfg.retrace_min)
                        far = lo_prev[j] - imp * (1 - cfg.retrace_max)
                        if near <= high[i] <= far:
                            if zone_frame is not None and not smc.in_zone(
                                    zone_frame, i, -1.0, low[i], high[i], smc_cfg):
                                continue
                            rows.append((idx[i] + step, -1.0, lo_prev[j], "retrace"))
                            break

    return pd.DataFrame(rows, columns=["known_at", "side", "level", "kind"])


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """
    Tick-weighted VWAP, reset at each 17:00 New York session roll.

    Cumulative to and including the current bar, which is legitimate:
    every term in it is known at that bar's close.
    """
    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    vol = pd.to_numeric(bars["Volume"], errors="coerce").fillna(0.0)
    if (vol <= 0).all():
        raise ValueError("volume is zero on every bar, so VWAP is undefined")
    sess = pd.Series(session_id(bars.index).to_numpy(), index=bars.index)
    num = (tp * vol).groupby(sess).cumsum()
    den = vol.groupby(sess).cumsum().replace(0, np.nan)
    return (num / den).rename("vwap")


def true_range(bars: pd.DataFrame) -> pd.Series:
    """
    Wilder's true range: the greater of this bar's own range and its
    gap from the previous close, in either direction.

        TR = max(high - low,
                 |high - prev_close|,
                 |low  - prev_close|)

    WHY IT IS NOT THE SAME AS THE EWMA SIGMA THIS MODULE USED
    -----------------------------------------------------------
    The EWMA sigma is built from CLOSE-TO-CLOSE log returns. It
    cannot see what happened inside a bar, and it cannot see a gap
    at all: a bar that opens well away from the previous close and
    then goes nowhere contributes almost nothing to it.

    A stop is hit by the INTRABAR extreme, not by the close. So a
    close-to-close measure systematically understates the distance
    price can travel against a position within one bar, and a stop
    floored on it is placed too tight - most of all around the
    session open and the seconds after a release, which is exactly
    where this rule set trades.

    True range is the standard fix and is what ATR is built from.
    Whether it actually helps here is a measurement, not an
    assumption: a wider stop lowers cost in units of risk but also
    moves the target further away.
    """
    high = bars["High"].astype("float64")
    low = bars["Low"].astype("float64")
    prev = bars["Close"].astype("float64").shift(1)
    tr = pd.concat([(high - low),
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    return tr.rename("true_range")


def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average true range, Wilder's smoothing, SHIFTED so that the
    value at bar t uses only bars up to t-1.

    The shift is the whole safety property. An ATR that includes
    the current bar sizes the stop using the very range the stop is
    about to be tested against.
    """
    tr = true_range(bars)
    out = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return out.shift(1).rename("atr")


def entries_5m(bars_5m: pd.DataFrame, setups: pd.DataFrame,
               bias_1h: pd.Series | None, cfg: MTFConfig | None = None) -> pd.DataFrame:
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
    # The stop floor's unit. "sigma" is the EWMA of close-to-close
    # log returns and is dimensionless, so it is multiplied by price
    # below. "atr" is already in price units and is not.
    if cfg.stop_mode == "atr":
        vol = atr(bars_5m, cfg.atr_period).to_numpy()
        vol_is_price = True
    else:
        r = np.log(bars_5m["Close"]).diff()
        vol = np.sqrt(r.pow(2).ewm(alpha=1 - cfg.lam,
                                   adjust=False).mean()).shift(1).to_numpy()
        vol_is_price = False
    sig = vol

    # bias_1h=None removes the 1h direction filter entirely - no
    # session VWAP, no 9 EMA. The setup's own side then decides,
    # and every setup becomes actionable in both directions. This
    # is not a small variant: it is the whole top of the cascade
    # switched off, and it roughly doubles the trade count, which
    # is worth having for its own sake because statistical power
    # on a 6bp effect is the binding constraint everywhere here.
    bias = (None if bias_1h is None
            else align_to(bars_5m.index, bias_1h).to_numpy())
    idx = bars_5m.index

    # VWAP AS A LOCATION FILTER, NOT A TREND FILTER
    # ----------------------------------------------
    # The 9-EMA-of-VWAP trend gate was removed from this rule because
    # it cut two thirds of the trades and the win rate went UP without
    # it. But that was VWAP used as a DIRECTION signal, and this is a
    # fade strategy - the direction is already decided by the setup.
    #
    # "revert" uses VWAP the way a fade should: as the session's
    # volume-weighted fair value. A long is only taken when price is
    # BELOW it and a short only when price is ABOVE it, so the trade
    # is always toward value rather than away from it. That is a
    # location condition, like the fair value gap, and it is the use
    # of VWAP consistent with the only effect this data confirmed -
    # short-horizon reversion.
    #
    # "trend" restores the old behaviour for comparison.
    vwap = (session_vwap(bars_5m).to_numpy()
            if cfg.vwap_filter != "none" else None)

    rows = []
    last_bar = -1
    for s in setups.sort_values("known_at").itertuples(index=False):
        start = int(idx.searchsorted(s.known_at, side="left"))
        for k in range(start, min(start + cfg.trigger_window, len(idx) - 1)):
            if k <= last_bar or not np.isfinite(sig[k]) or sig[k] <= 0:
                continue
            if bias is not None and bias[k] != s.side:
                continue
            if vwap is not None and np.isfinite(vwap[k]):
                if cfg.vwap_filter == "revert":
                    # long only below value, short only above it
                    if s.side > 0 and close[k] >= vwap[k]:
                        continue
                    if s.side < 0 and close[k] <= vwap[k]:
                        continue
                elif cfg.vwap_filter == "trend":
                    if s.side > 0 and close[k] <= vwap[k]:
                        continue
                    if s.side < 0 and close[k] >= vwap[k]:
                        continue
            # THE TRIGGER: this 5m bar closed beyond the PREVIOUS
            # 5m close, in the setup's direction. Nothing else.
            # Deliberately not "beyond the 30m setup level" - the
            # 30m bar already closed beyond it, and re-requiring
            # that would just demand the move continue, turning a
            # retracement entry into a second breakout entry.
            moved = close[k] > close[k - 1] if s.side > 0 else close[k] < close[k - 1]
            if not moved:
                continue
            entry = close[k]
            structural = low[k] if s.side > 0 else high[k]
            floor = (cfg.stop_sigma * sig[k] if vol_is_price
                     else cfg.stop_sigma * sig[k] * entry)
            d = max(abs(entry - structural), floor)
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


# ============================================================
# SESSION WINDOWS
# ============================================================
#
# Each session is defined in ITS OWN timezone, not as a fixed UTC
# offset. London and New York shift by an hour three weeks apart
# in spring and a week apart in autumn, and Tokyo does not shift
# at all, so a window written as "13:00-21:00 UTC" is the right
# window for part of the year and the wrong one for the rest.
# Converting the timestamp into the session's own clock and
# taking the local hour is correct on every date by construction.
#
# The windows are the cash hours of each centre, not the whole
# time the market is technically open:
#
#     Tokyo      09:00-17:00  Asia/Tokyo
#     London     08:00-16:30  Europe/London
#     New York   08:00-16:00  America/New_York
#
# A window is a LIQUIDITY claim, not a fitted parameter. Naming
# which centres an instrument belongs to is a statement about
# where its order flow lives - yen in Tokyo and New York, gold
# and the euro in London and New York, the Nasdaq only in New
# York - and it is the kind of restriction that can be written
# down before seeing a result. That is the whole reason it is
# admissible here: nothing else in this module gets to be chosen
# after the fact either.

SESSION_WINDOWS: dict[str, tuple[str, float, float]] = {
    "tokyo": ("Asia/Tokyo", 9.0, 17.0),
    "london": ("Europe/London", 8.0, 16.5),
    "newyork": ("America/New_York", 8.0, 16.0),
}

# Which centres each instrument is allowed to trade in.
INSTRUMENT_SESSIONS: dict[str, tuple[str, ...]] = {
    "USD/JPY": ("tokyo", "newyork"),
    "XAU/USD": ("london", "newyork"),
    "EUR/USD": ("london", "newyork"),
    "NAS100": ("newyork",),
}


def in_sessions(index: pd.DatetimeIndex, names) -> pd.Series:
    """
    True where the timestamp falls inside ANY of the named sessions.

    Union, not intersection: an instrument traded in London and New
    York is tradeable in either, and the overlap is simply in both.
    """
    idx = pd.DatetimeIndex(index)
    mask = pd.Series(False, index=idx)
    for n in names:
        if n not in SESSION_WINDOWS:
            raise ValueError(f"unknown session {n!r}; "
                             f"known: {sorted(SESSION_WINDOWS)}")
        tz, lo, hi = SESSION_WINDOWS[n]
        local = idx.tz_convert(tz)
        hour = local.hour + local.minute / 60.0
        # Weekends are already absent from the bar data, but an
        # index built by hand may not be, and a Saturday inside
        # London hours is not a London session.
        weekday = local.dayofweek < 5
        mask |= pd.Series((hour >= lo) & (hour < hi) & weekday, index=idx)
    return mask.rename("in_session")
