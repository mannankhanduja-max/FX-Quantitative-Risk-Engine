"""
Breakout, retracement, entry on resumption. 2:1, stop to breakeven.

THE RULE, IN ORDER
-------------------
    1  TREND      the 9-period EMA of the session VWAP is above
                  the VWAP (long bias) or below it (short bias)

    2  BREAKOUT   a bar CLOSES beyond the highest high (or lowest
                  low) of the previous `lookback` bars, in the
                  same direction as the bias

    3  RETRACE    within `retrace_max_bars`, price pulls back
                  between `min_fraction` and `max_fraction` of the
                  breakout impulse, measured from the breakout
                  close back toward the level that was broken

    4  ENTRY      the first bar that closes back beyond the
                  breakout bar's extreme, in the original
                  direction. Fill at that close.

    5  BRACKET    stop at the retracement extreme (floored at
                  `stop_min_sigma` EWMA sigmas), target at 2x that
                  distance, stop to breakeven once price has
                  travelled the instrument's 10-pip equivalent.

Longs and shorts are symmetric. Nothing is asymmetric except the
data: see `fxrisk.data.intraday` on 6J=F being quoted inverted.

WHY EACH GATE EXISTS, AND WHAT IT COSTS
----------------------------------------
Gate 1 is the only part of this inherited from the VWAP/EMA work
in `fxrisk.indicators`, and it is the weakest link - that signal
was measured to have a rank IC between -0.011 and +0.003 on daily
bars, i.e. no directional content at all. It is kept here as a
FILTER rather than a signal, which is a different job: a filter
only has to be correlated with the conditions under which the
breakout works, not with returns directly. That is a lower bar,
but it is not a free pass, and the `--no-trend` variant in
`breakout_test.py` measures whether it earns its place.

Gate 2 requiring a close rather than a touch is the single most
important line in this module. A touch-based breakout fires on
every stop run and liquidity sweep, and on 15-minute bars those
are most of what looks like a breakout.

Gate 3 is where the overfitting risk lives. Two free parameters -
the retracement band and the wait - and a 60-day sample. The
defaults were set from the shape of the idea, not fitted, and
`breakout_test.py --sweep` shows the neighbourhood so the reader
can see whether the result sits on a peak or a plateau. A result
that only survives at one parameter setting is noise.

LOOK-AHEAD
-----------
Every level used to make a decision at bar t is computed from
bars strictly before t, or from bar t's own close where the
decision is explicitly a close-based one and the fill is at that
same close. The VWAP is the one place worth being careful: a
session VWAP computed with `expanding()` includes the current
bar, which is correct for a trader watching a live chart at the
bar's close and would be wrong if the fill were assumed at the
bar's open. Fills here are at the close.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxrisk.data.intraday import session_id


@dataclass
class SetupConfig:
    """Everything the entry rule can be argued about."""

    lookback: int = 20
    ema_span: int = 9
    retrace_max_bars: int = 12
    min_fraction: float = 0.33
    max_fraction: float = 1.0
    stop_min_sigma: float = 1.0
    ewma_lambda: float = 0.94
    use_trend_filter: bool = True

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if self.ema_span < 1:
            raise ValueError("ema_span must be at least 1")
        if self.retrace_max_bars < 1:
            raise ValueError("retrace_max_bars must be at least 1")
        if not 0.0 < self.min_fraction < self.max_fraction:
            raise ValueError("need 0 < min_fraction < max_fraction")
        if self.max_fraction > 2.0:
            raise ValueError("max_fraction above 2 is not a retracement")
        if self.stop_min_sigma <= 0:
            raise ValueError("stop_min_sigma must be positive")


def session_vwap(bars: pd.DataFrame, volume_col: str = "Volume") -> pd.Series:
    """
    VWAP anchored at the start of each session and reset at the next.

    This is the intraday convention and it is NOT what
    `fxrisk.indicators.rolling_vwap` computes. A rolling 20-bar
    VWAP drags yesterday's prices across today's open; a session
    VWAP is the average price everyone who traded today has paid,
    which is the thing the rule is actually referring to.

    Raises if volume is absent or dead, for the same reason
    `rolling_vwap` does: an equal-weighted average called a VWAP
    is a lie about what the number means.
    """
    if volume_col not in bars.columns:
        raise ValueError(f"no '{volume_col}' column - VWAP needs volume")

    vol = pd.to_numeric(bars[volume_col], errors="coerce").fillna(0.0)
    if (vol <= 0).all():
        raise ValueError(
            "volume is zero on every bar, so VWAP is undefined. Yahoo "
            "reports zero volume for FX spot symbols (EURUSD=X, JPY=X); "
            "use the CME futures in config.UNIVERSE_INTRADAY, which "
            "report real contract volume."
        )

    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    sess = session_id(bars.index)

    num = (tp * vol).groupby(sess).cumsum()
    den = vol.groupby(sess).cumsum().replace(0, np.nan)
    return (num / den).rename("session_vwap")


def ewma_sigma(bars: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    """Per-bar EWMA volatility of log returns, shifted so it is causal."""
    r = np.log(bars["Close"]).diff()
    var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var).shift(1).rename("sigma")


def context(bars: pd.DataFrame, cfg: SetupConfig | None = None) -> pd.DataFrame:
    """
    Everything the entry loop needs, as columns.

    `hi_prev` / `lo_prev` are the breakout levels: the extreme of
    the previous `lookback` bars, EXCLUDING the current one. The
    shift is what makes a close above `hi_prev` a breakout rather
    than a tautology.
    """
    cfg = cfg or SetupConfig()
    vwap = session_vwap(bars)
    ema = vwap.ewm(span=cfg.ema_span, adjust=False).mean()

    out = pd.DataFrame(index=bars.index)
    out["vwap"] = vwap
    out["ema"] = ema
    out["bias"] = np.sign(ema - vwap).fillna(0.0)
    out["hi_prev"] = bars["High"].rolling(cfg.lookback).max().shift(1)
    out["lo_prev"] = bars["Low"].rolling(cfg.lookback).min().shift(1)
    out["sigma"] = ewma_sigma(bars, cfg.ewma_lambda)
    out["session"] = session_id(bars.index).to_numpy()
    return out


def find_setups(bars: pd.DataFrame, cfg: SetupConfig | None = None) -> pd.DataFrame:
    """
    Walk the bars and return one row per entry the rule produces.

    Columns: bar, side, entry, stop, plus the diagnostics needed
    to see WHY each trade was taken - the breakout bar, the depth
    of the retracement, and whether the stop was set by structure
    or by the sigma floor.

    OVERLAP IS NOT HANDLED HERE. After an entry the scan resumes
    at the following bar, and a failed setup only costs one bar,
    so in a trend this function will happily emit entries that sit
    on top of one another. That is deliberate - whether two
    entries overlap depends on when the first one EXITS, which
    this function cannot know. `barriers.walk_explicit` drops them
    once it does, and reports how many. Anything that consumes
    `find_setups` directly has to deal with it, or it is counting
    one position several times.
    """
    cfg = cfg or SetupConfig()
    ctx = context(bars, cfg)

    high = bars["High"].to_numpy(dtype="float64")
    low = bars["Low"].to_numpy(dtype="float64")
    close = bars["Close"].to_numpy(dtype="float64")
    hi_prev = ctx["hi_prev"].to_numpy(dtype="float64")
    lo_prev = ctx["lo_prev"].to_numpy(dtype="float64")
    bias = ctx["bias"].to_numpy(dtype="float64")
    sigma = ctx["sigma"].to_numpy(dtype="float64")
    sess = ctx["session"].to_numpy()

    n = len(bars)
    rows = []
    i = cfg.lookback + 1

    while i < n - 1:
        up = np.isfinite(hi_prev[i]) and close[i] > hi_prev[i]
        dn = np.isfinite(lo_prev[i]) and close[i] < lo_prev[i]
        if not (up or dn):
            i += 1
            continue

        side = 1.0 if up else -1.0
        if cfg.use_trend_filter and bias[i] != side:
            i += 1
            continue

        level = hi_prev[i] if up else lo_prev[i]
        impulse = abs(close[i] - level)
        if impulse <= 0 or not np.isfinite(sigma[i]) or sigma[i] <= 0:
            i += 1
            continue

        # The band price must pull back into, in price terms.
        near = level + side * impulse * (1.0 - cfg.min_fraction)
        far = level + side * impulse * (1.0 - cfg.max_fraction)

        bo_extreme = high[i] if up else low[i]
        retrace_extreme = bo_extreme
        armed = False
        entered = False

        for j in range(i + 1, min(i + 1 + cfg.retrace_max_bars, n)):
            # A setup does not survive the session boundary. The
            # VWAP it was measured against no longer exists.
            if sess[j] != sess[i]:
                break

            if not armed:
                reach = low[j] if up else high[j]
                retrace_extreme = (
                    min(retrace_extreme, reach) if up else max(retrace_extreme, reach)
                )
                # Too deep is a failed breakout, not a pullback.
                beyond = reach < far if up else reach > far
                if beyond:
                    break
                touched = reach <= near if up else reach >= near
                if touched:
                    armed = True
                continue

            # Armed: enter on the first close back beyond the
            # breakout bar's extreme, in the original direction.
            resumed = close[j] > bo_extreme if up else close[j] < bo_extreme
            if not resumed:
                # A retracement that keeps going is a failure.
                reach = low[j] if up else high[j]
                retrace_extreme = (
                    min(retrace_extreme, reach) if up else max(retrace_extreme, reach)
                )
                beyond = reach < far if up else reach > far
                if beyond:
                    break
                continue

            entry = close[j]
            structural = retrace_extreme
            floor_d = cfg.stop_min_sigma * sigma[j] * entry
            struct_d = abs(entry - structural)
            stop_d = max(struct_d, floor_d)
            stop = entry - side * stop_d

            rows.append(
                {
                    "bar": j,
                    "time": bars.index[j],
                    "side": side,
                    "entry": entry,
                    "stop": stop,
                    "breakout_bar": i,
                    "breakout_level": level,
                    "impulse_frac": impulse / close[i],
                    "retrace_frac": abs(bo_extreme - retrace_extreme) / impulse,
                    "wait_bars": j - i,
                    "stop_from": "structure" if struct_d >= floor_d else "sigma_floor",
                }
            )
            entered = True
            i = j
            break

        i += 1

    return pd.DataFrame(rows)


def describe(setups: pd.DataFrame) -> str:
    """What the entry rule produced, before any outcome is known."""
    if setups.empty:
        return "  no setups"
    longs = int((setups["side"] > 0).sum())
    floored = int((setups["stop_from"] == "sigma_floor").sum())
    return "\n".join(
        [
            f"  setups                {len(setups)}",
            f"  long / short          {longs} / {len(setups) - longs}",
            f"  mean wait to entry    {setups['wait_bars'].mean():.1f} bars",
            f"  mean retracement      {setups['retrace_frac'].mean():.0%} of impulse",
            f"  stop set by sigma     {floored} ({floored / len(setups):.0%})",
        ]
    )
