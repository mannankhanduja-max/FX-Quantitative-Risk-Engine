"""
Breakout, retracement INTO A FAIR VALUE GAP, entry on resumption.

This is `fxrisk.strategies.breakout_retrace` with one gate added: the
retracement has to touch a live, causally-known fair value gap on the
correct side, not merely fall inside a depth band. See
`fxrisk.strategies.smc` for why a location condition is strictly
stronger than a depth condition, and `fxrisk.strategies.mtf.setups_30m`
for the only place this combination has been tested before (the 30m
leg of the multi-timeframe cascade). This module runs the same idea as
its own, standalone, single-timeframe rule, meant for 1-minute bars.

THE RULE, IN ORDER
-------------------
    1  TREND      the 9-period EMA of the session VWAP is above the
                  VWAP (long bias) or below it (short bias)

    2  BREAKOUT   a bar CLOSES beyond the highest high (or lowest
                  low) of the previous `lookback` bars, in the same
                  direction as the bias

    3  RETRACE    within `retrace_max_bars`, price pulls back
                  between `min_fraction` and `max_fraction` of the
                  breakout impulse AND the retracement bar's range
                  touches a fair value gap that was already known
                  (formed and closed) before that bar, on the same
                  side as the trade

    4  ENTRY      the first bar that closes back beyond the
                  breakout bar's extreme, in the original
                  direction. Fill at that close.

    5  BRACKET    stop at the retracement extreme (floored at
                  `stop_min_sigma` EWMA sigmas), target at 2x that
                  distance, stop to breakeven once price has
                  travelled the instrument's breakeven trigger.

Longs and shorts are symmetric.

CAUSALITY
----------
The zone frame is computed once for the whole series
(`smc.fair_value_gaps`), and every zone carries `known_bar`: the bar at
whose CLOSE the gap first exists. `smc.in_zone` only counts a zone with
`known_bar < bar`, so a gap cannot confirm a retracement on the same
bar it was formed on - the same one-bar-leak guard `smc.py` documents
and `tests/test_smc.py` enforces by perturbing the future.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxrisk.data.intraday import session_id
from fxrisk.strategies.smc import SMCConfig, fair_value_gaps, in_zone


@dataclass
class FVGRetraceConfig:
    """Everything the entry rule can be argued about."""

    lookback: int = 20
    ema_span: int = 9
    retrace_max_bars: int = 12
    min_fraction: float = 0.33
    max_fraction: float = 1.0
    stop_min_sigma: float = 1.0
    ewma_lambda: float = 0.94
    use_trend_filter: bool = True
    smc: SMCConfig = None  # type: ignore[assignment]

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
        if self.smc is None:
            self.smc = SMCConfig()


def session_vwap(bars: pd.DataFrame, volume_col: str = "Volume") -> pd.Series:
    """Session-anchored VWAP. Identical to `breakout_retrace.session_vwap`."""
    if volume_col not in bars.columns:
        raise ValueError(f"no '{volume_col}' column - VWAP needs volume")
    vol = pd.to_numeric(bars[volume_col], errors="coerce").fillna(0.0)
    if (vol <= 0).all():
        raise ValueError(
            "volume is zero on every bar, so VWAP is undefined. Dukascopy "
            "reports a tick count per bar for spot FX and metals; this "
            "bars frame has none. Re-check the fetch."
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


def context(bars: pd.DataFrame, cfg: FVGRetraceConfig) -> pd.DataFrame:
    """Everything the entry loop needs, as columns. See breakout_retrace.context."""
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


def find_setups(bars: pd.DataFrame,
                cfg: FVGRetraceConfig | None = None,
                zone_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Walk the bars and return one row per entry the rule produces.

    `zone_frame` is `smc.fair_value_gaps(bars, cfg.smc)` if not given -
    passed in explicitly lets a caller compute it once and reuse it
    across a parameter sweep, since the zones themselves do not depend
    on `lookback`/`retrace_max_bars`/etc.

    Same overlap caveat as `breakout_retrace.find_setups`: this
    function can emit entries that sit on top of one another in a
    trend; `fxrisk.risk.barriers.walk_explicit` drops the overlapping
    ones and reports the count.
    """
    cfg = cfg or FVGRetraceConfig()
    zone_frame = (fair_value_gaps(bars, cfg.smc) if zone_frame is None
                 else zone_frame)
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

        near = level + side * impulse * (1.0 - cfg.min_fraction)
        far = level + side * impulse * (1.0 - cfg.max_fraction)

        bo_extreme = high[i] if up else low[i]
        retrace_extreme = bo_extreme
        armed = False
        zone_touch_bar = None
        entered = False

        for j in range(i + 1, min(i + 1 + cfg.retrace_max_bars, n)):
            if sess[j] != sess[i]:
                break

            if not armed:
                reach = low[j] if up else high[j]
                retrace_extreme = (
                    min(retrace_extreme, reach) if up else max(retrace_extreme, reach)
                )
                beyond = reach < far if up else reach > far
                if beyond:
                    break
                touched = reach <= near if up else reach >= near
                if touched and in_zone(zone_frame, j, side, low[j], high[j], cfg.smc):
                    armed = True
                    zone_touch_bar = j
                continue

            resumed = close[j] > bo_extreme if up else close[j] < bo_extreme
            if not resumed:
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
                    "zone_touch_bar": zone_touch_bar,
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
