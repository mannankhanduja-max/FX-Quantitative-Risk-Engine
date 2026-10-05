"""
Synthetic price paths with NO edge in them, by construction.

WHAT THIS IS FOR
-----------------
Every positive result in this study was a win rate above the
no-information benchmark

    P(target first) = b / (a + b)          in log space

which is exact for driftless Brownian motion with CONSTANT volatility.
Real volatility is not constant: it clusters. And this study's stops
are placed an ATR-scaled distance beyond the extreme of a bar that
just moved unusually far - which is to say, conditioned on high
realised volatility, which mean-reverts. So the benchmark may be
misspecified for exactly the trades it is being applied to.

The way to find out is to build a world with nothing in it and see
whether the measurement still finds something. These generators
produce log-price paths whose expected increment is EXACTLY zero at
every step. Any win rate above the benchmark, on this data, is the
benchmark being wrong. There is nothing else it could be.

TWO REGIMES, AND THE FIRST ONE IS THE CONTROL
----------------------------------------------
`constant` holds volatility fixed, so the benchmark's assumptions hold
exactly. It must come out flat. If it does not, the finding is a bug
somewhere in the pipeline and not a statement about volatility - which
is precisely why running only the interesting regime would be
worthless.

`garch` adds GARCH(1,1) clustering with persistence 0.98 and nothing
else. Same zero drift, same everything else, one difference.

WHY THE PATHS ARE BUILT FROM SUB-STEPS
---------------------------------------
The barrier walk reads High and Low, so a bar needs a real intrabar
range - a path that only has closes would understate touches and
flatter the test. Each bar is the sum of `sub` independent increments,
and its High and Low come from that sub-path. The bar return is the
sum of its own sub-steps, so the GARCH recursion is driven by the same
number the bar reports: the fine structure and the coarse structure
cannot disagree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# GARCH(1,1) at 5-minute resolution. alpha + beta = 0.98 is ordinary
# for intraday FX; the point is persistence, not a fitted match.
ALPHA = 0.08
BETA = 0.90
SIGMA_5M = 3.0e-4          # ~3bp per 5m, near EUR/USD's own scale


def garch_path(n_bars: int, seed: int, regime: str = "garch",
               sub: int = 10, s0: float = 1.1000,
               alpha: float = ALPHA, beta: float = BETA,
               sigma: float = SIGMA_5M,
               freq: str = "5min",
               start: str = "2020-01-01") -> pd.DataFrame:
    """
    A driftless 5m OHLCV frame. `regime` is "garch" or "constant".

    Zero drift is imposed in LOG space, which is the space the
    benchmark is derived in. The price level therefore drifts upward
    slightly under exponentiation, and that is correct: it is what a
    driftless log process looks like in price terms.
    """
    if regime not in ("garch", "constant"):
        raise ValueError(f"regime must be 'garch' or 'constant', got {regime!r}")
    if not 0 <= alpha + beta < 1:
        raise ValueError("alpha + beta must be in [0, 1) for a stationary "
                         "variance; otherwise the unconditional level is "
                         "undefined and the two regimes are not comparable")
    if n_bars < 2 or sub < 1:
        raise ValueError("need at least 2 bars and 1 sub-step")

    rng = np.random.default_rng(seed)
    var_uncond = sigma * sigma
    omega = var_uncond * (1.0 - alpha - beta)

    z = rng.standard_normal((n_bars, sub))
    bar_ret = np.empty(n_bars)
    step = np.empty((n_bars, sub))
    var = var_uncond
    prev = 0.0
    for i in range(n_bars):
        if regime == "garch":
            var = omega + alpha * prev * prev + beta * var
        s = np.sqrt(var / sub)
        step[i] = z[i] * s
        bar_ret[i] = step[i].sum()
        prev = bar_ret[i]

    # Level at every sub-step, and the bar opens at the level the
    # previous bar closed at, so there are no synthetic gaps.
    flat = step.reshape(-1)
    logp = np.log(s0) + np.cumsum(flat)
    grid = logp.reshape(n_bars, sub)
    open_ = np.empty(n_bars)
    open_[0] = np.log(s0)
    open_[1:] = grid[:-1, -1]

    px = np.exp(grid)
    out = pd.DataFrame({
        "Open": np.exp(open_),
        "High": np.maximum(px.max(axis=1), np.exp(open_)),
        "Low": np.minimum(px.min(axis=1), np.exp(open_)),
        "Close": px[:, -1],
        "Volume": 1.0,
    }, index=pd.date_range(start, periods=n_bars, freq=freq, tz="UTC"))
    return out


def realised_clustering(bars: pd.DataFrame, lag: int = 1) -> float:
    """
    Autocorrelation of squared log returns - the thing that separates
    the two regimes. Near zero for `constant`, clearly positive for
    `garch`. Used by the tests so the control cannot silently become
    a second copy of the treatment.
    """
    r = np.log(bars["Close"]).diff().dropna().to_numpy()
    r2 = r * r
    if len(r2) <= lag + 2:
        return float("nan")
    a, b = r2[:-lag], r2[lag:]
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
