"""
Inducement and Turtle Soup: two fades of a failed extreme.

Companions to `sweeps.py`. All three say the same thing in different
dialects - price reached beyond a level where orders were resting,
did not hold, and the failure is the trade. They differ in WHICH
level and in what has to happen around it, and those differences are
the only reason to test them separately.

They also sit on the side of the evidence. The signal study found no
momentum in these instruments and confirmed short-term reversion out
of sample, so a fade of a failed extension is at least pointed the
right way, unlike every breakout variant tested before it.

SWING POINTS ARE CONFIRMED LATE, AND THAT MATTERS
---------------------------------------------------
A swing low at bar i is only a swing low once `k` bars on BOTH sides
are higher. That is knowable at bar i+k, not at bar i. Backtests of
these setups routinely use swings the chart shows in hindsight and
report spectacular results; the whole edge is the k bars of
foresight. Here `_swings` returns, for every bar, the most recent
swing CONFIRMED by that bar, and `tests/test_liquidity.py` perturbs
the future to prove nothing leaks.

INDUCEMENT
-----------
The idea: before price reaches an obvious level, it first takes out
a smaller, nearer pool of stops - the "inducement" - which fills the
orders needed to move it. The trade is taken when that minor pool is
swept and immediately reclaimed.

    1  a major level      the lowest low of the last `major_lookback`
                          bars (for a long)
    2  the inducement     the most recent CONFIRMED swing low sitting
                          ABOVE that major level - the nearer pool
    3  the sweep          price trades below the inducement
    4  the reclaim        and closes back above it, within `window`
                          bars of the sweep

Entry at that close, stop beyond the sweep's low. The bearish case
mirrors it.

TURTLE SOUP
------------
Linda Raschke's setup from *Street Smarts*, implemented to its
stated conditions rather than to the general idea:

    1  a new N-bar extreme is made
    2  the PREVIOUS N-bar extreme is at least `min_age` bars old
    3  price closes back through that previous extreme

Condition 2 is what separates it from an ordinary sweep, and it is
usually the one dropped. The reasoning is that a fresh extreme has
not had time to accumulate resting orders, so the failure means
little; an old extreme is where stops have piled up, and running it
is an event. Dropping the condition turns the setup into "fade every
new high", which is a different and much weaker claim - the
`--no-age` variant in the runner measures exactly that difference.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LiquidityConfig:
    """Geometry shared by both setups."""

    swing_k: int = 2            # bars either side that define a swing
    major_lookback: int = 20    # bars whose extreme is the "major" level
    soup_lookback: int = 20     # N in "new N-bar extreme"
    min_age: int = 4            # bars the previous extreme must have stood
    window: int = 3             # bars allowed between sweep and reclaim
    stop_sigma: float = 1.0     # floor on the stop, in EWMA sigmas
    lam: float = 0.94

    def __post_init__(self) -> None:
        if self.swing_k < 1:
            raise ValueError("swing_k must be at least 1")
        if self.major_lookback < 3 or self.soup_lookback < 3:
            raise ValueError("lookbacks must be at least 3")
        if self.min_age < 0:
            raise ValueError("min_age cannot be negative")
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.stop_sigma <= 0:
            raise ValueError("stop_sigma must be positive")


def _sigma(close: pd.Series, lam: float) -> np.ndarray:
    r = np.log(close).diff()
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()).shift(1).to_numpy()


def _swings(bars: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Most recent CONFIRMED swing low and high available at each bar.

    A swing at bar j needs k higher bars on each side, so it is
    confirmed at j+k and may not be used before then. Returning the
    value indexed by the bar that can see it removes any temptation
    to index it by the bar it happened on.
    """
    high = bars["High"].to_numpy(dtype="float64")
    low = bars["Low"].to_numpy(dtype="float64")
    n = len(bars)
    swing_low = np.full(n, np.nan)
    swing_high = np.full(n, np.nan)

    last_lo, last_hi = np.nan, np.nan
    for i in range(n):
        j = i - k                       # the bar that could now be confirmed
        if j - k >= 0:
            lo_j = low[j]
            if lo_j == np.min(low[j - k:j + k + 1]) and \
                    np.sum(low[j - k:j + k + 1] == lo_j) == 1:
                last_lo = lo_j
            hi_j = high[j]
            if hi_j == np.max(high[j - k:j + k + 1]) and \
                    np.sum(high[j - k:j + k + 1] == hi_j) == 1:
                last_hi = hi_j
        swing_low[i] = last_lo
        swing_high[i] = last_hi
    return swing_low, swing_high


def _stop_distance(entry: float, structural: float, sig: float,
                   stop_sigma: float) -> float:
    return max(abs(entry - structural), stop_sigma * sig * entry)


def inducement_entries(bars: pd.DataFrame,
                       cfg: LiquidityConfig | None = None) -> pd.DataFrame:
    """
    Sweep of the nearer pool, reclaimed, with a major level beyond it.
    """
    cfg = cfg or LiquidityConfig()
    high = bars["High"].to_numpy(dtype="float64")
    low = bars["Low"].to_numpy(dtype="float64")
    close = bars["Close"].to_numpy(dtype="float64")
    sig = _sigma(bars["Close"], cfg.lam)
    sw_lo, sw_hi = _swings(bars, cfg.swing_k)

    major_lo = bars["Low"].rolling(cfg.major_lookback).min().shift(1).to_numpy()
    major_hi = bars["High"].rolling(cfg.major_lookback).max().shift(1).to_numpy()

    n = len(bars)
    rows = []
    i = cfg.major_lookback + cfg.swing_k + 1
    while i < n - 1:
        if not np.isfinite(sig[i]) or sig[i] <= 0:
            i += 1
            continue

        # Long: an inducement low ABOVE the major low is swept and reclaimed.
        ind_lo = sw_lo[i]
        took = (np.isfinite(ind_lo) and np.isfinite(major_lo[i])
                and ind_lo > major_lo[i] and low[i] < ind_lo)
        if took:
            for j in range(i, min(i + cfg.window + 1, n - 1)):
                if close[j] > ind_lo:
                    entry = close[j]
                    d = _stop_distance(entry, np.min(low[i:j + 1]), sig[j],
                                       cfg.stop_sigma)
                    rows.append({"bar": j, "side": 1.0, "entry": entry,
                                 "stop": entry - d, "level": ind_lo,
                                 "kind": "inducement"})
                    i = j
                    break

        ind_hi = sw_hi[i]
        took = (np.isfinite(ind_hi) and np.isfinite(major_hi[i])
                and ind_hi < major_hi[i] and high[i] > ind_hi)
        if took:
            for j in range(i, min(i + cfg.window + 1, n - 1)):
                if close[j] < ind_hi:
                    entry = close[j]
                    d = _stop_distance(entry, np.max(high[i:j + 1]), sig[j],
                                       cfg.stop_sigma)
                    rows.append({"bar": j, "side": -1.0, "entry": entry,
                                 "stop": entry + d, "level": ind_hi,
                                 "kind": "inducement"})
                    i = j
                    break
        i += 1

    return pd.DataFrame(rows)


def turtle_soup_entries(bars: pd.DataFrame,
                        cfg: LiquidityConfig | None = None,
                        require_age: bool = True) -> pd.DataFrame:
    """
    New N-bar extreme against an OLD prior extreme, then reclaimed.

    `require_age=False` drops Raschke's age condition, which is the
    control: it turns the setup into "fade every new extreme" and
    shows what the condition is worth.
    """
    cfg = cfg or LiquidityConfig()
    N = cfg.soup_lookback
    high = bars["High"].to_numpy(dtype="float64")
    low = bars["Low"].to_numpy(dtype="float64")
    close = bars["Close"].to_numpy(dtype="float64")
    sig = _sigma(bars["Close"], cfg.lam)
    n = len(bars)

    rows = []
    i = N + 1
    while i < n - 1:
        if not np.isfinite(sig[i]) or sig[i] <= 0:
            i += 1
            continue

        prior_lows = low[i - N:i]
        prior_low = prior_lows.min()
        # How long the previous extreme has stood, in bars.
        age_lo = N - int(np.argmin(prior_lows))
        if low[i] < prior_low and (age_lo >= cfg.min_age or not require_age):
            for j in range(i, min(i + cfg.window + 1, n - 1)):
                if close[j] > prior_low:
                    entry = close[j]
                    d = _stop_distance(entry, np.min(low[i:j + 1]), sig[j],
                                       cfg.stop_sigma)
                    rows.append({"bar": j, "side": 1.0, "entry": entry,
                                 "stop": entry - d, "level": prior_low,
                                 "age": age_lo, "kind": "turtle_soup"})
                    i = j
                    break

        prior_highs = high[i - N:i]
        prior_high = prior_highs.max()
        age_hi = N - int(np.argmax(prior_highs))
        if high[i] > prior_high and (age_hi >= cfg.min_age or not require_age):
            for j in range(i, min(i + cfg.window + 1, n - 1)):
                if close[j] < prior_high:
                    entry = close[j]
                    d = _stop_distance(entry, np.max(high[i:j + 1]), sig[j],
                                       cfg.stop_sigma)
                    rows.append({"bar": j, "side": -1.0, "entry": entry,
                                 "stop": entry + d, "level": prior_high,
                                 "age": age_hi, "kind": "turtle_soup"})
                    i = j
                    break
        i += 1

    return pd.DataFrame(rows)
