"""
Does a signal carry information? Measured before anything is traded.

THE QUESTION, AND WHY RANK IC
------------------------------
A signal's information coefficient is the rank correlation between
what it said at t and what price did over the next h bars. It is the
cleanest single answer to "is there anything here", for three reasons:

  - It needs no threshold, stop, target or position size - so it
    cannot be tuned into looking good the way a rule can.
  - Rank rather than Pearson, so one crash bar cannot manufacture it.
  - It maps to what a strategy can earn. By the fundamental law of
    active management, IR is roughly IC times the square root of the
    number of independent bets. A tiny IC can be worth having - but
    only if it is real and only if its size in basis points clears
    the cost of acting on it.

OVERLAPPING RETURNS WILL LIE TO YOU
------------------------------------
A 24-bar forward return sampled every bar shares 23 of its 24 bars
with its neighbour. Correlating a signal with that series and taking
the naive t-statistic treats 24 nearly identical observations as 24
independent ones and inflates t by up to sqrt(24), about five times.
That single mistake produces most of the spurious "signals" in
retail backtesting.

The fix used here is blocking. IC is computed separately within each
(instrument, calendar month) cell. Cells are close to independent of
one another - they share at most h bars at a month boundary - so the
t-statistic is taken across CELLS, and the overlap inside each cell
affects only how noisy that cell's IC is, not how many independent
observations the test thinks it has.

MULTIPLE COMPARISONS
---------------------
Eight features at three horizons is twenty-four tests. At a naive 5%
level, one or two "discoveries" are expected from noise alone. The
Holm step-down procedure controls the chance of even one false
discovery across the whole family, and is uniformly more powerful
than plain Bonferroni while making no extra assumptions.

SIZE, NOT JUST SIGN
--------------------
A statistically real IC can still be economically worthless. The
quintile edge answers "how many basis points per trade": within each
cell, rank the signal into fifths, and take half the difference in
mean forward return between the top and bottom fifth - the average
of going long the top and short the bottom, per trade. It is compared
to the round trip actually paid. An IC of 0.01 with t = 4 on a million
bars is real and cannot pay a 2bp spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def cell_ics(feature: pd.Series, fwd: pd.Series, instrument: str,
             min_obs: int = 60) -> pd.DataFrame:
    """
    Rank IC and quintile edge per calendar month for one instrument.

    Months with fewer than `min_obs` usable pairs are dropped rather
    than kept with a noisy estimate - a 12-observation Spearman
    correlation is almost pure noise and would dilute the t-statistic
    across cells without adding information.
    """
    df = pd.DataFrame({"x": feature, "y": fwd}).dropna()
    if df.empty:
        return pd.DataFrame()
    # Months are cut in UTC. The timezone is dropped only after that
    # conversion, so a bar's month cannot shift with the host's clock.
    ix = df.index.tz_convert("UTC").tz_localize(None) if df.index.tz else df.index
    month = ix.to_period("M")

    rows = []
    for m, g in df.groupby(month):
        if len(g) < min_obs or g["x"].nunique() < 5:
            continue
        ic = stats.spearmanr(g["x"], g["y"]).statistic
        q = pd.qcut(g["x"].rank(method="first"), 5, labels=False)
        top = g["y"][q == 4].mean()
        bot = g["y"][q == 0].mean()
        rows.append({"instrument": instrument, "month": str(m), "n": len(g),
                     "ic": float(ic),
                     "edge_bp": float((top - bot) / 2.0 * 10_000)})
    return pd.DataFrame(rows)


@dataclass
class ICResult:
    feature: str
    horizon: int
    cells: int
    mean_ic: float
    t: float
    p: float
    hit: float           # share of cells whose IC has the pooled sign
    edge_bp: float       # mean quintile edge per trade, basis points


def summarise(cells: pd.DataFrame, feature: str, horizon: int) -> ICResult:
    """Pool cells into one test: is the mean IC across cells non-zero?"""
    if cells.empty or len(cells) < 3:
        return ICResult(feature, horizon, len(cells), np.nan, np.nan,
                        np.nan, np.nan, np.nan)
    ic = cells["ic"].to_numpy(dtype="float64")
    n = len(ic)
    mean = ic.mean()
    sd = ic.std(ddof=1)
    t = mean / (sd / np.sqrt(n)) if sd > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan
    hit = float((np.sign(ic) == np.sign(mean)).mean()) if mean != 0 else 0.5
    return ICResult(feature, horizon, n, float(mean), float(t), p, hit,
                    float(cells["edge_bp"].mean()))


def holm(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Holm-Bonferroni step-down. Returns reject/keep in the input order.

    Sort p ascending; reject the k-th smallest while p_(k) <= alpha /
    (m - k + 1), and stop at the first failure - everything after it is
    kept, even if its own p would clear a looser bar. NaNs are never
    rejected and do not count towards m.
    """
    idx = [i for i, p in enumerate(pvalues) if p is not None and np.isfinite(p)]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvalues[i])
    reject = [False] * len(pvalues)
    for k, i in enumerate(order):
        if pvalues[i] <= alpha / (m - k):
            reject[i] = True
        else:
            break
    return reject
