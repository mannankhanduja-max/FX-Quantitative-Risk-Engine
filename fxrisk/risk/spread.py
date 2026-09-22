"""
Estimating the spread you actually pay, bar by bar.

WHY A FLAT COST WAS ALWAYS A LIE
---------------------------------
Every backtest in this repository so far charged a FLAT cost per
side - `BREAKOUT_COST_BP = 1.0` - to every trade regardless of
when it happened. That is not a conservative simplification. It
is an optimistic one, and in a specific direction.

The spread is not a constant. It is narrowest when the market it
belongs to is open and busy, and it widens - by a factor of two
to ten, depending on the instrument - at three predictable times:

  1  THE ROLLOVER, around 17:00 New York. Liquidity providers
     step away while the value date rolls. Spot FX spreads that
     sit at 0.1 pip in London routinely print 2-5 pips here, and
     gold is worse.

  2  SCHEDULED NEWS. A market maker facing a number it cannot
     predict widens or withdraws before the print and reprices
     after it. The blowout starts BEFORE the release, not at it.

  3  THIN HOURS generally - any time the instrument's own centre
     is shut.

A strategy that trades 1,800 times a year is enormously exposed
to this, and a flat cost hides the exposure completely. Worse, a
breakout or sweep rule is ACTIVELY DRAWN to the wide-spread
moments: a news print is exactly what pierces a 20-bar high, so
the trades the rule likes most are disproportionately the trades
that cost most. Charging them the London rate flatters the result.

So this module does two separate things, and the distinction
matters:

    `corwin_schultz`  ESTIMATES what the spread actually was,
                      so the backtest can charge it. This makes
                      results WORSE and more honest.

    `blackout`        REFUSES to trade at the times it is known
                      to be wide. This is a rule, not a fit.

Doing only the second without the first would be cheating: you
would drop the expensive trades while still pricing the rest at a
flat rate, and book the improvement twice.

CORWIN-SCHULTZ: IMPLEMENTED, MEASURED, AND REJECTED HERE
---------------------------------------------------------
The obvious tool is Corwin & Schultz (2012), "A Simple Way to
Estimate Bid-Ask Spreads from Daily High and Low Prices", Journal
of Finance 67(2). A bar's high-low range contains the variance of
the true price plus the spread; variance scales with time and the
spread does not, so comparing one bar's range against two bars
combined separates them.

It is implemented below as `corwin_schultz`, and on five-minute
bars it DOES NOT MEASURE THE SPREAD. Its estimate by hour of the
New York day, against the tick count in the same bars:

    EUR/USD      10:00 ET        17:00 ET (the rollover)
    ticks/bar      925              68
    C-S estimate   0.75 bp          0.22 bp

The rollover is the widest routine spread of the day, and the
estimator calls it the narrowest. It is tracking VOLATILITY, not
spread: at this frequency the high-low range is dominated by real
price movement rather than by the bid-ask bounce the paper's
identification depends on. The estimator is designed for daily
bars, and used at five minutes it inverts. Trusting it would have
charged the cheapest cost exactly where the true cost is highest -
which is worse than the flat rate it was meant to replace.

It stays in the module because the negative result is worth
keeping and because it is correct at daily frequency, where
`histdata.py` bars could use it.

WHAT IS USED INSTEAD: ACTIVITY
-------------------------------
Tick count per bar. In FX microstructure the quoted spread is
roughly inverse to quoting activity, and unlike the range-based
estimator it has the right SHAPE at this frequency, on every
instrument here:

    EUR/USD   925 ticks at 10:00 ET   ->   68 at 17:00     13.6x
    USD/JPY  1813 ticks at 10:00 ET   ->  111 at 17:00     16.3x
    XAU/USD  0.35 lots at 10:00 ET    -> 0.08 at 16:00      4.4x
    NAS100   0.23 lots at 10:00 ET    -> 0.04 at 18:00      5.8x

Normalising by each instrument's own median makes the differing
units irrelevant - gold and the index report lots, not tick
counts, so only the ratio is meaningful.

This is a PROXY and is not the spread. The real fix is one
download away: Dukascopy serves the ask side as well as the bid,
and ask minus bid is the spread, measured rather than inferred.
`fetch_intraday.py --side ask` gets it. Until then the honest
description of what follows is "cost shaped like inverse
activity, with the same median as the flat rate", not "the
spread".

BACKTEST-ONLY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxrisk.data.intraday import session_id

_K = 3 - 2 * np.sqrt(2)


def corwin_schultz(bars: pd.DataFrame, within_session: bool = True) -> pd.Series:
    """
    Proportional spread estimate per bar, from its own and the
    previous bar's high-low range.

    `within_session` blanks the first bar of each session, where
    the pair straddles a market closure and the combined range is
    a gap rather than a spread.
    """
    high = pd.to_numeric(bars["High"], errors="coerce")
    low = pd.to_numeric(bars["Low"], errors="coerce")
    if (low <= 0).any() or (high <= 0).any():
        raise ValueError("non-positive prices; the estimator is on logs")

    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)

    hi2 = np.maximum(high, high.shift(1))
    lo2 = np.minimum(low, low.shift(1))
    gamma = np.log(hi2 / lo2) ** 2

    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    s = s.clip(lower=0.0)                       # the paper's own rule

    if within_session:
        sess = pd.Series(session_id(bars.index).to_numpy(), index=bars.index)
        s[sess != sess.shift(1)] = np.nan

    return s.rename("spread")


def smoothed(bars: pd.DataFrame, window: int = 12,
             within_session: bool = True) -> pd.Series:
    """
    The estimator is noisy bar to bar; the median of a short
    trailing window is what a cost model should use. Trailing and
    closed on the left, so a bar never prices itself.
    """
    raw = corwin_schultz(bars, within_session=within_session)
    return raw.rolling(window, min_periods=max(2, window // 3)).median().shift(1)


def activity(bars: pd.DataFrame, window: int = 12) -> pd.Series:
    """Trailing median tick count, as a fraction of the instrument's own."""
    v = pd.to_numeric(bars["Volume"], errors="coerce")
    if not (v > 0).any():
        raise ValueError("no positive volume; the activity proxy is undefined")
    med = v.median()
    return (v.rolling(window, min_periods=max(2, window // 3))
             .median().shift(1) / med).rename("activity")


def cost_bp_series(bars: pd.DataFrame, base_bp: float = 1.0,
                   window: int = 12, cap: float = 10.0,
                   power: float = 0.5) -> pd.Series:
    """
    Per-bar cost in basis points per side, shaped by inverse
    activity and rescaled so the MEDIAN bar pays `base_bp`.

    Rescaling to the median is deliberate. It changes the shape of
    the cost through the day without moving its overall level, so
    a result here stays comparable to every flat-cost result in
    the repository. Taking a proxy's absolute level at face value
    would change the cost base and the trade population at the
    same time, and no comparison would survive it.

    `power` is the elasticity of spread to activity. 0.5 - the
    spread doubling when activity falls fourfold - is the
    conservative reading of the microstructure literature, which
    puts it between 0.3 and 1.0. It is not fitted to anything
    here, and `--cost-power` in the runner shows what the choice
    is worth.

    `cap` bounds the multiple. The activity tail is heavy and one
    dead holiday bar would otherwise dominate an average.
    """
    if base_bp <= 0:
        raise ValueError("base_bp must be positive")
    if cap < 1:
        raise ValueError("cap must be at least 1")
    if not 0 < power <= 2:
        raise ValueError("power must be in (0, 2]")

    a = activity(bars, window=window)
    mult = (1.0 / a.clip(lower=1e-9)) ** power
    mult = mult.clip(lower=1.0 / cap, upper=cap)
    # Renormalise so the median bar pays exactly base_bp.
    med = mult.median()
    if np.isfinite(med) and med > 0:
        mult = mult / med
    return (base_bp * mult.clip(upper=cap)).fillna(base_bp).rename("cost_bp")


# ============================================================
# BLACKOUTS
# ============================================================
#
# Times to refuse a trade outright, as clock rules that could be
# written down before seeing any result - which is what keeps
# them out of the fitted-parameter category.
#
# ROLLOVER. 17:00 New York is the value-date roll and the widest
# routine spread of the day. The window opens before it because
# providers pull back ahead of the roll, not at it.
#
# SCHEDULED RELEASES. Without a calendar file, the recurring
# clock slots are what can be excluded honestly:
#
#     08:30 ET   US CPI, PPI, jobless claims, and NFP on the
#                first Friday - the single most spread-hostile
#                minute of the month
#     10:00 ET   ISM, consumer confidence, JOLTS
#     14:00 ET   FOMC statement days
#
# These are slots where a release is COMMON, not dates when one
# occurred; excluding the slot on a quiet Tuesday costs some good
# trades. That is the honest version. A real calendar would be
# better and is the obvious next step.

ROLLOVER = (16, 45, 18, 0)          # 16:45-18:00 ET

RELEASE_SLOTS = [
    (8, 30),
    (10, 0),
    (14, 0),
]
RELEASE_BEFORE_MIN = 10             # providers widen ahead of the print
RELEASE_AFTER_MIN = 20


def blackout(index: pd.DatetimeIndex, rollover: bool = True,
             releases: bool = True) -> pd.Series:
    """
    True where a trade should be REFUSED on spread grounds.

    Returns a mask to exclude, not to keep - so the caller reads
    `entries[~blackout(...)]` and the sense is hard to get wrong.
    """
    idx = pd.DatetimeIndex(index)
    et = idx.tz_convert("America/New_York")
    mins = et.hour * 60 + et.minute
    out = np.zeros(len(idx), dtype=bool)

    if rollover:
        h0, m0, h1, m1 = ROLLOVER
        out |= (mins >= h0 * 60 + m0) & (mins < h1 * 60 + m1)

    if releases:
        for h, m in RELEASE_SLOTS:
            t = h * 60 + m
            out |= (mins >= t - RELEASE_BEFORE_MIN) & (mins <= t + RELEASE_AFTER_MIN)

    return pd.Series(out, index=idx, name="blackout")


def wide_spread(bars: pd.DataFrame, multiple: float = 3.0,
                window: int = 12, trailing: int = 2000) -> pd.Series:
    """
    True where the estimated spread is more than `multiple` times
    its own TRAILING median.

    Trailing, not full-sample: a full-sample median is computed
    from data the trade could not have seen, and every threshold
    in this repository is trailing for that reason.
    """
    if multiple <= 1:
        raise ValueError("multiple must exceed 1")
    s = smoothed(bars, window=window)
    med = s.rolling(trailing, min_periods=200).median()
    return ((s > multiple * med) & med.notna()).rename("wide_spread")
