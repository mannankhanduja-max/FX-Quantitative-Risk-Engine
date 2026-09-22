"""
What win rate should a rule with NO information produce?

Every result in this repository has been compared to the
cost-adjusted breakeven hurdle - the win rate needed to stop losing
money. That is the right bar for "is this worth trading". It is the
wrong bar for "does this know anything", because it ignores where
the stop and target actually sit on each trade.

This module supplies the other bar. Under the Black-Scholes world -
price a geometric Brownian motion, no drift, constant volatility -
the probability of touching an upper barrier before a lower one has
a closed form. That is the win rate a coin flip produces GIVEN this
rule's own barrier geometry, trade by trade. Comparing the realised
win rate against it isolates information from bracket design.

THE FORMULA
------------
Work in log space. With entry at 0, target at a = ln(target/entry)
and stop at -b = ln(entry/stop), both positive for a long, a
driftless Brownian motion hits a before -b with probability

    P(target first) = b / (a + b)

which is the classic gambler's-ruin result: the odds are the inverse
ratio of the distances. Under a drift nu = mu - sigma^2/2 it becomes

    P = (1 - exp(-2*nu*b/s2)) / (exp(2*nu*a/s2) - exp(-2*nu*b/s2))

and the driftless case is its limit as nu -> 0.

WHY ZERO DRIFT IS THE RIGHT NULL
---------------------------------
Black-Scholes prices under the risk-neutral measure, where the drift
is set by the rates differential, not by anyone's forecast. Over a
10-hour FX trade that differential moves price by a fraction of a
basis point - far below the noise. Setting it to zero is both the
honest null hypothesis ("this rule knows nothing") and numerically
indistinguishable from the risk-neutral drift at these horizons.

The drifted form is kept because it answers a different and useful
question: how much drift would a rule NEED for its realised win rate
to be fair? `implied_drift` inverts for exactly that, and expresses
it as an annualised return - which is usually the moment the size of
the claim becomes obvious.

WHAT THIS CANNOT CAPTURE, AND WHY IT MATTERS HERE
--------------------------------------------------
The formula is for a trade with no time limit: eventually one
barrier is touched. A trade closed by a bar limit touched neither,
so its outcome is not in the model's sample space at all. When time
exits are a meaningful share, compare on barrier-resolved trades
only - the same restriction `summarise_explicit` already applies to
the cost hurdle, for the same reason.

Volatility does not appear in the driftless formula. That is not an
approximation - a scaling of time does not change WHICH barrier is
hit first, only when. So this benchmark needs no volatility estimate
and cannot be wrong about one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def touch_probability(entry, target, stop, drift: float = 0.0,
                      sigma: float = 0.01) -> np.ndarray:
    """
    P(target touched before stop) under GBM, per trade.

    Works for longs and shorts: `target` above `entry` and `stop`
    below it means a long, and the mirrored arrangement a short.
    Returns NaN where the geometry is degenerate.
    """
    e = np.asarray(entry, dtype="float64")
    t = np.asarray(target, dtype="float64")
    s = np.asarray(stop, dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.abs(np.log(t / e))        # distance to target, log units
        b = np.abs(np.log(e / s))        # distance to stop

    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    out = np.full(a.shape, np.nan)

    if drift == 0.0:
        out[ok] = b[ok] / (a[ok] + b[ok])
        return out

    s2 = sigma ** 2
    nu = drift - 0.5 * s2
    # Scale function of Brownian motion with drift, S(x) = exp(-2*nu*x/s2):
    #     P(hit a before -b) = (S(0) - S(-b)) / (S(a) - S(-b))
    # Getting these two exponent signs the wrong way round inverts the
    # answer - a strong upward drift then reports a near-zero chance of
    # reaching the upper barrier. The nu -> 0 limit below is b/(a+b),
    # which is what test_positive_drift_raises_the_probability and
    # test_implied_drift_round_trips pin.
    with np.errstate(over="ignore", invalid="ignore"):
        num = 1.0 - np.exp(2.0 * nu * b / s2)
        den = np.exp(-2.0 * nu * a / s2) - np.exp(2.0 * nu * b / s2)
        out[ok] = (num[ok] / den[ok])
    return np.clip(out, 0.0, 1.0)


def implied_drift(win_rate: float, a: float, b: float, sigma: float,
                  lo: float = -5.0, hi: float = 5.0) -> float:
    """
    The annualised drift a rule's win rate implies, by bisection.

    Answers "what would have to be true about the market for this
    win rate to be fair". A number far outside anything an asset
    plausibly drifts at is a sign the win rate came from something
    other than a directional edge - a time limit, a breakeven stop,
    or luck.
    """
    if not 0.0 < win_rate < 1.0 or a <= 0 or b <= 0:
        return float("nan")

    def f(mu):
        return float(touch_probability(1.0, np.exp(a), np.exp(-b),
                                       drift=mu, sigma=sigma)) - win_rate

    flo, fhi = f(lo), f(hi)
    if not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def benchmark(trades: pd.DataFrame) -> dict:
    """
    Realised win rate against the no-information benchmark.

    `trades` is a `barriers.walk_explicit` frame. Only barrier
    exits are counted, because a time exit is outside the model's
    sample space.

    The z-statistic treats each trade as an independent Bernoulli
    draw with its own probability, which is the Poisson-binomial
    setting: the variance is the sum of p(1-p), not n*p_bar*(1-p_bar).
    """
    if trades.empty:
        return {"trades": 0}

    at_barrier = trades["outcome"] != "time"
    t = trades[at_barrier]
    if t.empty:
        return {"trades": 0}

    p = touch_probability(t["entry"].to_numpy(),
                          t["target"].to_numpy(),
                          t["stop0"].to_numpy())
    ok = np.isfinite(p)
    p = p[ok]
    wins = (t["outcome"].to_numpy()[ok] == "target").astype(float)

    n = len(p)
    expected = p.sum()
    var = float((p * (1.0 - p)).sum())
    z = (wins.sum() - expected) / np.sqrt(var) if var > 0 else np.nan

    return {
        "trades": int(n),
        "actual_win": float(wins.mean()),
        "model_win": float(p.mean()),
        "excess": float(wins.mean() - p.mean()),
        "z": float(z),
        "time_exits_excluded": int((~at_barrier).sum()),
    }
