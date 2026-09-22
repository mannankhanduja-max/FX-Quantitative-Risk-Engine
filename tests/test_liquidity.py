"""
Tests for fxrisk.strategies.liquidity.

Swing-based setups are the easiest place in this repository to leak
the future: a swing low is only a swing low once later bars confirm
it, and a backtest that uses the chart's hindsight view of swings
will report an edge that is entirely those few bars of foresight.
The first test is the one that matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.strategies import liquidity as lq


def _bars(h, l, c):
    idx = pd.date_range("2024-01-02", periods=len(c), freq="1h",
                        tz="America/New_York")
    return pd.DataFrame({"Open": c, "High": h, "Low": l, "Close": c,
                         "Volume": 1000.0}, index=idx)


def _noise(n, seed=0, base=100.0, amp=0.15):
    rng = np.random.default_rng(seed)
    c = base + np.cumsum(rng.normal(0, amp * 0.1, n))
    h = c + amp
    l = c - amp
    return list(h), list(l), list(c)


# ------------------------------------------------------------
# The leak
# ------------------------------------------------------------

def test_swings_are_confirmed_late_not_in_hindsight():
    """
    A swing low at bar j is knowable only at j+k. The value reported
    AT bar j must therefore not be bar j's own low.
    """
    h, l, c = _noise(40, seed=1)
    l[20] = 95.0                       # an obvious swing low at bar 20
    b = _bars(h, l, c)
    lo, _ = lq._swings(b, k=2)
    assert lo[20] != 95.0, "swing used before it was confirmed"
    assert lo[22] == 95.0, "swing not available once confirmed"


@pytest.mark.parametrize("fn", ["inducement_entries", "turtle_soup_entries"])
def test_entries_do_not_depend_on_the_future(fn):
    h, l, c = _noise(200, seed=2)
    b = _bars(h, l, c)
    base = getattr(lq, fn)(b)

    shocked = b.copy()
    k = shocked.index[-1]
    shocked.loc[k, "Low"] = 80.0
    shocked.loc[k, "High"] = 120.0
    after = getattr(lq, fn)(shocked)

    if base.empty and after.empty:
        pytest.skip("no entries on this fixture")
    n = min(len(base), len(after))
    cols = ["bar", "side", "entry", "stop"]
    assert base[cols].iloc[:n].equals(after[cols].iloc[:n])


# ------------------------------------------------------------
# Turtle Soup
# ------------------------------------------------------------

def _soup_fixture(gap_bars):
    """
    A 20-bar low set `gap_bars` ago, then a new low that reclaims it.

    The quiet bars oscillate by a tick rather than sitting flat: a
    dead-flat series has zero EWMA volatility, and every entry rule
    here refuses to size a stop against sigma = 0. A trailing bar is
    appended because an entry needs a following bar to walk into.
    """
    n = 28
    h = [101.0] * n
    l = [100.0] * n
    c = [100.5 + (0.01 if i % 2 else -0.01) for i in range(n)]
    ev = n - 2                          # the event bar, one from the end
    old = ev - gap_bars
    l[old] = 99.0                       # the prior extreme
    c[old] = 99.4                       # ... which does NOT reclaim, so the
    h[old] = 99.6                       #     bar that sets it takes no trade
    l[ev] = 98.5                        # the new low, undercutting it
    c[ev] = 99.4                        # closes back above the prior low
    h[ev] = 99.6
    return _bars(h, l, c)


def test_turtle_soup_needs_an_old_prior_extreme():
    cfg = lq.LiquidityConfig(soup_lookback=20, min_age=4, window=1)

    fresh = _soup_fixture(gap_bars=1)          # prior low set yesterday
    assert lq.turtle_soup_entries(fresh, cfg).empty

    aged = _soup_fixture(gap_bars=10)          # prior low ten bars old
    e = lq.turtle_soup_entries(aged, cfg)
    assert len(e) == 1
    assert e.iloc[0]["side"] == 1.0
    assert e.iloc[0]["age"] >= 4


def test_dropping_the_age_condition_admits_the_fresh_case():
    """The control: without the age rule it becomes 'fade every new low'."""
    cfg = lq.LiquidityConfig(soup_lookback=20, min_age=4, window=1)
    fresh = _soup_fixture(gap_bars=1)
    assert lq.turtle_soup_entries(fresh, cfg).empty
    assert len(lq.turtle_soup_entries(fresh, cfg, require_age=False)) == 1


def test_turtle_soup_needs_the_reclaim():
    """Undercut the level and stay under it: no trade."""
    cfg = lq.LiquidityConfig(soup_lookback=20, min_age=4, window=1)
    b = _soup_fixture(gap_bars=10).copy()
    k = b.index[-2]                    # the event bar, not the trailing one
    b.loc[k, "Close"] = 98.6           # closes BELOW the prior low
    assert lq.turtle_soup_entries(b, cfg).empty


def test_soup_stop_sits_beyond_the_new_extreme():
    cfg = lq.LiquidityConfig(soup_lookback=20, min_age=4, window=1,
                             stop_sigma=0.01)
    e = lq.turtle_soup_entries(_soup_fixture(gap_bars=10), cfg)
    assert e.iloc[0]["stop"] <= 98.5 + 1e-9


# ------------------------------------------------------------
# Inducement
# ------------------------------------------------------------

def _induce_fixture():
    """A major low far below, a confirmed swing low above it, then a
    sweep of that swing low and a reclaim - with live volatility and
    a trailing bar, for the reasons in _soup_fixture."""
    n = 42
    h = [101.0] * n
    l = [100.0] * n
    c = [100.5 + (0.01 if i % 2 else -0.01) for i in range(n)]
    ev = n - 2
    # Both levels must fall inside the 20-bar window ending at ev,
    # otherwise the rolling minimum never sees the major low and the
    # inducement ends up BEING the major level.
    l[ev - 16] = 97.0                  # the major low
    c[ev - 16] = 97.4
    l[ev - 8] = 99.0                   # the inducement, above it
    c[ev - 8] = 99.4
    l[ev] = 98.8                       # swept
    c[ev] = 99.3                       # reclaimed
    return h, l, c

def test_inducement_requires_a_major_level_beyond_it():
    """
    The whole premise is a nearer pool with a bigger one behind it.
    If the swept swing IS the lowest low around, there is no
    inducement - just a low.
    """
    h, l, c = _induce_fixture()
    b = _bars(h, l, c)
    cfg = lq.LiquidityConfig(swing_k=2, major_lookback=20, window=1)
    e = lq.inducement_entries(b, cfg)
    assert len(e) >= 1
    assert e.iloc[0]["side"] == 1.0
    assert e.iloc[0]["level"] == pytest.approx(99.0)


def test_inducement_side_is_a_fade_not_a_continuation():
    """Sweeping lows produces a LONG. Getting this backwards would
    silently convert the setup into the momentum trade that already
    failed everywhere else in this repository."""
    h, l, c = _induce_fixture()
    e = lq.inducement_entries(_bars(h, l, c),
                              lq.LiquidityConfig(major_lookback=20, window=1))
    assert (e["side"] > 0).all()
    assert (e["stop"] < e["entry"]).all()


def test_config_rejects_nonsense():
    for kw in ({"swing_k": 0}, {"major_lookback": 2}, {"soup_lookback": 2},
               {"min_age": -1}, {"window": 0}, {"stop_sigma": 0.0}):
        with pytest.raises(ValueError):
            lq.LiquidityConfig(**kw)
